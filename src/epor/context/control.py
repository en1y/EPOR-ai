"""Adapter between allowlisted context-laboratory control jobs and :mod:`epor.context`.

Four job types run here, each the twin of CLI commands and gated by the action
of the same name: ``context_extend`` (extension or sparse retrofit),
``context_evaluate`` (G6), ``context_profile`` (host profile or feasibility
search), and ``context_sweep``.  Projections and certification are not jobs:
a projection informs a human approval and a certificate raises a validated
context, so both stay on the owner's CLI.

Checkpoints are read only from beneath the artifact root and recipes only
from ``configs/models/``.  Results carry report IDs and small summaries whose
keys avoid the credential pattern (timings and throughput stay in the stored
report's measured section).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from epor.control.security import contained_path

_OUTPUT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def _roots(context: Any) -> tuple[Path, Path, Path]:
    settings = context.service.settings
    return (
        Path(settings.project_root).resolve(),
        Path(settings.artifact_root).resolve(),
        Path(settings.data_root).resolve(),
    )


def _checkpoint(artifact_root: Path, supplied: str) -> Path:
    from epor.training.run_checkpoint import resolve_checkpoint

    directory = contained_path(artifact_root, supplied, must_exist=True)
    if not directory.is_dir():
        raise FileNotFoundError(f"not a run or checkpoint directory: {supplied}")
    checkpoint = resolve_checkpoint(directory)
    contained_path(artifact_root, checkpoint, must_exist=True)
    return checkpoint


def _cancel(context: Any) -> Any:
    return lambda: bool(context.cancelled)


def _run_extend(spec: dict[str, Any], context: Any) -> dict[str, Any]:
    from epor.alignment.model import checkpoint_identity
    from epor.models.config import SparseAttention, model_config_from_mapping
    from epor.training import OptimizerSettings, PackedTrainingSettings
    from epor.training.control import memory_budget_bytes
    from epor.training.kernel import CheckpointRef, resolve_model_config, trained_context
    from epor.training.loop import TrainingCancelled
    from epor.training.run_checkpoint import latest_checkpoint, read_kernel_checkpoint
    from epor.training.runner import run_kernel

    from .extension import extension_config, retrofit_config

    project_root, artifact_root, data_root = _roots(context)
    parent_dir = _checkpoint(artifact_root, str(spec["checkpoint_path"]))
    trainer = read_kernel_checkpoint(parent_dir)
    parent = model_config_from_mapping(trainer["model_config"])
    trained = trained_context(trainer)
    if spec["mode"] == "extend":
        model = extension_config(
            parent,
            int(spec["target_context"]),
            trained_context=trained,
            method=spec.get("method", "yarn"),
        )
        length = int(spec.get("seq_len") or spec["target_context"])
    else:
        sparse = SparseAttention(
            index_heads=int(spec.get("index_heads", 2)),
            index_dim=int(spec.get("index_dim", 16)),
            top_k=int(spec.get("top_k", 64)),
            select=bool(spec.get("select", False)),
            kl_weight=float(spec.get("kl_weight", 1.0)),
        )
        model = retrofit_config(parent.model_copy(update={"sparse_attention": None}), sparse)
        length = int(spec.get("seq_len") or trained)
    model = resolve_model_config(model)
    name = str(spec.get("output_name", spec["mode"]))
    if not _OUTPUT_NAME.fullmatch(name):
        raise ValueError("output_name is not a safe artifact directory name")
    output_dir = contained_path(artifact_root, Path(context.job_id) / name)
    max_steps = int(spec.get("max_steps", 50))
    settings = PackedTrainingSettings(
        tokenized_id=PackedTrainingSettings.model_validate(trainer["settings"]).tokenized_id,
        init_checkpoint=CheckpointRef(
            path=str(parent_dir), checkpoint_sha256=checkpoint_identity(parent_dir)
        ),
        seq_len=length,
        tokens_per_batch=int(spec.get("tokens_per_batch") or length),
        min_document_pairs=spec.get("min_document_pairs"),
        precision=spec.get("precision", "bf16"),
        checkpoint_every=int(spec.get("checkpoint_every", 50)),
        keep_checkpoints=2,
        optimizer=OptimizerSettings(
            learning_rate=float(spec.get("learning_rate", 3e-4)),
            warmup_steps=int(spec.get("warmup_steps", 0)),
            total_steps=max_steps,
        ),
    )
    context.raise_if_cancelled()

    def on_step(step: int, total: int, loss: float) -> None:
        if context.cancelled:
            return
        context.progress(
            0.02 + 0.93 * step / total,
            message=f"Completed step {step} of {total}",
            details={"step": step, "max_steps": total, "loss": loss},
        )

    summary: dict[str, Any] | None = None
    try:
        summary = run_kernel(
            model,
            data_root,
            output_dir,
            settings,
            max_steps=max_steps,
            project_root=project_root,
            should_cancel=_cancel(context),
            on_step=on_step,
            memory_budget_bytes=memory_budget_bytes(spec),
        )
    except TrainingCancelled:
        context.raise_if_cancelled()
        raise
    finally:
        for file_name, kind, media in (
            ("manifest.json", "run-manifest", "application/json"),
            ("metrics.jsonl", "training-metrics", "application/x-ndjson"),
        ):
            if (output_dir / file_name).is_file():
                context.register_artifact(output_dir / file_name, kind=kind, media_type=media)
        latest = latest_checkpoint(output_dir) if output_dir.is_dir() else None
        if latest is not None:
            context.register_artifact(
                latest / "checkpoint.json",
                kind="checkpoint-manifest",
                media_type="application/json",
            )
    assert summary is not None
    context.progress(1.0, message="Context run complete")
    return {
        "run_id": summary["run_id"],
        "mode": spec["mode"],
        "steps_completed": summary["steps_completed"],
        "final_loss": summary["final_loss"],
        "configured_context": model.configured_max_context,
        "trained_context": max(trained, length),
        "validated_context_raised": False,
        "output_path": str(output_dir.relative_to(artifact_root)),
        "checkpoint_path": str(Path(summary["checkpoint_path"]).relative_to(artifact_root)),
        "scope": "proxy-scale local training; training at a length validates nothing",
    }


def _run_evaluate(spec: dict[str, Any], context: Any) -> dict[str, Any]:
    from epor.data.store import DataStore
    from epor.evaluation.decoder import load_kernel_scoring_model
    from epor.evaluation.report import store_report

    from .g6 import run_g6

    _, artifact_root, data_root = _roots(context)
    checkpoint = _checkpoint(artifact_root, str(spec["checkpoint_path"]))
    model = load_kernel_scoring_model(
        checkpoint, data_root, batch_size=int(spec.get("batch_size", 4))
    )
    context.raise_if_cancelled()
    context.progress(0.05, message="Scoring generated G6 items")
    report = run_g6(
        model,
        lengths=sorted(set(spec["lengths"])),
        items_per_cell=int(spec.get("items_per_cell", 2)),
        seed=int(spec.get("seed", 0)),
        perplexity=(data_root, model.tokenized_id, spec.get("split", "validation")),
        should_cancel=_cancel(context),
    )
    store_report(DataStore(data_root), report)
    context.progress(1.0, message="G6 report stored")
    return {
        "report_id": report.report_id,
        "lengths": sorted(set(spec["lengths"])),
        "accuracy": {
            task.task_id: task.metric("accuracy").value
            for task in report.tasks
            if task.task_type == "context-retrieval"
        },
        "scope": "scoring alone never raises a validated context",
    }


def _run_profile(spec: dict[str, Any], context: Any) -> dict[str, Any]:
    from epor.alignment.model import checkpoint_identity
    from epor.models.config import ModelConfig, load_model_config
    from epor.scaling.report import store_scaling_report

    from .probe import (
        default_memory_budget,
        feasibility_report,
        feasible_maximum,
        profile_checkpoint,
        profile_report,
    )

    project_root, artifact_root, data_root = _roots(context)
    budget = (
        int(float(spec["memory_budget_gib"]) * 1024**3)
        if spec.get("memory_budget_gib")
        else default_memory_budget()
    )
    timeout = float(spec.get("timeout_seconds", 300.0))

    def check(*_args: object) -> None:
        context.raise_if_cancelled()

    if spec["mode"] == "profile":
        checkpoint = _checkpoint(artifact_root, str(spec["checkpoint_path"]))
        config, records = profile_checkpoint(
            checkpoint,
            sorted(set(spec["lengths"])),
            quantize=spec.get("quantize", "none"),
            decode_tokens=int(spec.get("decode_tokens", 16)),
            memory_budget_bytes=budget,
            timeout_seconds=timeout,
            on_length=check,
        )
        report = profile_report(
            config, checkpoint_identity(checkpoint), records, quantize=spec.get("quantize", "none")
        )
        store_scaling_report(data_root, report)
        return {
            "report_id": report.report_id,
            "statuses": {str(r["length"]): r["status"] for r in report.results["lengths"]},
        }
    recipe = contained_path(project_root, str(spec["model_config"]), must_exist=True)
    base = load_model_config(recipe).model_dump(mode="json")
    if spec.get("vocab_size"):
        base["vocab_size"] = int(spec["vocab_size"])
    overrides = {
        "full": {},
        "local": {
            "sliding_window": int(spec.get("window", 1024)),
            "local_layers_per_global": int(spec.get("ratio", 1)),
        },
        "mla": {
            "attention_variant": "mla",
            "mla_kv_rank": int(spec.get("mla_rank", 32)),
            "mla_rope_dim": int(spec.get("mla_rope_dim", 8)),
        },
    }
    maximum, value, lengths = int(spec["max_length"]), int(spec.get("min_length", 1024)), []
    while value <= maximum:
        lengths.append(value)
        value *= 2
    variants = {
        name: ModelConfig.model_validate(
            {**base, **overrides[name], "configured_max_context": maximum}
        )
        for name in dict.fromkeys(spec["variants"])
    }
    results = feasible_maximum(
        variants, lengths, memory_budget_bytes=budget, timeout_seconds=timeout, on_point=check
    )
    report = feasibility_report(results, budget=budget, timeout_seconds=timeout)
    store_scaling_report(data_root, report)
    return {
        "report_id": report.report_id,
        "feasible_maximum": {
            name: {mode: results[name][mode]["max_ok_length"] for mode in ("train", "inference")}
            for name in results
        },
    }


def _run_sweep(spec: dict[str, Any], context: Any) -> dict[str, Any]:
    from epor.scaling.report import store_scaling_report
    from epor.training.runner import derive_kernel_model

    from .sweep import ContextSweepPlan, run_context_sweep

    project_root, artifact_root, data_root = _roots(context)
    recipe = contained_path(project_root, str(spec["model_config"]), must_exist=True)
    base = derive_kernel_model(
        recipe,
        data_root=data_root,
        tokenized_id=str(spec["tokenized_id"]),
        vocab_from_build=bool(spec.get("vocab_from_build", False)),
    )
    plan = ContextSweepPlan.model_validate(
        {
            key: spec[key]
            for key in (
                "axis",
                "values",
                "window",
                "ratio",
                "tokenized_id",
                "seq_len",
                "tokens_per_batch",
                "max_steps",
                "learning_rate",
                "eval_length",
                "items_per_cell",
            )
            if key in spec
        }
    )
    name = str(spec.get("output_name", "context-sweep"))
    if not _OUTPUT_NAME.fullmatch(name):
        raise ValueError("output_name is not a safe artifact directory name")

    def on_point(point: str) -> None:
        context.raise_if_cancelled()
        context.log(f"context sweep point {point}")

    report = run_context_sweep(
        base,
        plan,
        data_root=data_root,
        output_dir=contained_path(artifact_root, Path(context.job_id) / name),
        should_cancel=_cancel(context),
        on_point=on_point,
    )
    store_scaling_report(data_root, report)
    return {
        "report_id": report.report_id,
        "axis": plan.axis,
        "points": len(report.results["points"]),
    }


_HANDLERS = {
    "context_extend": _run_extend,
    "context_evaluate": _run_evaluate,
    "context_profile": _run_profile,
    "context_sweep": _run_sweep,
}


def run_context_job(name: str, spec: dict[str, Any], context: Any) -> dict[str, Any]:
    """Execute one allowlisted context job; never a shell command, projection, or certificate."""

    handler = _HANDLERS.get(name)
    if handler is None:
        raise ValueError(f"unsupported context control job: {name}")
    try:
        return handler(spec, context)
    except InterruptedError:
        context.raise_if_cancelled()
        raise


__all__ = ["run_context_job"]
