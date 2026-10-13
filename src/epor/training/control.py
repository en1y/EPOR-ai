"""Adapter between allowlisted control jobs and reference training functions."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from epor.reference_policy import load_reference_model_config, validate_reference_steps

from .loop import TrainingCancelled, evaluate, pretrain

_OUTPUT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def _contained(root: Path, supplied: str | Path, *, must_exist: bool) -> Path:
    root = root.resolve()
    candidate = Path(supplied)
    resolved = (candidate if candidate.is_absolute() else root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"path must remain beneath {root}") from exc
    if must_exist and not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def _register_if_present(
    context: Any,
    path: Path,
    *,
    kind: str,
    media_type: str,
) -> None:
    if path.is_file():
        context.register_artifact(path, kind=kind, media_type=media_type)


def _run_train(spec: dict[str, Any], context: Any) -> dict[str, Any]:
    settings = context.service.settings
    project_root = Path(settings.project_root).resolve()
    artifact_root = Path(settings.artifact_root).resolve()
    config_path = _contained(
        project_root,
        str(spec.get("model_config", spec.get("config_path", "configs/models/epor-tiny.yaml"))),
        must_exist=True,
    )
    load_reference_model_config(config_path)
    corpus_path = _contained(
        project_root,
        str(spec.get("corpus_path", "fixtures/tiny_corpus.txt")),
        must_exist=True,
    )
    output_name = str(spec.get("output_name", "tiny-train"))
    if not _OUTPUT_NAME.fullmatch(output_name):
        raise ValueError("output_name is not a safe artifact directory name")
    output_dir = _contained(
        artifact_root,
        Path(context.job_id) / output_name,
        must_exist=False,
    )
    max_steps = int(spec.get("max_steps", 20))
    seed = int(spec.get("seed", 1337))
    validate_reference_steps(max_steps)
    if not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed must be an unsigned 32-bit integer")

    context.raise_if_cancelled()
    context.log(f"Starting allowlisted tiny training for {max_steps} steps")
    context.progress(0.05, message="Validated tiny training inputs")

    def on_step(step: int, total: int, loss: float) -> None:
        # Avoid context.progress's deliberate cancellation exception until the
        # trainer has persisted its cancellation-boundary checkpoint.
        if context.cancelled:
            return
        context.progress(
            0.05 + 0.9 * step / total,
            message=f"Completed tiny training step {step} of {total}",
            details={"step": step, "max_steps": total, "loss": loss},
        )

    result = None
    try:
        result = pretrain(
            config_path,
            output_dir,
            max_steps=max_steps,
            corpus_path=corpus_path,
            model_seed=seed,
            data_seed=seed,
            checkpoint_every=max(1, min(10, max_steps)),
            project_root=project_root,
            should_cancel=lambda: bool(context.cancelled),
            on_step=on_step,
        )
    except TrainingCancelled:
        context.raise_if_cancelled()
        raise
    finally:
        manifest_path = result.manifest_path if result is not None else output_dir / "manifest.json"
        _register_if_present(
            context,
            manifest_path,
            kind="run-manifest",
            media_type="application/json",
        )
        _register_if_present(
            context,
            output_dir / "metrics.jsonl",
            kind="training-metrics",
            media_type="application/x-ndjson",
        )
        checkpoints = sorted((output_dir / "checkpoints").rglob("*.pt"))
        if checkpoints:
            _register_if_present(
                context,
                checkpoints[-1],
                kind="checkpoint",
                media_type="application/octet-stream",
            )

    assert result is not None
    context.progress(1.0, message="Tiny training complete")
    return {
        "run_id": result.run_id,
        "steps_completed": result.steps_completed,
        "checkpoint_path": str(result.checkpoint_path.relative_to(artifact_root)),
        "manifest_path": str(result.manifest_path.relative_to(artifact_root)),
        "final_loss": result.losses[-1] if result.losses else None,
    }


def _run_eval(spec: dict[str, Any], context: Any) -> dict[str, Any]:
    settings = context.service.settings
    project_root = Path(settings.project_root).resolve()
    artifact_root = Path(settings.artifact_root).resolve()
    checkpoint_path = _contained(artifact_root, str(spec["checkpoint_path"]), must_exist=True)
    corpus_path = _contained(
        project_root,
        str(spec.get("corpus_path", "fixtures/tiny_corpus.txt")),
        must_exist=True,
    )
    supplied_config = spec.get("model_config", spec.get("config_path"))
    if supplied_config is not None:
        # The checkpoint is self-describing, but an optional UI-supplied config
        # path is still resolved and contained rather than silently ignored.
        _contained(project_root, str(supplied_config), must_exist=True)
    report_dir = _contained(artifact_root, Path(context.job_id) / "tiny-eval", must_exist=False)
    context.raise_if_cancelled()
    context.progress(0.1, message="Loading tiny evaluation checkpoint")

    def on_batch(batch: int) -> None:
        if context.cancelled:
            return
        # The total is corpus-dependent, so report bounded asymptotic progress.
        context.progress(
            min(0.9, 0.1 + 0.8 * batch / (batch + 1)),
            message=f"Evaluated batch {batch}",
            details={"batch": batch},
        )

    try:
        result = evaluate(
            checkpoint_path,
            corpus_path=corpus_path,
            max_batches=spec.get("max_batches"),
            report_dir=report_dir,
            should_cancel=lambda: bool(context.cancelled),
            on_batch=on_batch,
        )
    except TrainingCancelled:
        context.raise_if_cancelled()
        raise
    context.register_artifact(
        result.report_path,
        kind="evaluation-report",
        media_type="application/json",
    )
    context.progress(1.0, message="Tiny evaluation complete")
    return {
        "model_id": result.model_id,
        "loss": result.loss,
        "perplexity": result.perplexity,
        "evaluated_tokens": result.evaluated_tokens,
        "batches": result.batches,
        "report_path": str(result.report_path.relative_to(artifact_root)),
    }


def _contained_directory(root: Path, supplied: str) -> Path:
    resolved = _contained(root, supplied, must_exist=False)
    if not resolved.is_dir():
        raise FileNotFoundError(f"not a run or checkpoint directory: {supplied}")
    return resolved


def _kernel_settings(spec: dict[str, Any]) -> Any:
    from .kernel import PackedTrainingSettings
    from .optim import OptimizerSettings

    max_steps = int(spec.get("max_steps", 200))
    return PackedTrainingSettings(
        tokenized_id=str(spec["tokenized_id"]),
        seq_len=int(spec.get("seq_len", 128)),
        tokens_per_batch=int(spec.get("tokens_per_batch", 4_096)),
        micro_batch_size=spec.get("micro_batch_size"),
        data_seed=int(spec.get("data_seed", 23)),
        model_seed=int(spec.get("model_seed", 17)),
        precision=spec.get("precision", "bf16"),
        activation_checkpointing=bool(spec.get("activation_checkpointing", False)),
        checkpoint_every=int(spec.get("checkpoint_every", 50)),
        keep_checkpoints=int(spec.get("keep_checkpoints", 3)),
        validation_split=spec.get("validation_split"),
        eval_max_steps=spec.get("eval_max_steps"),
        optimizer=OptimizerSettings(
            learning_rate=float(spec.get("learning_rate", 3e-4)),
            min_lr_ratio=float(spec.get("min_lr_ratio", 0.1)),
            warmup_steps=int(spec.get("warmup_steps", 0)),
            total_steps=int(spec.get("total_steps") or max_steps),
            weight_decay=float(spec.get("weight_decay", 0.1)),
            gradient_clip=float(spec.get("gradient_clip", 1.0)),
        ),
    )


def _run_kernel_train(spec: dict[str, Any], context: Any) -> dict[str, Any]:
    """Single-process kernel training; the trainer refuses any trajectory change on resume."""

    from .run_checkpoint import latest_checkpoint
    from .runner import derive_kernel_model, run_kernel

    settings = context.service.settings
    project_root = Path(settings.project_root).resolve()
    artifact_root = Path(settings.artifact_root).resolve()
    data_root = Path(settings.data_root).resolve()
    if spec.get("strategy", "none") != "none":
        raise ValueError("control jobs train in one process; use the CLI for local ddp/fsdp2")
    recipe = _contained(
        project_root,
        str(spec.get("model_config", "configs/models/epor-tiny.yaml")),
        must_exist=True,
    )
    output_name = str(spec.get("output_name", "kernel"))
    if not _OUTPUT_NAME.fullmatch(output_name):
        raise ValueError("output_name is not a safe artifact directory name")
    output_dir = _contained(artifact_root, Path(context.job_id) / output_name, must_exist=False)
    resume_from = (
        None
        if spec.get("resume_from") is None
        else _contained_directory(artifact_root, str(spec["resume_from"]))
    )
    max_steps = int(spec.get("max_steps", 200))
    model = derive_kernel_model(
        recipe,
        data_root=data_root,
        tokenized_id=str(spec["tokenized_id"]),
        vocab_from_build=bool(spec.get("vocab_from_build", False)),
        qk_norm=spec.get("qk_norm"),
        mtp_depth=spec.get("mtp_depth"),
    )
    training = _kernel_settings(spec)

    context.raise_if_cancelled()
    context.log(f"Starting kernel training to step {max_steps} ({training.precision}, 1 process)")
    context.progress(0.02, message="Validated kernel training inputs")

    def on_step(step: int, total: int, loss: float) -> None:
        # Leave the cancellation exception to the trainer, which checkpoints
        # the last completed step before stopping.
        if context.cancelled:
            return
        context.progress(
            0.02 + 0.93 * step / total,
            message=f"Completed kernel step {step} of {total}",
            details={"step": step, "max_steps": total, "loss": loss},
        )

    summary: dict[str, Any] | None = None
    try:
        summary = run_kernel(
            model,
            data_root,
            output_dir,
            training,
            max_steps=max_steps,
            resume_from=resume_from,
            project_root=project_root,
            should_cancel=lambda: bool(context.cancelled),
            on_step=on_step,
        )
    except TrainingCancelled:
        context.raise_if_cancelled()
        raise
    finally:
        _register_if_present(
            context,
            output_dir / "manifest.json",
            kind="run-manifest",
            media_type="application/json",
        )
        _register_if_present(
            context,
            output_dir / "metrics.jsonl",
            kind="training-metrics",
            media_type="application/x-ndjson",
        )
        latest = latest_checkpoint(output_dir) if output_dir.is_dir() else None
        if latest is not None:
            _register_if_present(
                context,
                latest / "checkpoint.json",
                kind="checkpoint-manifest",
                media_type="application/json",
            )

    assert summary is not None
    context.progress(1.0, message="Kernel training complete")
    evaluation = summary.get("evaluation")
    return {
        "run_id": summary["run_id"],
        "lineage_id": summary["lineage_id"],
        "start_step": summary["start_step"],
        "steps_completed": summary["steps_completed"],
        "final_loss": summary["final_loss"],
        "final_main_loss": summary["main_losses"][-1] if summary["main_losses"] else None,
        "tokens_seen": summary["tokens_seen"],
        "tokens_per_batch": training.tokens_per_batch,
        "precision": training.precision,
        "strategy": "none",
        "model": summary["model"],
        "evaluation": evaluation,
        "output_path": str(output_dir.relative_to(artifact_root)),
        "checkpoint_path": str(Path(summary["checkpoint_path"]).relative_to(artifact_root)),
        "scope": "proxy-scale local training run; not a validated result",
    }


def _run_kernel_eval(spec: dict[str, Any], context: Any) -> dict[str, Any]:
    from .kernel import evaluate_packed_checkpoint
    from .run_checkpoint import resolve_checkpoint

    settings = context.service.settings
    artifact_root = Path(settings.artifact_root).resolve()
    checkpoint = resolve_checkpoint(
        _contained_directory(artifact_root, str(spec["checkpoint_path"]))
    )
    _contained(artifact_root, checkpoint, must_exist=False)
    report_dir = _contained(artifact_root, Path(context.job_id) / "kernel-eval", must_exist=False)
    split = spec.get("split", "validation")
    context.raise_if_cancelled()
    context.progress(0.05, message=f"Evaluating {checkpoint.name} on the {split} split")

    def on_batch(batch: int) -> None:
        if context.cancelled:
            return
        context.progress(
            min(0.95, 0.05 + 0.9 * batch / (batch + 1)),
            message=f"Evaluated batch {batch}",
            details={"batch": batch},
        )

    try:
        result = evaluate_packed_checkpoint(
            checkpoint,
            Path(settings.data_root).resolve(),
            split=split,
            max_steps=spec.get("max_steps"),
            report_dir=report_dir,
            should_cancel=lambda: bool(context.cancelled),
            on_batch=on_batch,
        )
    except TrainingCancelled:
        context.raise_if_cancelled()
        raise
    assert result.report_path is not None
    context.register_artifact(
        result.report_path, kind="evaluation-report", media_type="application/json"
    )
    if split == "train":
        context.log("in-sample measurement on the training split", level="warning")
    context.progress(1.0, message="Kernel evaluation complete")
    return {
        "model_id": result.model_id,
        "checkpoint_path": str(checkpoint.relative_to(artifact_root)),
        **result.evaluation.model_dump(mode="json"),
        "report_path": str(result.report_path.relative_to(artifact_root)),
        "scope": "proxy-scale local measurement; not a validated result",
    }


def run_control_job(name: str, spec: dict[str, Any], context: Any) -> dict[str, Any]:
    """Execute one allowlisted training job; never a shell command."""

    if name == "tiny_train":
        return _run_train(spec, context)
    if name == "tiny_eval":
        return _run_eval(spec, context)
    if name == "kernel_train":
        return _run_kernel_train(spec, context)
    if name == "kernel_evaluate":
        return _run_kernel_eval(spec, context)
    raise ValueError(f"unsupported training control job: {name}")
