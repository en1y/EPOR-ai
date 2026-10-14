"""Adapter between allowlisted alignment control jobs and :mod:`epor.alignment`.

Six job types run here, each the job twin of a CLI command and gated by the
action of the same name: ``align_generate``, ``align_judge``,
``align_dataset``, ``align_sft``, ``align_preference``, and
``align_evaluate``.  Two pieces of alignment work are deliberately *not* jobs:

* registering or importing operator teacher material (``align_register``)
  stays on the owner's CLI, like code execution in v0.0.6;
* a human verdict (``align_audit``) is never a job: it is recorded by a
  verified human through ``POST /api/v1/alignment/audits/{id}/verdict`` or the
  CLI.  No handler here writes a verdict, so a job can never mark anything
  audited.

Checkpoints are read only from beneath the artifact root.  Results carry
record, dataset, and report IDs plus small summaries whose keys avoid the
credential pattern; model text never appears in a job result or event.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from epor.control.security import contained_path

_OUTPUT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def _roots(context: Any) -> tuple[Path, Path, Path, Path]:
    settings = context.service.settings
    return (
        Path(settings.project_root).resolve(),
        Path(settings.artifact_root).resolve(),
        Path(settings.data_root).resolve(),
        Path(settings.safety_root).resolve(),
    )


def _checkpoint(artifact_root: Path, supplied: str) -> Path:
    from epor.training.run_checkpoint import resolve_checkpoint

    directory = contained_path(artifact_root, supplied, must_exist=True)
    if not directory.is_dir():
        raise FileNotFoundError(f"not a run or checkpoint directory: {supplied}")
    checkpoint = resolve_checkpoint(directory)
    contained_path(artifact_root, checkpoint, must_exist=True)
    return checkpoint


def _ref(artifact_root: Path, supplied: str) -> Any:
    from epor.training.kernel import CheckpointRef

    from .model import checkpoint_identity

    directory = _checkpoint(artifact_root, supplied)
    return CheckpointRef(path=str(directory), checkpoint_sha256=checkpoint_identity(directory))


def _model(artifact_root: Path, data_root: Path, supplied: str) -> Any:
    from .model import load_chat_model

    return load_chat_model(_checkpoint(artifact_root, supplied), data_root)


def _sampling(spec: dict[str, Any]) -> Any:
    from .records import Sampling

    temperature = float(spec.get("temperature", 0.0))
    max_new = int(spec.get("max_new_tokens", 48))
    seed = int(spec.get("seed", 0))
    if temperature == 0:
        return Sampling(decoding="greedy", max_new_tokens=max_new, seed=seed)
    return Sampling(
        decoding="temperature",
        temperature=temperature,
        top_k=spec.get("top_k"),
        max_new_tokens=max_new,
        seed=seed,
    )


def _cancel(context: Any) -> Any:
    return lambda: bool(context.cancelled)


def _run_generate(spec: dict[str, Any], context: Any) -> dict[str, Any]:
    from .records import AlignmentStore, parent

    _, artifact_root, data_root, _ = _roots(context)
    store = AlignmentStore(data_root)
    model = _model(artifact_root, data_root, str(spec["checkpoint_path"]))
    mode = spec["mode"]
    sampling = _sampling(spec)
    context.raise_if_cancelled()
    context.progress(0.05, message=f"Generating {mode} candidates")
    if mode == "teacher-sample":
        from .fixtures import load_fixtures, prompt_lineage
        from .teachers import sample_teacher

        prompts: list[Any] = []
        if spec.get("fixture_prompts", True):
            fixtures = load_fixtures()
            prompts += [(i.messages, (prompt_lineage(fixtures, i),)) for i in fixtures.prompts]
        for identifier in spec.get("prompt_records", []):
            prompts.append((store.get(identifier).messages, (parent(identifier),)))
        if not prompts:
            raise ValueError("no prompts: enable fixture_prompts or name prompt_records")
        written = sample_teacher(store, model, prompts, sampling, should_cancel=_cancel(context))
        result: dict[str, Any] = {"examples": written}
    elif mode == "critique":
        from .critique import critique_and_revise

        entries = critique_and_revise(
            store,
            model,
            list(spec["examples"]),
            list(spec["obligations"]),
            sampling,
            should_cancel=_cancel(context),
        )
        result = {"candidates": entries}
    else:
        from .judge import generate_eval_items

        result = {
            "eval_items": generate_eval_items(
                store, model, int(spec.get("count", 6)), sampling, should_cancel=_cancel(context)
            )
        }
    context.progress(1.0, message="Candidates stored")
    return {
        "mode": mode,
        **result,
        "generator": model.generator.model_dump(mode="json"),
        "note": "candidates only; nothing here trains without an audited judging run",
    }


def _run_judge(spec: dict[str, Any], context: Any) -> dict[str, Any]:
    from .judge import judge_eval_items, judge_pairs, unjudged_revisions
    from .records import AlignmentStore

    _, artifact_root, data_root, _ = _roots(context)
    store = AlignmentStore(data_root)
    judge = _model(artifact_root, data_root, str(spec["checkpoint_path"]))
    subject = spec["subject"]
    min_margin = float(spec.get("min_margin", 0.05))
    context.raise_if_cancelled()
    context.progress(0.05, message=f"Judging {subject} candidates")
    if subject == "eval-item":
        items = list(spec.get("items", []))
        if spec.get("all_unjudged"):
            judged = {
                record.inputs[0]
                for _, record in store.records("judgement")
                if record.subject == "eval-item"
            }
            items += [item for item in store.ids("eval-item") if item not in judged]
        if not items:
            raise ValueError("no evaluation items to judge")
        result = judge_eval_items(
            store, judge, items, min_margin=min_margin, should_cancel=_cancel(context)
        )
    else:
        pairs = [tuple(pair) for pair in spec.get("pairs", [])]
        if spec.get("all_unjudged") and subject == "revision":
            pairs += unjudged_revisions(store)
        if not pairs:
            raise ValueError("no pairs to judge")
        result = judge_pairs(
            store,
            judge,
            pairs,  # type: ignore[arg-type]
            obligation=str(spec.get("obligation", "no-harmful-output")),
            min_margin=min_margin,
            subject=subject,
            should_cancel=_cancel(context),
        )
    accepted = sum(1 for item in result["judgements"] if store.get(item).accepted)
    context.progress(1.0, message="Judge decisions stored; a human audit comes next")
    return {
        **result,
        "accepted": accepted,
        "rejected": len(result["judgements"]) - accepted,
        "audit_state": "unaudited",
    }


def _run_dataset(spec: dict[str, Any], context: Any) -> dict[str, Any]:
    from epor.tokenizer import load_tokenizer

    from .audit import AuditLedger, default_gate
    from .datasets import (
        build_preference_dataset,
        build_sft_dataset,
        describe_dataset,
        select_records,
    )
    from .records import AlignmentStore

    _, artifact_root, data_root, safety_root = _roots(context)
    store = AlignmentStore(data_root)
    if spec.get("checkpoint_path"):
        tokenizer = _model(artifact_root, data_root, str(spec["checkpoint_path"])).tokenizer
    else:
        tokenizer = load_tokenizer(data_root, str(spec["tokenizer_id"]))
    kind = spec["kind"]
    ids = select_records(store, kind, spec.get("include", []), spec.get("records", []))
    if not ids:
        raise ValueError("no records selected")
    context.raise_if_cancelled()
    build = build_sft_dataset if kind == "sft" else build_preference_dataset
    dataset_id, manifest = build(
        store,
        tokenizer,
        ids,
        validation_percent=int(spec.get("validation_percent", 10)),
        gate=default_gate(AuditLedger(safety_root), store),
    )
    context.progress(1.0, message="Dataset stored")
    return describe_dataset(dataset_id, manifest)


def _register_run(context: Any, output_dir: Path) -> None:
    from epor.training.run_checkpoint import latest_checkpoint

    for name, kind, media in (
        ("manifest.json", "run-manifest", "application/json"),
        ("metrics.jsonl", "training-metrics", "application/x-ndjson"),
    ):
        if (output_dir / name).is_file():
            context.register_artifact(output_dir / name, kind=kind, media_type=media)
    latest = latest_checkpoint(output_dir) if output_dir.is_dir() else None
    if latest is not None:
        context.register_artifact(
            latest / "checkpoint.json", kind="checkpoint-manifest", media_type="application/json"
        )


def _output(artifact_root: Path, context: Any, spec: dict[str, Any], default: str) -> Path:
    name = str(spec.get("output_name", default))
    if not _OUTPUT_NAME.fullmatch(name):
        raise ValueError("output_name is not a safe artifact directory name")
    return contained_path(artifact_root, Path(context.job_id) / name)


def _on_step(context: Any) -> Any:
    def on_step(step: int, total: int, loss: float) -> None:
        # The trainer checkpoints before raising on cancellation.
        if context.cancelled:
            return
        context.progress(
            0.02 + 0.93 * step / total,
            message=f"Completed step {step} of {total}",
            details={"step": step, "max_steps": total, "loss": loss},
        )

    return on_step


def _run_sft(spec: dict[str, Any], context: Any) -> dict[str, Any]:
    from epor.models.config import model_config_from_mapping
    from epor.training import OptimizerSettings, PackedTrainingSettings
    from epor.training.kernel import LogitDistillation, resolve_model_config
    from epor.training.loop import TrainingCancelled
    from epor.training.run_checkpoint import read_kernel_checkpoint
    from epor.training.runner import resume_inputs, run_kernel

    project_root, artifact_root, data_root, _ = _roots(context)
    output_dir = _output(artifact_root, context, spec, "sft")
    max_steps = int(spec.get("max_steps", 50))
    resume_from = None
    if spec.get("resume_from"):
        resume_from = _checkpoint(artifact_root, str(spec["resume_from"]))
        model, settings, _, _ = resume_inputs(resume_from)
    else:
        init = _ref(artifact_root, str(spec["init_path"]))
        trainer = read_kernel_checkpoint(init.path)
        model = resolve_model_config(model_config_from_mapping(trainer["model_config"]))
        distillation = None
        if spec.get("logit_teachers"):
            distillation = LogitDistillation(
                teachers=tuple(_ref(artifact_root, str(p)) for p in spec["logit_teachers"]),
                weight=float(spec.get("distill_weight", 1.0)),
                temperature=float(spec.get("distill_temperature", 1.0)),
            )
        settings = PackedTrainingSettings(
            tokenized_id=PackedTrainingSettings.model_validate(trainer["settings"]).tokenized_id,
            chat_dataset_id=str(spec["dataset_id"]),
            init_checkpoint=init,
            distillation=distillation,
            seq_len=int(spec.get("seq_len", 128)),
            tokens_per_batch=int(spec.get("tokens_per_batch", 1_024)),
            micro_batch_size=spec.get("micro_batch_size"),
            precision=spec.get("precision", "bf16"),
            checkpoint_every=int(spec.get("checkpoint_every", 50)),
            keep_checkpoints=int(spec.get("keep_checkpoints", 2)),
            validation_split="validation" if spec.get("validation", True) else None,
            optimizer=OptimizerSettings(
                learning_rate=float(spec.get("learning_rate", 1e-4)),
                warmup_steps=int(spec.get("warmup_steps", 0)),
                total_steps=max_steps,
                weight_decay=float(spec.get("weight_decay", 0.0)),
            ),
        )
    context.raise_if_cancelled()
    context.progress(0.02, message="Validated fine-tuning inputs")
    summary: dict[str, Any] | None = None
    try:
        summary = run_kernel(
            model,
            data_root,
            output_dir,
            settings,
            max_steps=max_steps,
            resume_from=resume_from,
            project_root=project_root,
            should_cancel=_cancel(context),
            on_step=_on_step(context),
        )
    except TrainingCancelled:
        context.raise_if_cancelled()
        raise
    finally:
        _register_run(context, output_dir)
    assert summary is not None
    context.progress(1.0, message="Fine-tuning complete")
    return {
        "run_id": summary["run_id"],
        "start_step": summary["start_step"],
        "steps_completed": summary["steps_completed"],
        "final_loss": summary["final_loss"],
        "objective": "logit-distillation" if settings.distillation else "sft",
        "chat_dataset_id": settings.chat_dataset_id,
        "evaluation": summary.get("evaluation"),
        "output_path": str(output_dir.relative_to(artifact_root)),
        "checkpoint_path": str(Path(summary["checkpoint_path"]).relative_to(artifact_root)),
        "scope": "proxy-scale local fine-tune; not evidence of alignment or safety",
    }


def _run_preference(spec: dict[str, Any], context: Any) -> dict[str, Any]:
    from epor.training import OptimizerSettings
    from epor.training.loop import TrainingCancelled

    from .monitors import load_monitor_config
    from .preference import DpoMonitor, DpoSettings, MonitorAbortError, train_preference

    project_root, artifact_root, data_root, _ = _roots(context)
    output_dir = _output(artifact_root, context, spec, "dpo")
    resume_from = (
        _checkpoint(artifact_root, str(spec["resume_from"])) if spec.get("resume_from") else None
    )
    max_steps = int(spec.get("max_steps", 50))
    policy = _ref(artifact_root, str(spec["init_path"]))
    limits = load_monitor_config().dpo_step_logratio
    settings = DpoSettings(
        dataset_id=str(spec["dataset_id"]),
        policy_init=policy,
        reference=(
            _ref(artifact_root, str(spec["reference_path"]))
            if spec.get("reference_path")
            else policy
        ),
        beta=float(spec.get("beta", 0.1)),
        pairs_per_step=int(spec.get("pairs_per_step", 4)),
        seed=int(spec.get("seed", 29)),
        precision=spec.get("precision", "fp32"),
        optimizer=OptimizerSettings(
            learning_rate=float(spec.get("learning_rate", 1e-4)),
            weight_decay=0.0,
            total_steps=max_steps,
        ),
        monitor=DpoMonitor(logratio_alert=limits.alert, logratio_abort=limits.abort),
        checkpoint_every=int(spec.get("checkpoint_every", 25)),
    )
    context.raise_if_cancelled()
    try:
        result = train_preference(
            data_root,
            output_dir,
            settings,
            max_steps=max_steps,
            resume_from=resume_from,
            project_root=project_root,
            should_cancel=_cancel(context),
            on_step=_on_step(context),
        )
    except TrainingCancelled:
        context.raise_if_cancelled()
        raise
    except MonitorAbortError as exc:
        context.log(str(exc), level="error")
        raise
    finally:
        _register_run(context, output_dir)
    for alert in result.alerts:
        context.log(alert, level="warning")
    context.progress(1.0, message="Preference optimization complete")
    return {
        "run_id": result.run_id,
        "start_step": result.start_step,
        "steps_completed": result.steps_completed,
        "final_loss": result.losses[-1] if result.losses else None,
        "alerts": list(result.alerts),
        "evaluation": None if result.evaluation is None else result.evaluation.model_dump(),
        "output_path": str(output_dir.relative_to(artifact_root)),
        "checkpoint_path": str(result.checkpoint_path.relative_to(artifact_root)),
        "scope": "proxy-scale DPO baseline; not evidence of alignment or safety",
    }


def _run_evaluate(spec: dict[str, Any], context: Any) -> dict[str, Any]:
    from .audit import AuditLedger, calibration
    from .monitors import audited_pairs, held_out_pairs, overoptimization_report, store_report
    from .records import AlignmentStore

    _, artifact_root, data_root, safety_root = _roots(context)
    store = AlignmentStore(data_root)
    ledger = AuditLedger(safety_root)
    mode = spec["mode"]
    max_new = int(spec.get("max_new_tokens", 32))
    context.raise_if_cancelled()
    context.progress(0.05, message=f"Measuring {mode}")
    if mode == "calibration":
        report = {"schema_version": 1, "kind": "calibration"}
        report.update(calibration(ledger, str(spec["judging_run"]), store=store))
        report_id = store_report(store, "calibration", report)
        context.progress(1.0, message="Calibration stored")
        return {
            "report_id": report_id,
            "audited": report["audited"],
            "pending": report["pending"],
            "disagreement_rate": report["disagreement_rate"],
        }
    if mode == "monitor":
        from .fixtures import load_fixtures

        policy = _model(artifact_root, data_root, str(spec["policy_path"]))
        reference = _model(artifact_root, data_root, str(spec["reference_path"]))
        held_out = held_out_pairs(store, str(spec["dataset_id"]))
        prompts = [item.messages for item in load_fixtures().prompts]
        prompts += [messages for messages, _, _ in held_out]
        report_id, report = overoptimization_report(
            store,
            policy,
            reference,
            prompts=prompts,
            held_out_pairs=held_out,
            audited_pairs=audited_pairs(store, ledger),
            beta=float(spec.get("beta", 0.1)),
            max_new_tokens=max_new,
            should_cancel=_cancel(context),
        )
        if report["status"] in {"alert", "abort"}:
            context.log(f"overoptimization monitor status {report['status']}", level="warning")
        context.progress(1.0, message="Overoptimization report stored")
        return {
            "report_id": report_id,
            "status": report["status"],
            # Keys chosen to avoid the control plane's credential pattern; the
            # stored report keeps the monitors' full names.
            "monitor_status": {
                "kl": report["monitors"]["kl_per_token"]["status"],
                "proxy_minus_audited": report["monitors"]["proxy_minus_audited"]["status"],
                "length_ratio": report["monitors"]["length_ratio"]["status"],
            },
        }
    from .g10 import run_g10

    before = _model(artifact_root, data_root, str(spec["before_path"]))
    after = _model(artifact_root, data_root, str(spec["after_path"]))
    report_id, report = run_g10(
        store,
        before,
        after,
        ledger=ledger,
        actor_id=str(getattr(context, "job_id", "job")),
        tolerance=float(spec.get("tolerance", 0.0)),
        max_new_tokens=max_new,
        should_cancel=_cancel(context),
    )
    if report["gate"] != "no-regression-observed":
        context.log(
            f"G10 gate {report['gate']}: "
            + (
                "the aligned checkpoint fails closed"
                if report["gate"] == "blocked"
                else "failures are queued for human review"
            ),
            level="warning",
        )
    context.progress(1.0, message="G10 report stored")
    return {
        "report_id": report_id,
        "gate": report["gate"],
        "regressed_categories": report["regressed_categories"],
        "failing_items": len(report["failing_items"]),
        "claim": report["claim"],
    }


_HANDLERS = {
    "align_generate": _run_generate,
    "align_judge": _run_judge,
    "align_dataset": _run_dataset,
    "align_sft": _run_sft,
    "align_preference": _run_preference,
    "align_evaluate": _run_evaluate,
}


def run_alignment_job(name: str, spec: dict[str, Any], context: Any) -> dict[str, Any]:
    """Execute one allowlisted alignment job; never a shell command or a verdict."""

    handler = _HANDLERS.get(name)
    if handler is None:
        raise ValueError(f"unsupported alignment control job: {name}")
    try:
        return handler(spec, context)
    except InterruptedError:
        context.raise_if_cancelled()
        raise


__all__ = ["run_alignment_job"]
