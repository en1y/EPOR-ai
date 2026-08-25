"""Adapter between allowlisted control jobs and reference training functions."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from epor.models.config import load_model_config, structural_parameter_count

from .loop import TrainingCancelled, evaluate, pretrain

_OUTPUT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_MAX_CONTROL_MODEL_PARAMETERS = 50_000_000


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
    model_config = load_model_config(config_path)
    if model_config.vocab_size != 260 or model_config.architecture != "dense-decoder":
        raise ValueError("tiny training requires the dense 260-token debug model contract")
    parameter_count = structural_parameter_count(model_config)
    if parameter_count > _MAX_CONTROL_MODEL_PARAMETERS:
        raise ValueError("tiny training model exceeds the 50,000,000-parameter control-plane limit")
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
    if not 1 <= max_steps <= 1_000:
        raise ValueError("max_steps must be between 1 and 1000")
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


def run_control_job(name: str, spec: dict[str, Any], context: Any) -> dict[str, Any]:
    """Execute one of the two allowlisted training jobs; never a shell command."""

    if name == "tiny_train":
        return _run_train(spec, context)
    if name == "tiny_eval":
        return _run_eval(spec, context)
    raise ValueError(f"unsupported training control job: {name}")
