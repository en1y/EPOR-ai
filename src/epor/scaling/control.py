"""Adapter between allowlisted scaling-laboratory control jobs and the scaling runners.

``scale_profile``, ``scale_sweep``, and ``scale_mixture`` run here.  Every
model config resolves beneath the project root, every output directory
beneath ``<artifact_root>/<job id>/``, and every config is bounded by the local
kernel policy exactly as on the CLI.  Fits and projections stay on the CLI:
they are cheap arithmetic over a stored report and an operator-written inputs
file, and a projection never provisions anything.

Job results carry the report ID and a small fixed summary
(:func:`epor.scaling.summary.scaling_summary`); the full report is served,
verified, by the read-only console endpoints.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from epor.control.security import contained_path
from epor.data.store import DataStore

from .report import ScalingReport, store_scaling_report
from .summary import scaling_summary


def _roots(context: Any) -> tuple[Path, Path, Path]:
    settings = context.service.settings
    return (
        Path(settings.project_root).resolve(),
        Path(settings.artifact_root).resolve(),
        Path(settings.data_root).resolve(),
    )


def _config_path(project_root: Path, supplied: str) -> str:
    path = contained_path(project_root, supplied, must_exist=True)
    if not path.is_file():
        raise FileNotFoundError(f"model config not found: {supplied}")
    return path.relative_to(project_root).as_posix()


def packed_template(training: Mapping[str, Any], *, total_steps: int) -> Any:
    """``PackedTrainingSettings`` from a job's ``training`` block."""

    from epor.training import OptimizerSettings, PackedTrainingSettings

    warmup = int(training.get("warmup_steps", 0))
    return PackedTrainingSettings(
        tokenized_id=str(training["tokenized_id"]),
        seq_len=int(training.get("seq_len", 128)),
        tokens_per_batch=int(training.get("tokens_per_batch", 4_096)),
        micro_batch_size=training.get("micro_batch_size"),
        precision=training.get("precision", "bf16"),
        data_seed=int(training.get("data_seed", 23)),
        model_seed=int(training.get("model_seed", 17)),
        checkpoint_every=int(training.get("checkpoint_every", 50)),
        validation_split=training.get("validation_split"),
        eval_max_steps=training.get("eval_max_steps"),
        optimizer=OptimizerSettings(
            learning_rate=float(training.get("learning_rate", 3e-4)),
            min_lr_ratio=float(training.get("min_lr_ratio", 0.1)),
            warmup_steps=warmup,
            total_steps=total_steps,
            weight_decay=float(training.get("weight_decay", 0.1)),
            gradient_clip=float(training.get("gradient_clip", 1.0)),
        ),
    )


def _output(artifact_root: Path, job_id: str, name: str) -> Path:
    return contained_path(artifact_root, Path(job_id) / name, must_exist=False)


def _progress(context: Any, label: str) -> Callable[[int, int, dict[str, Any]], None]:
    def report(done: int, total: int, result: dict[str, Any]) -> None:
        if context.cancelled:
            return
        context.progress(
            0.02 + 0.95 * done / max(total, 1),
            message=f"Completed {label} {done} of {total}",
            details={
                label: done,
                f"{label}s": total,
                "name": result.get("mixture_id") or result.get("rung"),
                "status": result.get("status", "completed"),
            },
        )

    return report


def _cancellable(context: Any, work: Callable[[], ScalingReport]) -> ScalingReport:
    from epor.training.loop import TrainingCancelled

    try:
        return work()
    except TrainingCancelled:
        context.raise_if_cancelled()
        raise


def _finish(context: Any, report: ScalingReport, data_root: Path, **extra: Any) -> dict[str, Any]:
    store_scaling_report(DataStore(data_root), report)
    for warning in report.warnings:
        context.log(warning, level="warning")
    context.progress(1.0, message="Scaling report stored")
    return {**scaling_summary(report), "warnings": list(report.warnings), **extra}


def _run_profile(spec: Mapping[str, Any], context: Any) -> dict[str, Any]:
    from epor.models.config import load_model_config

    from .ladder import derive_rung
    from .profile import ProfileSettings, profile_training_step

    project_root, _, data_root = _roots(context)
    config = load_model_config(
        project_root / _config_path(project_root, str(spec.get("model_config")))
    )
    if spec.get("vocab_from_build") is not None:
        from epor.training.runner import build_vocab_size

        config = derive_rung(config, build_vocab_size(data_root, str(spec["vocab_from_build"])))
    settings = ProfileSettings(
        batch_size=int(spec.get("batch_size", 4)),
        seq_len=int(spec.get("seq_len", 128)),
        warmup_steps=int(spec.get("warmup_steps", 2)),
        timed_steps=int(spec.get("timed_steps", 5)),
        precision=spec.get("precision", "fp32"),
        seed=int(spec.get("seed", 0)),
        threads=spec.get("threads"),
        peak_flops_per_second=spec.get("peak_flops_per_second"),
        device_label=str(spec.get("device_label", "cpu")),
    )
    context.raise_if_cancelled()
    context.progress(0.05, message=f"Profiling {config.slug} on this host")
    report = _cancellable(
        context,
        lambda: profile_training_step(
            config, settings, should_cancel=lambda: bool(context.cancelled)
        ),
    )
    return _finish(context, report, data_root)


def _run_sweep(spec: Mapping[str, Any], context: Any) -> dict[str, Any]:
    from .sweep import SweepPlan, run_sweep

    project_root, artifact_root, data_root = _roots(context)
    training = spec["training"]
    plan = SweepPlan(
        rungs=tuple(_config_path(project_root, str(path)) for path in spec["rungs"]),
        token_budgets=tuple(int(value) for value in spec["token_budgets"]),
        seeds=tuple(int(value) for value in spec.get("seeds", [0])),
        # Each point sets its own schedule length; the template carries warmup.
        training=packed_template(
            training, total_steps=max(1, int(training.get("warmup_steps", 0)))
        ),
    )
    output = _output(artifact_root, context.job_id, str(spec.get("output_name", "sweep")))
    context.raise_if_cancelled()
    context.progress(0.02, message="Starting the scaling sweep")
    report = _cancellable(
        context,
        lambda: run_sweep(
            plan,
            data_root=data_root,
            output_dir=output,
            project_root=project_root,
            should_cancel=lambda: bool(context.cancelled),
            on_point=_progress(context, "point"),
        ),
    )
    return _finish(context, report, data_root, output_path=str(output.relative_to(artifact_root)))


def _run_mixture(spec: Mapping[str, Any], context: Any) -> dict[str, Any]:
    from .mixture import MixturePlan, NamedMixture, run_mixture_ablation

    project_root, artifact_root, data_root = _roots(context)
    max_steps = int(spec["max_steps"])
    template = packed_template(
        {**spec["training"], "validation_split": None, "eval_max_steps": None},
        total_steps=max_steps,
    )
    plan = MixturePlan(
        model=_config_path(project_root, str(spec.get("model_config"))),
        mixtures=tuple(NamedMixture.model_validate(item) for item in spec["mixtures"]),
        max_steps=max_steps,
        evaluation_split=str(spec.get("evaluation_split", "validation")),
        evaluation_max_steps=spec.get("evaluation_max_steps"),
        training=template,
    )
    output = _output(artifact_root, context.job_id, str(spec.get("output_name", "mixture")))
    context.raise_if_cancelled()
    context.progress(0.02, message="Starting the mixture ablation")
    report = _cancellable(
        context,
        lambda: run_mixture_ablation(
            plan,
            data_root=data_root,
            output_dir=output,
            project_root=project_root,
            should_cancel=lambda: bool(context.cancelled),
            on_mixture=_progress(context, "mixture"),
        ),
    )
    return _finish(context, report, data_root, output_path=str(output.relative_to(artifact_root)))


def run_scaling_job(name: str, spec: Mapping[str, Any], context: Any) -> dict[str, Any]:
    """Execute one allowlisted scaling job; never a shell command, never provisioning."""

    if name == "scale_profile":
        return _run_profile(spec, context)
    if name == "scale_sweep":
        return _run_sweep(spec, context)
    if name == "scale_mixture":
        return _run_mixture(spec, context)
    raise ValueError(f"unsupported scaling control job: {name}")


__all__ = ["packed_template", "run_scaling_job"]
