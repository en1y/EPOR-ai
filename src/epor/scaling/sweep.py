"""Resumable model-size x token-budget x seed sweeps on the local training kernel.

:func:`run_sweep` expands a :class:`SweepPlan` into points (rung, token
budget, seed).  For every point:

* The rung is derived with the tokenized build's vocabulary
  (:func:`epor.scaling.ladder.derive_rung`).  A rung above the local kernel
  bound is recorded with status ``projection-only`` and its parameter counts;
  it is never allocated and never silently dropped.
* A local rung trains with :func:`epor.training.kernel.pretrain_packed`
  (through :func:`epor.training.runner.derive_kernel_model`, so the usual
  bound check applies) for ``ceil(token_budget / tokens_per_batch)`` steps.
  The point overrides ``model_seed`` and ``data_seed`` with its seed and sets
  the cosine schedule's ``total_steps`` to its step count (warmup capped at
  it), so each budget is a complete schedule rather than a truncated one.
  ``keep_checkpoints`` is 1 to bound disk use.

Point identity
--------------
``point_id`` is the SHA-256 of the canonical JSON of the derived config hash
and the point's training trajectory (which includes the tokenized build,
seeds, geometry, optimizer, and step count).  Completed points are written to
``<output>/points/<point_id>.json``; a rerun skips them.  A point whose run
was interrupted resumes from its last checkpoint
(``<output>/runs/<point_id>/attempt-<n>``); an attempt that never reached a
checkpoint is left in place for diagnosis and a new attempt starts.  Failed
points are recorded with status ``failed`` and kept in the report (the compute
policy requires failed runs in scaling analysis); cancellation stops the sweep
immediately, and nothing is sealed until every point is resolved.

The sealed report's content holds deterministic results (losses, tokens,
loss curves); wall time, throughput, RSS, run IDs, and directories go to its
``measured`` section.  Scope is ``proxy-scale`` (``harness-fixture`` for test
fixtures): a loss here is a measurement of a small local model, not evidence
about a release-scale model.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from epor.data.store import canonical_json_bytes, sha256_bytes
from epor.models.config import ModelConfig, load_model_config
from epor.reference_policy import KERNEL_MAX_STEPS

from .flops import estimate_flops, parameter_counts
from .ladder import LADDER_DIRECTORY, LADDER_RUNGS, PROJECT_ROOT, derive_rung, rung_status
from .report import KAPLAN_2020, ScalingReport, ScalingScope, seal_scaling_report

LOSS_CURVE_POINTS = 128


class SweepPlan(BaseModel):
    """What to sweep.  ``rungs`` are model-config paths relative to the project root."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    rungs: tuple[str, ...] = tuple(
        (LADDER_DIRECTORY / f"{slug}.yaml").as_posix() for slug in LADDER_RUNGS
    )
    token_budgets: tuple[int, ...] = Field(min_length=1)
    seeds: tuple[int, ...] = (0,)
    # Template for every point (``tokenized_id``, geometry, optimizer, precision).
    training: Any

    @field_validator("rungs", "token_budgets", "seeds")
    @classmethod
    def unique_values(cls, value: tuple[Any, ...]) -> tuple[Any, ...]:
        if not value:
            raise ValueError("sweep axes must not be empty")
        if len(set(value)) != len(value):
            raise ValueError("sweep axes must not repeat values")
        return value

    @field_validator("token_budgets")
    @classmethod
    def positive_budgets(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if any(budget < 1 for budget in value):
            raise ValueError("token budgets must be positive")
        return value

    @field_validator("seeds")
    @classmethod
    def non_negative_seeds(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if any(seed < 0 for seed in value):
            raise ValueError("seeds must be non-negative")
        return value

    @field_validator("training")
    @classmethod
    def training_settings(cls, value: Any) -> Any:
        from epor.training.kernel import PackedTrainingSettings

        return (
            value
            if isinstance(value, PackedTrainingSettings)
            else PackedTrainingSettings.model_validate(value)
        )

    def payload(self) -> dict[str, Any]:
        return {
            **self.model_dump(mode="json", exclude={"training"}),
            "training": self.training.model_dump(mode="json"),
        }


def _point_settings(plan: SweepPlan, budget: int, seed: int) -> tuple[Any, int]:
    template = plan.training
    steps = math.ceil(budget / template.tokens_per_batch)
    if steps > KERNEL_MAX_STEPS:
        raise ValueError(f"token budget {budget:,} needs more than {KERNEL_MAX_STEPS:,} steps")
    optimizer = template.optimizer.model_copy(
        update={"total_steps": steps, "warmup_steps": min(template.optimizer.warmup_steps, steps)}
    )
    settings = template.model_copy(
        update={
            "model_seed": seed,
            "data_seed": seed,
            "optimizer": optimizer,
            "keep_checkpoints": 1,
            "checkpoint_every": min(template.checkpoint_every, steps),
        }
    )
    return type(template).model_validate(settings.model_dump(mode="json")), steps


def point_id(config: ModelConfig, settings: Any, steps: int) -> str:
    return sha256_bytes(
        canonical_json_bytes(
            {
                "config_sha256": config.sha256,
                "trajectory": settings.trajectory(),
                "max_steps": steps,
            }
        )
    )


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _loss_curve(run_dir: Path, steps: int) -> list[list[float]]:
    by_step: dict[int, float] = {}
    metrics = run_dir / "metrics.jsonl"
    if metrics.is_file():
        for line in metrics.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                by_step[int(record["step"])] = float(record["main_loss"])
    if not by_step:
        return []
    stride = max(1, math.ceil(steps / LOSS_CURVE_POINTS))
    chosen = sorted({step for step in by_step if step % stride == 0 or step == 1} | {max(by_step)})
    return [[float(step), by_step[step]] for step in chosen]


def _completed_record(run_dir: Path, steps: int) -> tuple[dict[str, Any], dict[str, Any]] | None:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = read_json(manifest_path)
    if manifest.get("status") != "succeeded" or manifest.get("steps_completed") != steps:
        return None
    metrics = manifest.get("metrics") or {}
    evaluation = manifest.get("evaluation") or {}
    result = {
        "final_main_loss": metrics.get("main_loss"),
        "final_training_loss": metrics.get("training_loss"),
        "tokens_seen": metrics.get("tokens_seen"),
        "target_tokens_seen": metrics.get("target_tokens_seen"),
        "validation_main_loss": evaluation.get("main_loss"),
        "validation_perplexity": evaluation.get("perplexity"),
        "validation_target_tokens": evaluation.get("target_tokens"),
        "loss_curve": _loss_curve(run_dir, steps),
    }
    usage = manifest.get("actual_resource_use") or {}
    throughput = manifest.get("throughput") or {}
    measured = {
        "run_id": manifest.get("run_id"),
        "lineage_id": manifest.get("lineage_id"),
        "run_dir": str(run_dir),
        "wall_seconds_last_segment": usage.get("wall_seconds"),
        "maximum_rss_bytes": usage.get("maximum_rss_bytes"),
        "tokens_per_second": throughput.get("tokens_per_second"),
        "step_seconds": throughput.get("step_seconds"),
        "hardware": manifest.get("hardware"),
    }
    return result, measured


def train_point(
    config: ModelConfig,
    settings: Any,
    steps: int,
    point_dir: Path,
    *,
    data_root: Path,
    project_root: Path,
    should_cancel: Callable[[], bool] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from epor.training.kernel import pretrain_packed
    from epor.training.run_checkpoint import latest_checkpoint, read_kernel_checkpoint
    from epor.training.runner import derive_kernel_model

    model = derive_kernel_model(
        config, data_root=data_root, tokenized_id=settings.tokenized_id, vocab_from_build=True
    )
    attempts = sorted(
        (path for path in point_dir.glob("attempt-*") if path.is_dir()),
        key=lambda path: int(path.name.split("-", 1)[1]),
    )
    resume: Path | None = None
    run_dir: Path | None = None
    if attempts:
        last = attempts[-1]
        done = _completed_record(last, steps)
        if done is not None:
            return done
        checkpoint = latest_checkpoint(last)
        if checkpoint is not None and int(read_kernel_checkpoint(checkpoint)["step"]) < steps:
            resume, run_dir = last, last
    if run_dir is None:
        run_dir = point_dir / f"attempt-{len(attempts) + 1}"
    point_dir.mkdir(parents=True, exist_ok=True)
    pretrain_packed(
        model,
        data_root,
        run_dir,
        settings,
        max_steps=steps,
        resume_from=resume,
        project_root=project_root,
        should_cancel=should_cancel,
    )
    record = _completed_record(run_dir, steps)
    if record is None:
        raise RuntimeError(f"sweep point finished without a succeeded manifest: {run_dir}")
    return record


def write_point(path: Path, payload: dict[str, Any]) -> None:
    from epor.training.checkpoint import atomic_write_json

    atomic_write_json(path, payload)


def run_sweep(
    plan: SweepPlan,
    *,
    data_root: str | Path,
    output_dir: str | Path,
    project_root: str | Path = PROJECT_ROOT,
    should_cancel: Callable[[], bool] | None = None,
    on_point: Callable[[int, int, dict[str, Any]], None] | None = None,
    scope: ScalingScope = "proxy-scale",
) -> ScalingReport:
    """Run (or resume) every point of ``plan`` and return the sealed sweep report."""

    from epor.training.loop import TrainingCancelled
    from epor.training.manifest import hardware_snapshot, utc_now
    from epor.training.runner import build_vocab_size, is_cancellation

    root = Path(project_root).resolve()
    data = Path(data_root).resolve()
    output = Path(output_dir).resolve()
    vocab = build_vocab_size(data, plan.training.tokenized_id)
    rungs: list[ModelConfig] = []
    for relative in plan.rungs:
        path = Path(relative)
        rungs.append(load_model_config(path if path.is_absolute() else root / path))
    expanded: list[tuple[ModelConfig, ModelConfig, int, int]] = [
        (recipe, derive_rung(recipe, vocab), budget, seed)
        for recipe in rungs
        for budget in plan.token_budgets
        for seed in plan.seeds
    ]
    points: list[dict[str, Any]] = []
    measured_points: list[dict[str, Any]] = []
    for index, (recipe, derived, budget, seed) in enumerate(expanded):
        if should_cancel is not None and should_cancel():
            raise TrainingCancelled("sweep cancellation requested")
        settings, steps = _point_settings(plan, budget, seed)
        identifier = point_id(derived, settings, steps)
        status = rung_status(derived)
        counts = parameter_counts(derived)
        flops = estimate_flops(derived, settings.seq_len)
        base: dict[str, Any] = {
            "point_id": identifier,
            "rung": recipe.slug,
            "recipe_config_sha256": recipe.sha256,
            "config_sha256": derived.sha256,
            "vocab_size": derived.vocab_size,
            "parameters_total": counts.total,
            "parameters_non_embedding": counts.non_embedding,
            "token_budget": budget,
            "seed": seed,
            "max_steps": steps,
            "planned_tokens": steps * settings.tokens_per_batch,
            "train_flops_per_token": flops.train_flops_per_token,
            "planned_train_flops": flops.train_flops_per_token * steps * settings.tokens_per_batch,
        }
        point_file = output / "points" / f"{identifier}.json"
        if status.execution == "projection-only":
            result: dict[str, Any] = {**base, "status": "projection-only", "reason": status.reason}
            measured: dict[str, Any] = {"point_id": identifier}
        elif point_file.is_file():
            stored = read_json(point_file)
            result, measured = stored["result"], stored["measured"]
        else:
            try:
                outcome, run_measured = train_point(
                    derived,
                    settings,
                    steps,
                    output / "runs" / identifier,
                    data_root=data,
                    project_root=root,
                    should_cancel=should_cancel,
                )
                result = {**base, "status": "completed", **outcome}
            except Exception as exc:
                if is_cancellation(exc):
                    raise
                result = {
                    **base,
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:500],
                }
                run_measured = {}
            measured = {"point_id": identifier, **run_measured}
            write_point(point_file, {"result": result, "measured": measured})
        points.append(result)
        measured_points.append(measured)
        if on_point is not None:
            on_point(index + 1, len(expanded), result)

    statuses = [point["status"] for point in points]
    warnings = [
        "Proxy-scale sweep: losses are measurements of small local models on the identified "
        "build, not scaling evidence for release-scale models."
    ]
    if "projection-only" in statuses:
        warnings.append(
            "Rungs above the local kernel bound are projection-only and were not trained."
        )
    if "failed" in statuses:
        warnings.append("Some points failed; they are retained with their error.")
    return seal_scaling_report(
        kind="scaling-sweep",
        scope=scope,
        inputs={
            "plan": plan.payload(),
            "tokenized_id": plan.training.tokenized_id,
            "build_vocab_size": vocab,
            "rung_config_sha256": [recipe.sha256 for recipe in rungs],
            "seed_policy": "model_seed = data_seed = point seed",
            "schedule_policy": "optimizer.total_steps = point steps; warmup capped at it",
        },
        results={
            "points": points,
            "counts": {
                status: statuses.count(status)
                for status in ("completed", "projection-only", "failed")
            },
            "loss_definition": (
                "final_main_loss: last-step training main loss; validation_main_loss: "
                "mean next-token loss on the validation split (null when the plan has none)"
            ),
            "parameter_definition": "parameters_non_embedding is the N used by scaling fits",
        },
        warnings=warnings,
        citations=[KAPLAN_2020],
        measured={
            "recorded_at": utc_now(),
            "host": hardware_snapshot(),
            "output_dir": str(output),
            "points": measured_points,
        },
    )


def sweep_points(report: ScalingReport) -> list[dict[str, Any]]:
    if report.kind != "scaling-sweep":
        raise ValueError("not a scaling-sweep report")
    return list(report.results["points"])


__all__ = ["LOSS_CURVE_POINTS", "SweepPlan", "point_id", "run_sweep", "sweep_points"]
