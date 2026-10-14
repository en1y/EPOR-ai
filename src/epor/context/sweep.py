"""Window, local:global ratio, and split-RoPE-base sweeps against a full-attention control.

**[reported-result]** Gemma 3 observes little perplexity movement across
local-to-global ratios in its own study.  **[EPOR-adaptation]** EPOR judges
each axis on what a ratio can hide from perplexity: KV-cache bytes per token
and position-resolved retrieval (passkey and RULER-style multi-key needles at
the beginning, middle, and end), with validation loss reported beside them.

Each sweep varies exactly one axis — ``window`` (sliding-window sizes at a
fixed ratio), ``ratio`` (local layers per global layer at a fixed window), or
``rope-base`` (the local layers' RoPE base at a fixed window and ratio; the
value ``0`` means one shared base) — and always trains the full-attention
control from the same seed on the same tokens.  Points are trained with the
v0.0.5 kernel from scratch, kept one directory per point, and skipped on
re-run when their run already succeeded.  The sealed ``context-sweep`` report
holds losses and retrieval as content and wall-clock time as measured.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from epor.models.cache import kv_cache_bytes
from epor.models.config import ModelConfig
from epor.scaling.report import ScalingReport, seal_scaling_report

from .citations import GEMMA3, PASSKEY, RULER

Axis = Literal["window", "ratio", "rope-base"]


class ContextSweepPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    axis: Axis
    values: tuple[float, ...] = Field(min_length=1, max_length=8)
    window: int = Field(default=64, ge=1)
    ratio: int = Field(default=1, ge=1)
    tokenized_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    seq_len: int = Field(ge=32, le=65_536)
    tokens_per_batch: int = Field(ge=32)
    max_steps: int = Field(ge=1, le=100_000)
    learning_rate: float = Field(default=3e-3, gt=0, le=1)
    seed: int = Field(default=0, ge=0)
    precision: Literal["fp32", "bf16"] = "fp32"
    eval_length: int = Field(ge=96)
    items_per_cell: int = Field(default=2, ge=1, le=16)

    @model_validator(mode="after")
    def fits(self) -> ContextSweepPlan:
        if self.tokens_per_batch % self.seq_len:
            raise ValueError("tokens_per_batch must be a multiple of seq_len")
        if len(set(self.values)) != len(self.values):
            raise ValueError("sweep values must not repeat")
        return self


def variant(base: ModelConfig, plan: ContextSweepPlan, value: float | None) -> ModelConfig:
    """The control (``value is None``) or one point of the swept axis."""

    configured = max(base.configured_max_context, plan.seq_len, plan.eval_length + 64)
    fields: dict[str, Any] = {
        **base.model_dump(mode="json"),
        "configured_max_context": configured,
        "sliding_window": 0,
        "local_layers_per_global": 0,
        "local_rope_theta": None,
    }
    if value is not None:
        window, ratio, local_base = plan.window, plan.ratio, None
        if plan.axis == "window":
            window = int(value)
        elif plan.axis == "ratio":
            ratio = int(value)
        elif value:
            local_base = float(value)
        fields.update(
            sliding_window=window, local_layers_per_global=ratio, local_rope_theta=local_base
        )
    return ModelConfig.model_validate(fields)


def _name(plan: ContextSweepPlan, value: float | None) -> str:
    if value is None:
        return "control-full-attention"
    return f"{plan.axis}-{value:g}"


def run_context_sweep(
    base: ModelConfig,
    plan: ContextSweepPlan,
    *,
    data_root: Path,
    output_dir: Path,
    should_cancel: Callable[[], bool] | None = None,
    on_point: Callable[[str], None] | None = None,
) -> ScalingReport:
    from epor.evaluation.decoder import load_kernel_scoring_model
    from epor.training import OptimizerSettings, PackedTrainingSettings, pretrain_packed
    from epor.training.run_checkpoint import latest_checkpoint

    from .g6 import run_g6

    points: list[dict[str, Any]] = []
    measured: list[dict[str, Any]] = []
    values: Sequence[float | None] = [None, *plan.values]
    # Every point is validated before any trains, so an impossible value is
    # refused up front instead of after the points before it have run.
    configs = [variant(base, plan, value) for value in values]
    for value, config in zip(values, configs, strict=True):
        if should_cancel is not None and should_cancel():
            raise InterruptedError("context sweep cancelled")
        name = _name(plan, value)
        if on_point is not None:
            on_point(name)
        run = output_dir / name
        began = time.perf_counter()
        manifest = run / "manifest.json"
        done = manifest.is_file() and json.loads(manifest.read_text()).get("status") == "succeeded"
        if not done:
            settings = PackedTrainingSettings(
                tokenized_id=plan.tokenized_id,
                seq_len=plan.seq_len,
                tokens_per_batch=plan.tokens_per_batch,
                data_seed=plan.seed,
                model_seed=plan.seed,
                precision=plan.precision,
                checkpoint_every=plan.max_steps,
                keep_checkpoints=1,
                optimizer=OptimizerSettings(
                    learning_rate=plan.learning_rate, total_steps=plan.max_steps
                ),
            )
            pretrain_packed(
                config,
                data_root,
                run,
                settings,
                max_steps=plan.max_steps,
                should_cancel=should_cancel,
            )
        summary = json.loads(manifest.read_text())
        checkpoint = latest_checkpoint(run)
        assert checkpoint is not None
        model = load_kernel_scoring_model(
            checkpoint, data_root, max_length=config.configured_max_context
        )
        report = run_g6(
            model,
            lengths=[plan.eval_length],
            families=["passkey", "niah-multikey"],
            items_per_cell=plan.items_per_cell,
            seed=plan.seed,
            should_cancel=should_cancel,
        )
        retrieval = {
            family: {
                bucket: report.task(family)
                .metric(f"accuracy.length-{plan.eval_length}.{bucket}")
                .value
                for bucket in ("beginning", "middle", "end")
            }
            for family in ("passkey", "niah-multikey")
        }
        evaluation = summary.get("evaluation") or {}
        points.append(
            {
                "name": name,
                "value": value,
                "config_sha256": config.sha256,
                "sliding_window": config.sliding_window,
                "local_layers_per_global": config.local_layers_per_global,
                "local_rope_theta": config.local_rope_theta,
                "kv_bytes_per_token_bf16": kv_cache_bytes(config, plan.eval_length)
                / plan.eval_length,
                "retrieval": retrieval,
                "g6_report_id": report.report_id,
                "final_loss": summary.get("metrics", {}).get("training_loss"),
                "validation_loss": evaluation.get("main_loss"),
            }
        )
        measured.append({"name": name, "seconds": time.perf_counter() - began, "run": str(run)})
    return seal_scaling_report(
        kind="context-sweep",
        scope="proxy-scale",
        inputs={
            "axis": plan.axis,
            "plan": plan.model_dump(mode="json"),
            "base_config_sha256": base.sha256,
        },
        results={"control": points[0], "points": points[1:]},
        warnings=[
            "Proxy-scale: tiny models trained for a few steps; retrieval near chance is "
            "expected and no ratio, window, or base is shown better here.",
        ],
        citations=[GEMMA3, RULER, PASSKEY],
        measured={"points": measured},
    )


__all__ = ["Axis", "ContextSweepPlan", "run_context_sweep", "variant"]
