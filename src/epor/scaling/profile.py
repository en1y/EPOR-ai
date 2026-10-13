"""Host hardware profiling: measured training-step throughput, memory, and MFU.

:func:`profile_training_step` builds the configuration on the CPU, runs
``warmup_steps`` untimed and ``timed_steps`` timed training steps (forward,
backward, gradient clipping, AdamW step) on seeded synthetic token batches of
``batch_size x seq_len``, and reports:

* measured: tokens/second, per-step seconds (mean/min/max and every step),
  achieved FLOP/s (analytic training FLOPs per token x measured tokens/s),
  process peak RSS (the lifetime high-water mark of this process, so it
  includes anything allocated before the profile) and RSS after the run, the
  host snapshot, and MFU;
* content: the config hash and counts, batch geometry, precision, seeds,
  thread count, the deterministic-algorithms setting, and the analytic FLOPs
  model (:mod:`epor.scaling.flops`).

MFU is ``achieved / peak``.  The device peak is an operator input
(``peak_flops_per_second``): EPOR never looks it up or guesses it, and a CPU's
peak depends on SIMD width, clocks, and core count that this module does not
measure.  Without it MFU is recorded as ``unknown``.

Synthetic tokens measure compute throughput only; data loading is measured by
the sweep's training runs.  The profile allocates only configs the local
training kernel accepts (:func:`epor.reference_policy.validate_kernel_model`)
and is cancellable between steps; it never provisions anything.
"""

from __future__ import annotations

import contextlib
import resource
import sys
import time
from collections.abc import Callable
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from epor.models.config import ModelConfig
from epor.reference_policy import validate_kernel_model, validate_kernel_run

from .flops import estimate_flops, parameter_counts
from .report import KAPLAN_2020, ScalingReport, ScalingScope, seal_scaling_report

MAX_PROFILE_STEPS = 1_000


class ProfileSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    batch_size: int = Field(default=4, ge=1)
    seq_len: int = Field(default=128, ge=2)
    warmup_steps: int = Field(default=2, ge=0)
    timed_steps: int = Field(default=5, ge=1)
    precision: Literal["fp32", "bf16"] = "fp32"
    seed: int = Field(default=0, ge=0)
    threads: int | None = Field(default=None, ge=1)
    peak_flops_per_second: float | None = Field(default=None, gt=0)
    device_label: str = Field(default="cpu", min_length=1, max_length=120)


def _peak_rss_bytes() -> int:
    maximum = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(maximum if sys.platform == "darwin" else maximum * 1024)


def _current_rss_bytes() -> int | None:
    try:
        import psutil
    except ImportError:  # pragma: no cover - psutil is a core dependency
        return None
    return int(psutil.Process().memory_info().rss)


def profile_training_step(
    config: ModelConfig,
    settings: ProfileSettings,
    *,
    should_cancel: Callable[[], bool] | None = None,
    scope: ScalingScope = "host-measurement",
) -> ScalingReport:
    """Measure one configuration's training-step throughput on this host."""

    import torch

    from epor.models.reference import EporDecoder
    from epor.training.loop import TrainingCancelled
    from epor.training.manifest import hardware_snapshot, utc_now
    from epor.training.optim import OptimizerSettings, build_optimizer, clip_gradients

    validate_kernel_model(config)
    total_steps = settings.warmup_steps + settings.timed_steps
    if total_steps > MAX_PROFILE_STEPS:
        raise ValueError(f"a profile runs at most {MAX_PROFILE_STEPS} steps")
    batch_tokens = settings.batch_size * settings.seq_len
    validate_kernel_run(total_steps, batch_tokens)
    if settings.seq_len > config.configured_max_context:
        raise ValueError("seq_len exceeds the config's configured_max_context")

    flops = estimate_flops(config, settings.seq_len)
    counts = parameter_counts(config)
    threads_before = torch.get_num_threads()
    deterministic_before = torch.are_deterministic_algorithms_enabled()
    step_seconds: list[float] = []
    try:
        if settings.threads is not None:
            torch.set_num_threads(settings.threads)
        torch.use_deterministic_algorithms(True)
        threads = torch.get_num_threads()
        model = EporDecoder(config, seed=settings.seed, device="cpu", dtype=torch.float32)
        model.train()
        optimizer = build_optimizer(model, OptimizerSettings())
        generator = torch.Generator(device="cpu")
        generator.manual_seed(settings.seed)
        # Token IDs above the reserved specials, so every row is ordinary text.
        low = max(config.pad_token_id, config.bos_token_id, config.eos_token_id) + 1
        autocast = (
            torch.autocast("cpu", dtype=torch.bfloat16)
            if settings.precision == "bf16"
            else contextlib.nullcontext()
        )
        for step in range(total_steps):
            if should_cancel is not None and should_cancel():
                raise TrainingCancelled("profile cancellation requested")
            tokens = torch.randint(
                low,
                config.vocab_size,
                (settings.batch_size, settings.seq_len + 1),
                generator=generator,
            )
            began = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            with autocast:
                output = model(tokens[:, :-1], tokens[:, 1:])
            assert output.loss is not None
            output.loss.backward()
            clip_gradients(model.parameters(), 1.0)
            optimizer.step()
            elapsed = time.perf_counter() - began
            if step >= settings.warmup_steps:
                step_seconds.append(elapsed)
    finally:
        torch.use_deterministic_algorithms(deterministic_before)
        torch.set_num_threads(threads_before)

    timed = sum(step_seconds)
    tokens_per_second = batch_tokens * len(step_seconds) / timed if timed > 0 else None
    achieved = (
        tokens_per_second * flops.train_flops_per_token if tokens_per_second is not None else None
    )
    peak = settings.peak_flops_per_second
    mfu = achieved / peak if achieved is not None and peak is not None else None
    measured = {
        "recorded_at": utc_now(),
        "host": hardware_snapshot(),
        "torch_threads": threads,
        "tokens_per_second": tokens_per_second,
        "step_seconds_mean": timed / len(step_seconds),
        "step_seconds_min": min(step_seconds),
        "step_seconds_max": max(step_seconds),
        "step_seconds": step_seconds,
        "achieved_flops_per_second": achieved,
        "mfu": mfu,
        "mfu_status": "measured" if mfu is not None else "unknown",
        "mfu_note": (
            "achieved analytic FLOP/s divided by the operator-supplied device peak"
            if mfu is not None
            else "unknown: no operator-supplied peak FLOP/s for this device"
        ),
        "process_peak_rss_bytes": _peak_rss_bytes(),
        "rss_after_bytes": _current_rss_bytes(),
    }
    inputs = {
        "model": {
            "slug": config.slug,
            "config_sha256": config.sha256,
            "vocab_size": config.vocab_size,
            "qk_norm": config.qk_norm,
            "mtp_depth": config.mtp_depth,
        },
        "settings": settings.model_dump(mode="json"),
        "batch_tokens": batch_tokens,
        "determinism": {
            "torch_deterministic_algorithms": True,
            "torch_threads": threads,
            "synthetic_tokens_seed": settings.seed,
            "model_seed": settings.seed,
        },
    }
    results = {
        "parameters": counts.model_dump(mode="json"),
        "flops": flops.model_dump(mode="json"),
        "flops_label": "disclosed formula, EPOR-adaptation for logits/MoE/MTP terms",
        "peak_flops_per_second": peak,
        "peak_flops_label": "operator-input" if peak is not None else "unknown",
        "timed_steps": len(step_seconds),
        "measured_quantities": [
            "tokens_per_second",
            "step_seconds",
            "achieved_flops_per_second",
            "mfu",
            "process_peak_rss_bytes",
        ],
    }
    warnings = [
        "Throughput is measured on synthetic tokens on this host's CPU and does not "
        "predict accelerator throughput."
    ]
    if peak is None:
        warnings.append("MFU is unknown: no operator-supplied peak FLOP/s.")
    return seal_scaling_report(
        kind="hardware-profile",
        scope=scope,
        inputs=inputs,
        results=results,
        warnings=warnings,
        citations=[KAPLAN_2020],
        measured=measured,
    )


__all__ = ["MAX_PROFILE_STEPS", "ProfileSettings", "profile_training_step"]
