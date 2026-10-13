"""AdamW, warmup/cosine schedule, and gradient clipping for the training kernel.

Defaults follow the v0.0.5 roadmap: AdamW with ``β1 = 0.9``, ``β2 = 0.95``,
decoupled weight decay ``0.1``, and global gradient-norm clipping at ``1.0``.
``eps = 1e-8`` is PyTorch's AdamW default (``EPOR-adaptation``: the roadmap
does not pin it).

Weight-decay groups
-------------------
Decay applies to every parameter with two or more dimensions: linear weights
*and* the token embedding (which the tiny recipes tie to the output head, so
it is a projection matrix as well).  One-dimensional parameters (RMSNorm
gains and any biases) are never decayed: shrinking a norm gain toward zero
changes the function's scale instead of regularizing it.  Grouping by
dimensionality rather than by name keeps the rule identical for the dense,
QK-norm, and MTP variants.  Tied parameters appear once.

Schedule
--------
:func:`learning_rate_at` is a pure function of the optimizer step, so resume
needs no scheduler state: the trainer writes the step's rate into every
parameter group before ``optimizer.step()``.  Steps are 1-based.  Warmup is
linear from ``peak / warmup_steps`` to the peak at ``warmup_steps``; cosine
decay then runs to ``min_lr_ratio * peak`` at ``total_steps`` and stays there.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import cast

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch import Tensor, nn


class OptimizerSettings(BaseModel):
    """AdamW, clipping, and schedule hyperparameters; all of them shape the trajectory."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    learning_rate: float = Field(default=3e-4, gt=0)
    min_lr_ratio: float = Field(default=0.1, ge=0, le=1)
    warmup_steps: int = Field(default=0, ge=0)
    total_steps: int = Field(default=1_000, ge=1)
    beta1: float = Field(default=0.9, gt=0, lt=1)
    beta2: float = Field(default=0.95, gt=0, lt=1)
    eps: float = Field(default=1e-8, gt=0)
    weight_decay: float = Field(default=0.1, ge=0)
    gradient_clip: float = Field(default=1.0, gt=0)

    @model_validator(mode="after")
    def warmup_fits(self) -> OptimizerSettings:
        if self.warmup_steps > self.total_steps:
            raise ValueError("warmup_steps must not exceed total_steps")
        return self


def learning_rate_at(step: int, settings: OptimizerSettings) -> float:
    """Learning rate for 1-based optimizer ``step`` (linear warmup, cosine decay)."""

    if step < 1:
        raise ValueError("optimizer steps are 1-based")
    peak = settings.learning_rate
    if settings.warmup_steps and step <= settings.warmup_steps:
        return peak * step / settings.warmup_steps
    floor = peak * settings.min_lr_ratio
    decay_span = settings.total_steps - settings.warmup_steps
    if decay_span <= 0 or step >= settings.total_steps:
        return floor
    progress = (step - settings.warmup_steps) / decay_span
    return floor + (peak - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))


def parameter_groups(model: nn.Module, weight_decay: float) -> list[dict[str, object]]:
    """Split trainable parameters into decayed (ndim >= 2) and undecayed groups."""

    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    seen: set[int] = set()
    for parameter in model.parameters():
        if not parameter.requires_grad or id(parameter) in seen:
            continue
        seen.add(id(parameter))
        (decay if parameter.ndim >= 2 else no_decay).append(parameter)
    return [
        {"params": decay, "weight_decay": weight_decay, "group": "decay"},
        {"params": no_decay, "weight_decay": 0.0, "group": "no_decay"},
    ]


def build_optimizer(model: nn.Module, settings: OptimizerSettings) -> torch.optim.AdamW:
    """AdamW over :func:`parameter_groups`; single-tensor kernels for reproducibility."""

    return torch.optim.AdamW(
        parameter_groups(model, settings.weight_decay),
        lr=settings.learning_rate,
        betas=(settings.beta1, settings.beta2),
        eps=settings.eps,
        weight_decay=settings.weight_decay,
        foreach=False,
    )


def set_learning_rate(optimizer: torch.optim.Optimizer, learning_rate: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = learning_rate


def clip_gradients(parameters: Iterable[Tensor], max_norm: float) -> float:
    """Clip the *global* gradient L2 norm to ``max_norm``; return the pre-clip norm.

    The norm is accumulated in float64 from per-parameter sums of squares in
    parameter order.  Under FSDP2 each gradient is a sharded DTensor: its local
    sums of squares are all-reduced (one collective for all parameters) before
    the same float64 accumulation.  So every topology computes the same clip
    coefficient up to float64 rounding of the sharded partial sums
    (``EPOR-adaptation``: PyTorch's ``clip_grad_norm_`` reduces FP32 shard norms
    in a topology-dependent order, which made FSDP2 and single-process
    trajectories drift at the last FP32 bit whenever clipping was active).
    Like ``clip_grad_norm_``, the coefficient is ``max_norm / (norm + 1e-6)``
    clamped to at most one.
    """

    grads = [parameter.grad for parameter in parameters if parameter.grad is not None]
    if not grads:
        return 0.0
    squares: list[Tensor] = []
    sharded: list[int] = []
    group = None
    for index, grad in enumerate(grads):
        to_local = getattr(grad, "to_local", None)
        if callable(to_local):
            local = cast(Tensor, to_local())
            if any(placement.is_shard() for placement in grad.placements):  # type: ignore[attr-defined]
                sharded.append(index)
                group = grad.device_mesh.get_group()  # type: ignore[attr-defined]
        else:
            local = grad
        squares.append(local.detach().to(torch.float64).pow(2).sum())
    stacked = torch.stack(squares)
    if sharded:
        import torch.distributed as dist

        selection = torch.tensor(sharded, dtype=torch.int64)
        partial = stacked.index_select(0, selection)
        dist.all_reduce(partial, group=group)
        stacked = stacked.index_copy(0, selection, partial)
    total_square = 0.0
    for value in stacked.tolist():
        total_square += value
    total = math.sqrt(total_square)
    coefficient = max_norm / (total + 1e-6)
    if coefficient < 1.0:
        with torch.no_grad():
            for grad in grads:
                grad.mul_(coefficient)
    return total
