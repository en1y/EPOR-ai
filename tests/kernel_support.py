"""Shared helpers for training-kernel tests, importable by spawned ranks.

``launch_local`` pickles worker functions by module and name, so every
function a spawned rank runs lives here at module level, not in a test file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from epor.models import ModelConfig, load_model_config
from epor.training import (
    FaultPlan,
    OptimizerSettings,
    PackedTrainingSettings,
    init_distributed,
    pretrain_packed,
)
from epor.training.distributed import DistributedStrategy

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TINY_CONFIG = PROJECT_ROOT / "configs" / "models" / "epor-tiny.yaml"


def kernel_config(vocab_size: int, **overrides: Any) -> ModelConfig:
    """The tiny recipe with the tokenized build's vocabulary (plus feature flags)."""

    base = load_model_config(TINY_CONFIG)
    return ModelConfig.model_validate(
        {**base.model_dump(mode="json"), "vocab_size": vocab_size, **overrides}
    )


def kernel_settings(tokenized_id: str, **overrides: Any) -> PackedTrainingSettings:
    """Small FP32 settings: 16-token rows, two rows per step, clipping active early."""

    optimizer = overrides.pop("optimizer", None) or OptimizerSettings(
        learning_rate=3e-3, warmup_steps=2, total_steps=8
    )
    values: dict[str, Any] = {
        "tokenized_id": tokenized_id,
        "seq_len": 16,
        "tokens_per_batch": 32,
        "micro_batch_size": 1,
        "precision": "fp32",
        "checkpoint_every": 2,
        "validation_split": None,
        "optimizer": optimizer,
        **overrides,
    }
    return PackedTrainingSettings.model_validate(values)


def same_weights(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> bool:
    return left.keys() == right.keys() and all(torch.equal(left[k], right[k]) for k in left)


def max_weight_difference(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> float:
    return max(float((left[k] - right[k]).abs().max()) for k in left)


def train_worker(
    rank: int,
    world_size: int,
    runs: list[dict[str, Any]],
) -> list[tuple[float, ...]]:
    """Run each described training job in order inside one process group.

    A run is ``{"strategy", "config", "data_root", "output", "settings",
    "max_steps", optional "resume_from", optional "fault"}``; reusing one
    spawn for several runs keeps the distributed tests fast.
    """

    del rank, world_size
    losses: list[tuple[float, ...]] = []
    contexts: dict[DistributedStrategy, Any] = {}
    for run in runs:
        strategy: DistributedStrategy = run["strategy"]
        if strategy not in contexts:
            contexts[strategy] = init_distributed(strategy)
        fault: FaultPlan | None = run.get("fault")
        result = pretrain_packed(
            run["config"],
            run["data_root"],
            run["output"],
            run["settings"],
            max_steps=run["max_steps"],
            resume_from=run.get("resume_from"),
            context=contexts[strategy],
            project_root=PROJECT_ROOT,
            fault_plan=fault,
        )
        losses.append(result.losses)
    return losses


def failing_worker(rank: int, world_size: int) -> None:
    raise ValueError(f"rank {rank} of {world_size} failed deliberately")


def sleeping_worker(rank: int, world_size: int, seconds: float) -> int:
    import time

    time.sleep(seconds)
    return rank
