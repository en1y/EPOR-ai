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


def guarded_worker(rank: int, world_size: int, run: dict[str, Any]) -> dict[str, Any]:
    """Train one run (as :func:`train_worker` describes, plus ``"budget"``) and report how it ended.

    The soft ``RLIMIT_AS`` is read before training, at every cancellation poll
    during it, and after it, so a test sees the cap each rank set and that it
    was released.
    """

    import resource

    def soft_limit() -> int:
        return resource.getrlimit(resource.RLIMIT_AS)[0]

    before = soft_limit()
    during: set[int] = set()

    def observe() -> bool:
        during.add(soft_limit())
        return False

    error: str | None = None
    try:
        pretrain_packed(
            run["config"],
            run["data_root"],
            run["output"],
            run["settings"],
            max_steps=run["max_steps"],
            context=init_distributed(run["strategy"]),
            project_root=PROJECT_ROOT,
            should_cancel=observe,
            fault_plan=run.get("fault"),
            memory_budget_bytes=run["budget"],
        )
    except Exception as exc:
        error = type(exc).__name__
    del rank, world_size
    return {"error": error, "before": before, "during": sorted(during), "after": soft_limit()}


def failing_worker(rank: int, world_size: int) -> None:
    raise ValueError(f"rank {rank} of {world_size} failed deliberately")


def sleeping_worker(rank: int, world_size: int, seconds: float) -> int:
    import time

    time.sleep(seconds)
    return rank


def _listening_inodes(table: str) -> dict[str, str]:
    """``{socket inode: local address}`` for every listening socket in a /proc table."""

    listening: dict[str, str] = {}
    path = Path("/proc/net") / table
    if not path.is_file():
        return listening
    for line in path.read_text(encoding="ascii").splitlines()[1:]:
        fields = line.split()
        if fields[3] == "0A":  # TCP_LISTEN
            listening[fields[9]] = fields[1]
    return listening


def listening_addresses_worker(rank: int, world_size: int) -> list[str]:
    """Every TCP address this rank listens on, read while the group is live."""

    import os

    from epor.training import init_distributed

    context = init_distributed("ddp")
    tensor = torch.ones(1)
    torch.distributed.all_reduce(tensor)
    assert float(tensor) == world_size
    listening = {**_listening_inodes("tcp"), **_listening_inodes("tcp6")}
    fd_dir = Path(f"/proc/{os.getpid()}/fd")
    addresses: list[str] = []
    for entry in fd_dir.iterdir():
        try:
            target = os.readlink(entry)
        except OSError:
            continue
        if target.startswith("socket:["):
            inode = target[len("socket:[") : -1]
            if inode in listening:
                addresses.append(listening[inode])
    context.barrier()
    del rank
    return addresses
