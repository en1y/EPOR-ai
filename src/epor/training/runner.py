"""Operator-facing launch of the training kernel: model derivation, topology, stopping.

The CLI and the control plane both start kernel work through this module, so
the rules below hold on every surface:

* **Model recipes stay immutable.**  The kernel needs the tokenized build's
  vocabulary.  A shipped recipe is never edited for that; an operator opts in
  with ``vocab_from_build`` and the run uses a *derived* config whose
  ``vocab_size`` and declared parameter counts come from the build.  The
  derived config has its own hash, recorded in the run manifest and checkpoint;
  the recipe file's hash is unchanged.  QK-norm and MTP may be switched on the
  same way, for ablations against the recipe's control block.
* **One process or a local group.**  ``strategy='none'`` trains in-process.
  ``ddp``/``fsdp2`` spawn ``world_size`` loopback-only ranks with
  :func:`~epor.training.distributed.launch_local`.  No other launcher is
  reachable from an EPOR surface.
* **Stopping is never gated.**  A stop request (a signal handled by the caller,
  or a control-plane cancellation) reaches every rank through one shared event;
  the trainer checkpoints its last completed step and raises
  :class:`~epor.training.loop.TrainingCancelled`.
* **Memory is budgeted.**  Every rank checks the run's certain allocations
  against one host budget before allocating, and caps its own address space
  at its share (:mod:`~epor.training.memory`), so an oversized run is refused
  or stops cleanly instead of reaching the host's OOM killer.
* **Fault injection is not here.**  :class:`~epor.training.faults.FaultPlan`
  is a test instrument passed directly to :func:`pretrain_packed`; no
  operator surface accepts one.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from epor.models.config import ModelConfig, load_model_config, structural_parameter_count
from epor.reference_policy import validate_kernel_model

from .distributed import DistributedStrategy, init_distributed, launch_local
from .kernel import (
    PackedTrainingResult,
    PackedTrainingSettings,
    pretrain_packed,
)
from .loop import TrainingCancelled
from .run_checkpoint import CheckpointError, read_kernel_checkpoint

MAX_LOCAL_WORLD_SIZE = 8
DEFAULT_LAUNCH_TIMEOUT_SECONDS = 6 * 60 * 60.0


class KernelRunError(Exception):
    """A rank's failure, carried back to the launching process by type name."""

    def __init__(self, error_type: str, message: str) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.message = message


def build_vocab_size(data_root: str | Path, tokenized_id: str) -> int:
    from epor.tokenizer.shards import load_tokenized_manifest

    return int(load_tokenized_manifest(Path(data_root), tokenized_id).vocab_size)


def derive_kernel_model(
    recipe: str | Path | ModelConfig,
    *,
    data_root: str | Path,
    tokenized_id: str,
    vocab_from_build: bool = False,
    qk_norm: bool | None = None,
    mtp_depth: int | None = None,
) -> ModelConfig:
    """The model config a kernel run trains: the recipe, or a declared derivation of it.

    Without ``vocab_from_build`` a vocabulary mismatch is refused with a
    message naming the flag, rather than silently resized.
    """

    base = recipe if isinstance(recipe, ModelConfig) else load_model_config(recipe)
    overrides: dict[str, Any] = {}
    build_vocab = build_vocab_size(data_root, tokenized_id)
    if build_vocab != base.vocab_size:
        if not vocab_from_build:
            raise ValueError(
                f"recipe vocab_size {base.vocab_size} differs from the tokenized build's "
                f"{build_vocab}; opt in with --vocab-from-build (vocab_from_build in a job) to "
                "train a derived config with the build's vocabulary; the recipe is never modified"
            )
        overrides["vocab_size"] = build_vocab
    if qk_norm is not None and qk_norm != base.qk_norm:
        overrides["qk_norm"] = qk_norm
    if mtp_depth is not None and mtp_depth != base.mtp_depth:
        overrides["mtp_depth"] = mtp_depth
    if not overrides:
        return validate_kernel_model(base)
    derived = ModelConfig.model_validate({**base.model_dump(mode="json"), **overrides})
    # Declared counts describe the recipe they belong to; a derived config
    # declares its own structure rather than inheriting stale targets.
    count = structural_parameter_count(derived)
    derived = derived.model_copy(
        update={"total_parameters": count, "active_parameters": count, "core_parameters": count}
    )
    return validate_kernel_model(ModelConfig.model_validate(derived.model_dump(mode="json")))


def resume_inputs(
    checkpoint: str | Path, runtime: dict[str, Any] | None = None
) -> tuple[ModelConfig, PackedTrainingSettings, Path, int]:
    """Model, settings, run directory, and step saved in a verified checkpoint.

    ``runtime`` may override only :data:`~epor.training.kernel.RUNTIME_FIELDS`;
    the trainer refuses anything that changes the trajectory.
    """

    from epor.models.config import model_config_from_mapping

    from .kernel import RUNTIME_FIELDS

    trainer = read_kernel_checkpoint(checkpoint)
    if trainer.get("command") == "alignment_dpo":
        raise CheckpointError(
            "this is a preference (DPO) checkpoint; resume it with `epor train preference --resume`"
        )
    config = model_config_from_mapping(trainer.get("model_config"))
    settings = PackedTrainingSettings.model_validate(trainer.get("settings"))
    updates = {key: value for key, value in (runtime or {}).items() if value is not None}
    unknown = set(updates) - RUNTIME_FIELDS
    if unknown:
        raise CheckpointError(
            "resume may change only runtime settings, not: " + ", ".join(sorted(unknown))
        )
    if updates:
        settings = PackedTrainingSettings.model_validate(
            {**settings.model_dump(mode="json"), **updates}
        )
    run_dir = Path(trainer["checkpoint_dir"]).parent.parent
    return config, settings, run_dir, int(trainer["step"])


def _summary(result: PackedTrainingResult, config: ModelConfig, context: dict[str, Any]) -> dict:
    return {
        **result.model_dump(mode="json"),
        "final_loss": result.losses[-1] if result.losses else None,
        "model": {
            "slug": config.slug,
            "config_sha256": config.sha256,
            "vocab_size": config.vocab_size,
            "qk_norm": config.qk_norm,
            "mtp_depth": config.mtp_depth,
            "parameters": structural_parameter_count(config),
        },
        "distributed": context,
    }


def _rank_main(
    rank: int,
    world_size: int,
    strategy: DistributedStrategy,
    arguments: dict[str, Any],
    stop: Any,
) -> dict[str, Any]:
    """One spawned rank: train and report a result or a typed failure."""

    import signal

    # The launching process owns signal handling and forwards a stop through
    # ``stop``; a terminal's Ctrl-C reaching this rank directly must not kill
    # it between checkpoint writes.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    del rank, world_size
    try:
        context = init_distributed(strategy)
        result = pretrain_packed(
            arguments["model"],
            arguments["data_root"],
            arguments["output_dir"],
            arguments["settings"],
            max_steps=arguments["max_steps"],
            resume_from=arguments["resume_from"],
            context=context,
            project_root=arguments["project_root"],
            should_cancel=stop.is_set,
            memory_budget_bytes=arguments["memory_budget_bytes"],
        )
    except Exception as exc:
        return {"ok": False, "type": type(exc).__name__, "message": str(exc)}
    return {"ok": True, "result": result.model_dump(mode="json")}


def run_kernel(
    model: ModelConfig,
    data_root: str | Path,
    output_dir: str | Path,
    settings: PackedTrainingSettings,
    *,
    max_steps: int,
    resume_from: str | Path | None = None,
    strategy: DistributedStrategy = "none",
    world_size: int = 1,
    project_root: str | Path | None = None,
    should_cancel: Callable[[], bool] | None = None,
    on_step: Callable[[int, int, float], None] | None = None,
    launch_timeout_seconds: float = DEFAULT_LAUNCH_TIMEOUT_SECONDS,
    stop_event: Any = None,
    memory_budget_bytes: int | None = None,
) -> dict[str, Any]:
    """Train in-process (``none``) or across a local loopback group; return a JSON summary.

    Distributed ranks raise :class:`KernelRunError` here with the original
    exception's type name, so callers map failures identically for every
    topology.  For a distributed run ``stop_event`` (a spawn-context event)
    is the stop channel; ``should_cancel`` and ``on_step`` apply in-process.
    ``memory_budget_bytes`` is the host budget for all ranks together (see
    :mod:`epor.training.memory`); each rank caps itself at its share.
    """

    if strategy == "none":
        if world_size != 1:
            raise ValueError("strategy 'none' runs exactly one process")
        result = pretrain_packed(
            model,
            data_root,
            output_dir,
            settings,
            max_steps=max_steps,
            resume_from=resume_from,
            project_root=project_root,
            should_cancel=should_cancel,
            on_step=on_step,
            memory_budget_bytes=memory_budget_bytes,
        )
        return _summary(result, model, {"strategy": "none", "world_size": 1})
    if not 1 <= world_size <= MAX_LOCAL_WORLD_SIZE:
        raise ValueError(f"world_size must be between 1 and {MAX_LOCAL_WORLD_SIZE}")
    import torch.multiprocessing as mp

    stop = stop_event if stop_event is not None else mp.get_context("spawn").Event()
    arguments = {
        "model": model,
        "data_root": str(data_root),
        "output_dir": str(output_dir),
        "settings": settings,
        "max_steps": max_steps,
        "resume_from": None if resume_from is None else str(resume_from),
        "project_root": None if project_root is None else str(project_root),
        "memory_budget_bytes": memory_budget_bytes,
    }
    threads = max(1, (os.cpu_count() or world_size) // world_size)
    ranks = launch_local(
        _rank_main,
        world_size,
        strategy,
        arguments,
        stop,
        timeout_seconds=launch_timeout_seconds,
        threads_per_rank=threads,
    )
    failures = [outcome for outcome in ranks if not outcome["ok"]]
    if failures:
        # A rank that hit its memory cap leaves its peers to fail in a
        # collective; report the cause, not the symptom.
        failure = next(
            (item for item in failures if item["type"] == "MemoryBudgetExceededError"),
            failures[0],
        )
        raise KernelRunError(str(failure["type"]), str(failure["message"]))
    result = PackedTrainingResult.model_validate(ranks[0]["result"])
    return _summary(
        result,
        model,
        {"strategy": strategy, "world_size": world_size, "threads_per_rank": threads},
    )


def is_cancellation(exc: BaseException) -> bool:
    return isinstance(exc, TrainingCancelled) or (
        isinstance(exc, KernelRunError) and exc.error_type == "TrainingCancelled"
    )


__all__ = [
    "DEFAULT_LAUNCH_TIMEOUT_SECONDS",
    "MAX_LOCAL_WORLD_SIZE",
    "KernelRunError",
    "build_vocab_size",
    "derive_kernel_model",
    "is_cancellation",
    "resume_inputs",
    "run_kernel",
]
