"""Atomic, verifiable training-kernel checkpoints over ``torch.distributed.checkpoint``.

Layout
------
A run keeps its checkpoints beneath ``<output>/checkpoints``::

    checkpoints/
      latest.json                 # {"name": "step-00000040", "step": 40, ...}
      step-00000040/
        state/                    # DCP: model + optimizer (sharded or not)
        rng-rank00000.pt          # that rank's torch CPU and Python RNG states
        trainer.json              # step, counters, data cursor, settings, config
        checkpoint.json           # sha256 + size of every file above, written last

Model and optimizer state go through ``get_state_dict``/``set_state_dict``, so
keys are the plain module FQNs whatever the wrapper (none, DDP, or FSDP2) and
DCP reshards on load: a checkpoint saved by one world size or strategy loads
into another.  Single-process runs use the same format without a process
group.  (The v0.0.1 debug ``pretrain`` path keeps its own single-file format.)

Atomicity
---------
Every rank writes into ``.tmp-<name>-<uuid>/``.  After all ranks finish, rank 0
writes ``trainer.json`` and the hashed ``checkpoint.json``, renames the
directory into place, and only then atomically replaces ``latest.json``.  A
crash at any earlier point leaves ``latest.json`` naming the previous complete
checkpoint; abandoned temporary directories are removed at the next start.
Loading re-verifies every hash before any tensor is read.
"""

from __future__ import annotations

import contextlib
import json
import random
import shutil
import uuid
import warnings
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    get_state_dict,
    set_model_state_dict,
    set_state_dict,
)
from torch.distributed.checkpoint.state_dict_loader import load as dcp_load
from torch.distributed.checkpoint.state_dict_saver import save as dcp_save

from .checkpoint import atomic_write_json
from .data import sha256_file
from .distributed import DistributedContext

KERNEL_CHECKPOINT_SCHEMA = "epor-kernel-checkpoint-v1"
LATEST_FILE = "latest.json"
TRAINER_FILE = "trainer.json"
MANIFEST_FILE = "checkpoint.json"
STATE_DIR = "state"
_TEMP_PREFIX = ".tmp-"


class CheckpointError(ValueError):
    """A checkpoint is missing, incomplete, corrupt, or incompatible with the run."""


@contextlib.contextmanager
def _single_process_quiet() -> Iterator[None]:
    """Silence DCP's warning that a deliberately process-group-free save is local."""

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="torch.distributed is disabled, unavailable or uninitialized",
            category=UserWarning,
        )
        yield


def checkpoint_name(step: int) -> str:
    return f"step-{step:08d}"


def _rng_file(rank: int) -> str:
    return f"rng-rank{rank:05d}.pt"


def _file_records(directory: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(p for p in directory.rglob("*") if p.is_file()):
        relative = path.relative_to(directory).as_posix()
        if relative == MANIFEST_FILE:
            continue
        records.append(
            {"path": relative, "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
        )
    return records


def remove_abandoned(root: Path) -> list[Path]:
    """Delete temporary checkpoint directories left behind by a crash."""

    removed: list[Path] = []
    if not root.is_dir():
        return removed
    for candidate in root.iterdir():
        if candidate.is_dir() and candidate.name.startswith(_TEMP_PREFIX):
            shutil.rmtree(candidate)
            removed.append(candidate)
    return removed


def _apply_retention(root: Path, keep: int) -> None:
    complete = sorted(
        path for path in root.glob("step-*") if path.is_dir() and (path / MANIFEST_FILE).is_file()
    )
    for stale in complete[: max(0, len(complete) - keep)]:
        shutil.rmtree(stale)


def save_kernel_checkpoint(
    root: str | Path,
    *,
    step: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    trainer_state: dict[str, Any],
    context: DistributedContext,
    keep_last: int,
    before_commit: Callable[[], None] | None = None,
) -> Path:
    """Write one checkpoint collectively; every rank must call it at the same step.

    ``before_commit`` runs on every rank after all state is on disk and before
    the directory is promoted (the fault-injection point for a crash during a
    checkpoint write).  Returns the promoted checkpoint directory.
    """

    if keep_last < 1:
        raise ValueError("keep_last must be at least 1")
    checkpoints = Path(root)
    name = checkpoint_name(step)
    temporary_name = None
    if context.is_main:
        checkpoints.mkdir(parents=True, exist_ok=True)
        temporary_name = f"{_TEMP_PREFIX}{name}-{uuid.uuid4().hex}"
        (checkpoints / temporary_name).mkdir()
    temporary = checkpoints / str(context.broadcast_object(temporary_name))

    model_state, optimizer_state = get_state_dict(model, optimizer)
    with _single_process_quiet():
        dcp_save(
            {"model": model_state, "optimizer": optimizer_state},
            checkpoint_id=temporary / STATE_DIR,
            no_dist=not context.distributed,
        )
    torch.save(
        {"torch_cpu": torch.get_rng_state(), "python": list(_python_state_as_lists())},
        temporary / _rng_file(context.rank),
    )
    context.barrier()
    if before_commit is not None:
        before_commit()
        # Promotion waits for every rank to pass the hook, so a rank that dies
        # inside it can never leave a promoted checkpoint behind.
        context.barrier()
    final = checkpoints / name
    if context.is_main:
        atomic_write_json(
            temporary / TRAINER_FILE,
            {**trainer_state, "schema_version": KERNEL_CHECKPOINT_SCHEMA, "step": step},
        )
        atomic_write_json(
            temporary / MANIFEST_FILE,
            {
                "schema_version": KERNEL_CHECKPOINT_SCHEMA,
                "step": step,
                "world_size": context.world_size,
                "files": _file_records(temporary),
            },
        )
        if final.exists():
            # A previous attempt promoted this step but crashed before
            # updating latest.json; the new, complete copy supersedes it.
            stale = checkpoints / f"{_TEMP_PREFIX}superseded-{uuid.uuid4().hex}"
            final.rename(stale)
            temporary.rename(final)
            shutil.rmtree(stale)
        else:
            temporary.rename(final)
        atomic_write_json(
            checkpoints / LATEST_FILE,
            {
                "schema_version": KERNEL_CHECKPOINT_SCHEMA,
                "name": name,
                "step": step,
                "checkpoint_sha256": sha256_file(final / MANIFEST_FILE),
            },
        )
        _apply_retention(checkpoints, keep_last)
    context.barrier()
    return final


def _python_state_as_lists() -> tuple[int, list[int], float | None]:
    version, internal, gauss = random.getstate()
    return version, list(internal), gauss


def resolve_checkpoint(path: str | Path) -> Path:
    """A checkpoint directory from itself, a ``checkpoints`` root, or a run directory."""

    candidate = Path(path).expanduser().resolve()
    for root in (candidate, candidate / "checkpoints"):
        latest = root / LATEST_FILE
        if latest.is_file():
            try:
                pointer = json.loads(latest.read_text(encoding="utf-8"))
                name = str(pointer["name"])
            except (OSError, ValueError, KeyError, TypeError) as exc:
                raise CheckpointError(f"latest-checkpoint pointer is unreadable: {latest}") from exc
            if not name.startswith("step-") or "/" in name:
                raise CheckpointError("latest-checkpoint pointer names an invalid directory")
            resolved = root / name
            if (resolved / MANIFEST_FILE).is_file():
                expected = pointer.get("checkpoint_sha256")
                if expected is not None and sha256_file(resolved / MANIFEST_FILE) != expected:
                    raise CheckpointError("latest-checkpoint pointer does not match its manifest")
            return resolved
    return candidate


def latest_checkpoint(run_dir: str | Path) -> Path | None:
    """The last complete checkpoint of a run, or ``None`` before the first one."""

    root = Path(run_dir).expanduser().resolve() / "checkpoints"
    if not (root / LATEST_FILE).is_file():
        return None
    return resolve_checkpoint(root)


def read_kernel_checkpoint(path: str | Path, *, verify: bool = True) -> dict[str, Any]:
    """Verify a checkpoint's file hashes and return its trainer state."""

    directory = resolve_checkpoint(path)
    manifest_path = directory / MANIFEST_FILE
    if not manifest_path.is_file():
        raise CheckpointError(f"not a complete training checkpoint: {directory}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        trainer = json.loads((directory / TRAINER_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise CheckpointError(f"checkpoint metadata is unreadable: {directory}") from exc
    if (
        manifest.get("schema_version") != KERNEL_CHECKPOINT_SCHEMA
        or trainer.get("schema_version") != KERNEL_CHECKPOINT_SCHEMA
    ):
        raise CheckpointError("unsupported training-checkpoint schema")
    if verify:
        recorded = manifest.get("files")
        if not isinstance(recorded, list) or recorded != _file_records(directory):
            raise CheckpointError(
                f"checkpoint files do not match their recorded hashes: {directory}"
            )
    if trainer.get("step") != manifest.get("step"):
        raise CheckpointError("checkpoint trainer state and manifest disagree on the step")
    trainer["checkpoint_dir"] = str(directory)
    trainer["saved_world_size"] = manifest.get("world_size")
    return trainer


def load_kernel_checkpoint(
    directory: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    context: DistributedContext,
    restore_rng: bool = True,
) -> None:
    """Load model (and optimizer) state into an already wrapped model, collectively.

    The caller verifies the directory with :func:`read_kernel_checkpoint`
    first.  A rank without its own RNG file (the world size grew) restores
    rank 0's states; the trainer reseeds per step anyway, so this only matters
    for code that draws random numbers outside a step.
    """

    path = Path(directory)
    if optimizer is None:
        model_only = {"model": get_model_state_dict(model)}
        with _single_process_quiet():
            dcp_load(model_only, checkpoint_id=path / STATE_DIR, no_dist=not context.distributed)
        set_model_state_dict(model, model_only["model"])
    else:
        model_state, optimizer_state = get_state_dict(model, optimizer)
        state: dict[str, Any] = {"model": model_state, "optimizer": optimizer_state}
        with _single_process_quiet():
            dcp_load(state, checkpoint_id=path / STATE_DIR, no_dist=not context.distributed)
        set_state_dict(
            model,
            optimizer,
            model_state_dict=state["model"],
            optim_state_dict=state["optimizer"],
        )
    if not restore_rng:
        return
    rng_path = path / _rng_file(context.rank)
    if not rng_path.is_file():
        rng_path = path / _rng_file(0)
    rng = torch.load(rng_path, map_location="cpu", weights_only=True)
    torch.set_rng_state(rng["torch_cpu"])
    version, internal, gauss = rng["python"]
    random.setstate((version, tuple(internal), gauss))
