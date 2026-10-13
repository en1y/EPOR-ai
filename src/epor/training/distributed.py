"""Launcher-agnostic process-group setup, model wrapping, and a local launcher.

Strategies
----------
``none``
    One process, no process group.
``ddp``
    :class:`~torch.nn.parallel.DistributedDataParallel`: replicated parameters,
    gradients averaged across ranks.  Non-final accumulation micro-batches run
    under ``no_sync``.
``fsdp2``
    ``torch.distributed.fsdp.fully_shard`` applied to every decoder block, every
    MTP depth's decoder block, and then the root (embeddings, final norm, MTP
    projections and norms, and the tied output head stay in the root group so
    the tie survives sharding).  Non-final
    micro-batches disable gradient reduction with
    ``set_requires_gradient_sync(False)``.  Verified to run on CPU with the
    ``gloo`` backend in PyTorch 2.13.

Both distributed strategies keep FP32 parameters, gradients, and reductions;
BF16 is applied by CPU autocast in the trainer, identically for every strategy.

Process groups come from the standard ``env://`` variables (``RANK``,
``WORLD_SIZE``, ``MASTER_ADDR``, ``MASTER_PORT``), so ``torchrun`` and the
local :func:`launch_local` helper start the same code.
"""

from __future__ import annotations

import contextlib
import os
import queue
import socket
import time
import traceback
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Literal, cast

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

DistributedStrategy = Literal["none", "ddp", "fsdp2"]
DEFAULT_TIMEOUT_SECONDS = 120.0


@dataclass(frozen=True, slots=True)
class DistributedContext:
    """Rank identity and the strategy the trainer applies to the model."""

    strategy: DistributedStrategy = "none"
    rank: int = 0
    world_size: int = 1

    def __post_init__(self) -> None:
        if self.world_size < 1 or not 0 <= self.rank < self.world_size:
            raise ValueError("rank must satisfy 0 <= rank < world_size")
        if self.strategy == "none" and self.world_size != 1:
            raise ValueError("strategy 'none' requires world_size 1")
        if self.strategy != "none" and not dist.is_initialized():
            raise RuntimeError(f"strategy {self.strategy!r} requires an initialized process group")

    @property
    def distributed(self) -> bool:
        return self.strategy != "none"

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    def barrier(self) -> None:
        if self.distributed:
            dist.barrier()

    def gather_objects(self, value: Any) -> list[Any]:
        """Every rank's ``value`` in rank order (a one-element list without a group)."""

        if not self.distributed:
            return [value]
        gathered: list[Any] = [None] * self.world_size
        dist.all_gather_object(gathered, value)
        return gathered

    def broadcast_object(self, value: Any) -> Any:
        """Rank 0's ``value`` on every rank."""

        if not self.distributed:
            return value
        box = [value]
        dist.broadcast_object_list(box, src=0)
        return box[0]

    def any(self, flag: bool) -> bool:
        """True on every rank when ``flag`` is true on any rank."""

        if not self.distributed:
            return flag
        tensor = torch.tensor([1 if flag else 0], dtype=torch.int64)
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
        return bool(tensor.item())


def init_distributed(
    strategy: DistributedStrategy,
    *,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> DistributedContext:
    """Join (or create) the ``env://`` gloo process group for ``strategy``.

    ``strategy='none'`` never touches ``torch.distributed``.  An already
    initialized group is reused, so a launcher may set it up itself.
    """

    if strategy == "none":
        return DistributedContext()
    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available in this build")
    if not dist.is_initialized():
        missing = [
            name
            for name in ("RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT")
            if name not in os.environ
        ]
        if missing:
            raise RuntimeError(
                f"distributed strategy {strategy!r} needs env:// variables: {', '.join(missing)}"
            )
        dist.init_process_group(
            backend="gloo",
            init_method="env://",
            timeout=timedelta(seconds=timeout_seconds),
        )
    return DistributedContext(
        strategy=strategy, rank=dist.get_rank(), world_size=dist.get_world_size()
    )


def destroy_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def wrap_model(model: nn.Module, context: DistributedContext) -> nn.Module:
    """Apply the context's strategy to a fully initialized CPU model."""

    if context.strategy == "none":
        return model
    if context.strategy == "ddp":
        from torch.nn.parallel import DistributedDataParallel

        return DistributedDataParallel(model)
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard

    mesh = init_device_mesh("cpu", (context.world_size,))
    blocks = getattr(model, "blocks", None)
    if isinstance(blocks, nn.ModuleList):
        for block in blocks:
            fully_shard(block, mesh=mesh)
    mtp_modules = getattr(model, "mtp_modules", None)
    if isinstance(mtp_modules, nn.ModuleList):
        # The decoder calls an MTP module's ``combine``/``output_norm`` directly
        # rather than its ``forward``, so only the inner block (entered through
        # ``forward``) gets its own unit; the small projections stay with root.
        for module in mtp_modules:
            fully_shard(cast(nn.Module, module.block), mesh=mesh)
    return cast(nn.Module, fully_shard(model, mesh=mesh))


def unwrap_model(model: nn.Module) -> nn.Module:
    """The user module beneath a DDP wrapper (FSDP2 shards in place)."""

    from torch.nn.parallel import DistributedDataParallel

    return model.module if isinstance(model, DistributedDataParallel) else model


@contextlib.contextmanager
def gradient_sync(model: nn.Module, enabled: bool) -> Iterator[None]:
    """Skip cross-rank gradient reduction for a non-final accumulation micro-batch."""

    from torch.nn.parallel import DistributedDataParallel

    if isinstance(model, DistributedDataParallel):
        if enabled:
            yield
        else:
            with model.no_sync():
                yield
        return
    setter = getattr(model, "set_requires_gradient_sync", None)
    if callable(setter):
        setter(enabled)
        try:
            yield
        finally:
            setter(True)
        return
    yield


def free_port() -> int:
    """An ephemeral localhost TCP port for a local rendezvous."""

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _local_worker(
    rank: int,
    world_size: int,
    port: int,
    threads: int | None,
    function: Callable[..., Any],
    args: tuple[Any, ...],
    results: Any,
) -> None:
    os.environ.update(
        RANK=str(rank),
        WORLD_SIZE=str(world_size),
        LOCAL_RANK=str(rank),
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
    )
    if threads is not None:
        torch.set_num_threads(threads)
    try:
        value = function(rank, world_size, *args)
        results.put((rank, True, value))
    except BaseException as exc:  # report every failure to the parent
        results.put((rank, False, f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"))
    finally:
        destroy_distributed()


def launch_local(
    function: Callable[..., Any],
    world_size: int,
    *args: Any,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    threads_per_rank: int | None = None,
) -> list[Any]:
    """Run ``function(rank, world_size, *args)`` in ``world_size`` spawned processes.

    Each process receives ``env://`` variables for a fresh localhost port; the
    function itself calls :func:`init_distributed`.  Return values (which must
    pickle) come back in rank order.  A rank that raises, exits, or outlives
    ``timeout_seconds`` fails the launch, and every process is terminated, so a
    broken collective cannot hang the caller.
    """

    if world_size < 1:
        raise ValueError("world_size must be positive")
    context = mp.get_context("spawn")
    results = context.Queue()
    port = free_port()
    processes = [
        context.Process(
            target=_local_worker,
            args=(rank, world_size, port, threads_per_rank, function, args, results),
            daemon=True,
        )
        for rank in range(world_size)
    ]
    for process in processes:
        process.start()
    collected: dict[int, Any] = {}
    failures: list[str] = []
    try:
        deadline = time.monotonic() + timeout_seconds
        while len(collected) + len(failures) < world_size:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"distributed launch exceeded {timeout_seconds:.0f}s")
            try:
                rank, ok, value = results.get(timeout=min(remaining, 1.0))
            except queue.Empty:
                dead = [
                    process.exitcode for process in processes if process.exitcode not in (None, 0)
                ]
                if dead and results.empty():
                    raise RuntimeError(
                        f"a distributed rank exited with code {dead[0]} without a result"
                    ) from None
                continue
            if ok:
                collected[rank] = value
            else:
                failures.append(f"rank {rank}: {value}")
                break
        if failures:
            raise RuntimeError("distributed launch failed\n" + "\n".join(failures))
        for process in processes:
            process.join(timeout=max(1.0, deadline - time.monotonic()))
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
        results.close()
    return [collected[rank] for rank in range(world_size)]
