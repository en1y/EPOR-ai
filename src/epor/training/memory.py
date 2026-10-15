"""Host-memory guard for kernel training: a pre-flight lower bound and a soft address-space cap.

An unguarded kernel step larger than the host's memory reaches the Linux OOM
killer, which kills the run (and possibly the session that started it) without
a manifest.  :func:`~epor.training.kernel.pretrain_packed` therefore applies
two checks to every run, on every surface:

1. **Pre-flight.**  Before it opens data or allocates anything, the run's
   certain allocations (:func:`epor.context.probe.certain_components`, the
   estimator the context laboratory's probes use) are compared with the
   memory budget: weights, gradients and AdamW moments per replica, the logits
   (and the BF16 copy under autocast) for every row in flight on the host, MTP
   logits, and the boolean attention masks.  A run whose lower bound already
   exceeds the budget is refused with :class:`MemoryBudgetRefusedError`,
   naming the component that dominates.  The bound leaves out activations,
   temporaries, allocator overhead, distillation teachers, and the sparse
   indexer, so passing it does not mean the run fits; activation checkpointing
   does not lower it.
2. **Runtime cap.**  Each training process (each local rank under
   :func:`~epor.training.distributed.launch_local`) lowers its *soft*
   ``RLIMIT_AS`` to its address-space size when training starts (``VmSize``)
   plus its share of the budget (the budget divided by the world size) and
   restores the previous soft limit when training ends.  An allocation past
   the cap fails inside the process (``MemoryError``, or the torch CPU
   allocator's "can't allocate memory") instead of exhausting the host; the
   trainer records ``termination_reason: out-of-memory`` and raises
   :class:`MemoryBudgetExceededError`.  Checkpoints already written are
   untouched, so the run resumes exactly from its last good one.

``RLIMIT_AS`` bounds virtual address space, not resident memory (Linux ignores
``RLIMIT_RSS``).  Resident growth can never exceed address-space growth, so
the cap is a hard upper bound on what the run can take, but a conservative
one: reservations made after the cap count against it, such as glibc's
per-thread malloc arenas (each reserves up to 64 MiB of untouched address
space), the stack of any thread started later (8 MiB by default), and shared
libraries loaded later.  A run with many threads therefore fails somewhat
before its resident memory reaches the budget.  The trainer starts its
intra-op thread pool before setting the cap, because OpenMP aborts the whole
process when it cannot start a thread.  The cap covers only the training processes: memory other
processes take after the budget is set is not covered, and the kernel's OOM
killer stays the last line.  A soft limit that is already lower (for example
from ``ulimit -S -v``) is kept.

The default budget is ``0.75 x MemAvailable`` (``psutil.virtual_memory()``)
measured once when training starts (by rank 0, then shared); an operator may
set it explicitly, up to the host's physical memory.
"""

from __future__ import annotations

import resource
from typing import Any

import psutil

from epor.context.probe import (
    _address_space,  # pyright: ignore[reportPrivateUsage]
    certain_components,
)
from epor.models.config import ModelConfig

DEFAULT_BUDGET_FRACTION = 0.75


class MemoryBudgetRefusedError(ValueError):
    """A run's certain allocations exceed its memory budget; nothing was allocated."""

    code = "memory_budget_refused"


class MemoryBudgetExceededError(RuntimeError):
    """A run reached its memory cap; checkpoints already written are intact."""

    code = "memory_budget_exceeded"


def gib(value: int) -> str:
    return f"{value / 1024**3:.2f} GiB"


def default_budget() -> int:
    return int(DEFAULT_BUDGET_FRACTION * psutil.virtual_memory().available)


def check_budget(budget: int) -> None:
    total = psutil.virtual_memory().total
    if not 0 < budget <= total:
        raise MemoryBudgetRefusedError(
            f"the memory budget of {gib(budget)} must be positive and at most this host's "
            f"{gib(total)} of physical memory"
        )


def preflight(
    config: ModelConfig, settings: Any, *, world_size: int, strategy: str, budget: int
) -> dict[str, Any]:
    """The run's certain allocations on this host; refuses when they exceed ``budget``."""

    rows = settings.micro_batch_size or settings.tokens_per_batch // settings.seq_len // world_size
    components = certain_components(
        config,
        settings.seq_len,
        "train",
        rows=rows * world_size,
        precision=settings.precision,
        optimizer=True,
        replicas=world_size if strategy == "ddp" else 1,
    )
    total = sum(components.values())
    if total > budget:
        dominant = max(components, key=components.__getitem__)
        raise MemoryBudgetRefusedError(
            f"the run's certain allocations ({gib(total)}) exceed its memory budget of "
            f"{gib(budget)}; {dominant} dominate at {gib(components[dominant])} "
            f"({rows * world_size} row(s) of {settings.seq_len:,} positions in flight, a "
            f"{config.vocab_size:,}-entry vocabulary, {settings.precision}). Lower the "
            "micro-batch or sequence length, or raise the budget"
        )
    return {"certain_bytes": total, "components": components, "rows_in_flight": rows * world_size}


def cap_address_space(share: int) -> dict[str, Any] | None:
    """Lower this process's soft ``RLIMIT_AS`` to its current size plus ``share``.

    Returns the record :func:`release_cap` needs, or ``None`` where the
    platform offers no address-space limit.
    """

    current = _address_space()
    if current is None or not hasattr(resource, "RLIMIT_AS"):
        return None
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    limit = current + share
    for bound in (soft, hard):
        if bound != resource.RLIM_INFINITY:
            limit = min(limit, bound)
    try:
        resource.setrlimit(resource.RLIMIT_AS, (limit, hard))
    except (ValueError, OSError):
        return None
    return {
        "mechanism": "RLIMIT_AS (soft)",
        "address_space_at_start_bytes": current,
        "share_bytes": share,
        "soft_limit_bytes": limit,
        "previous_soft_limit_bytes": None if soft == resource.RLIM_INFINITY else soft,
    }


def release_cap(cap: dict[str, Any] | None) -> None:
    if cap is not None:
        previous = cap["previous_soft_limit_bytes"]
        _, hard = resource.getrlimit(resource.RLIMIT_AS)
        resource.setrlimit(
            resource.RLIMIT_AS, (resource.RLIM_INFINITY if previous is None else previous, hard)
        )


def is_out_of_memory(exc: BaseException) -> bool:
    if isinstance(exc, MemoryError):
        return True
    text = str(exc).lower()
    return isinstance(exc, RuntimeError) and any(
        phrase in text for phrase in ("can't allocate memory", "not enough memory", "out of memory")
    )


__all__ = [
    "DEFAULT_BUDGET_FRACTION",
    "MemoryBudgetExceededError",
    "MemoryBudgetRefusedError",
    "cap_address_space",
    "check_budget",
    "default_budget",
    "is_out_of_memory",
    "preflight",
    "release_cap",
]
