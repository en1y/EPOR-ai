"""Deterministic fault injection for crash-recovery tests.

A :class:`FaultPlan` names one optimizer step and one point in it:

``step``
    after the optimizer update and metrics of that step, before any
    checkpoint of that step is written (a crash between checkpoints);
``checkpoint_write``
    while writing that step's checkpoint, after every rank's state is on disk
    and before the checkpoint is promoted or ``latest.json`` moves (a crash
    mid-checkpoint).

``mode='raise'`` raises :class:`InjectedFault` (the run's manifest records a
failure); ``mode='exit'`` calls ``os._exit`` to model a killed process that
runs no cleanup at all.  ``ranks`` limits the fault to some ranks.

Faults are only ever injected when a caller passes a plan to the trainer, or
explicitly builds one with :func:`fault_plan_from_env`; the trainer never reads
the environment on its own.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, Literal

FaultPoint = Literal["step", "checkpoint_write"]
FaultMode = Literal["raise", "exit"]
FAULT_EXIT_CODE: Final = 86


class InjectedFault(RuntimeError):
    """The deliberate failure raised by a ``mode='raise'`` fault plan."""


@dataclass(frozen=True, slots=True)
class FaultPlan:
    step: int
    point: FaultPoint = "step"
    mode: FaultMode = "raise"
    ranks: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if self.step < 1:
            raise ValueError("fault step must be a positive optimizer step")
        if self.point not in ("step", "checkpoint_write"):
            raise ValueError(f"unknown fault point: {self.point!r}")
        if self.mode not in ("raise", "exit"):
            raise ValueError(f"unknown fault mode: {self.mode!r}")

    def maybe_fire(self, point: FaultPoint, step: int, rank: int) -> None:
        if point != self.point or step != self.step:
            return
        if self.ranks is not None and rank not in self.ranks:
            return
        if self.mode == "exit":
            os._exit(FAULT_EXIT_CODE)
        raise InjectedFault(f"injected fault at step {step} ({point}) on rank {rank}")


def fault_plan_from_env(environ: Mapping[str, str] | None = None) -> FaultPlan | None:
    """Build a plan from ``EPOR_FAULT_AT_STEP`` / ``EPOR_FAULT_POINT`` / ``EPOR_FAULT_MODE``."""

    values = os.environ if environ is None else environ
    raw_step = values.get("EPOR_FAULT_AT_STEP")
    if not raw_step:
        return None
    point = values.get("EPOR_FAULT_POINT", "step")
    mode = values.get("EPOR_FAULT_MODE", "raise")
    if point not in ("step", "checkpoint_write") or mode not in ("raise", "exit"):
        raise ValueError("EPOR_FAULT_POINT or EPOR_FAULT_MODE is not a known value")
    return FaultPlan(step=int(raw_step), point=point, mode=mode)
