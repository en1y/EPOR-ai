"""Shared helpers for training-kernel tests, importable by spawned ranks.

``launch_local`` pickles worker functions by module and name, so every
function a spawned rank runs lives here at module level, not in a test file.
"""

from __future__ import annotations


def failing_worker(rank: int, world_size: int) -> None:
    raise ValueError(f"rank {rank} of {world_size} failed deliberately")


def sleeping_worker(rank: int, world_size: int, seconds: float) -> int:
    import time

    time.sleep(seconds)
    return rank
