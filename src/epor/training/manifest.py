"""Immutable run-manifest metadata for local reference experiments."""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import platform
import resource
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

try:
    import psutil
except ImportError:  # pragma: no cover - psutil is a core dependency
    psutil = None  # type: ignore[assignment]


def utc_now() -> str:
    """Return an RFC 3339 UTC timestamp."""

    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def code_revision(root: str | Path) -> str | None:
    """Read the current Git revision without mutating repository state."""

    try:
        result = subprocess.run(
            ["git", "-C", str(Path(root).resolve()), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    revision = result.stdout.strip()
    return revision or None


def dependency_versions() -> dict[str, str]:
    """Capture installed package versions in stable name order."""

    packages: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata["Name"]
        if name:
            packages[name.lower()] = distribution.version
    return dict(sorted(packages.items()))


def hardware_snapshot() -> dict[str, Any]:
    """Describe only local, non-secret hardware/runtime capabilities."""

    memory_bytes = psutil.virtual_memory().total if psutil is not None else None
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "cpu_logical": os.cpu_count(),
        "memory_bytes": memory_bytes,
        "torch_num_threads": torch.get_num_threads(),
        "device": "cpu",
    }


def resource_usage(started_wall: float, started_cpu: float) -> dict[str, int | float]:
    """Capture bounded process-level resource use for a completed tiny run."""

    maximum_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KiB while macOS reports bytes. Linux is EPOR's primary host,
    # but preserving the conversion makes local manifests portable.
    maximum_rss_bytes = maximum_rss if sys.platform == "darwin" else maximum_rss * 1024
    return {
        "wall_seconds": max(0.0, time.perf_counter() - started_wall),
        "process_cpu_seconds": max(0.0, time.process_time() - started_cpu),
        "maximum_rss_bytes": int(maximum_rss_bytes),
    }


def _file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def new_manifest(
    *,
    run_id: str,
    config_hash: str,
    data_hash: str,
    tokenizer_hash: str,
    model_seed: int,
    data_seed: int,
    started_at: str,
    parent_run_id: str | None,
    parent_checkpoint: str | None,
    project_root: str | Path,
) -> dict[str, Any]:
    """Build the complete initial manifest for a reference pretraining run."""

    root = Path(project_root).resolve()
    return {
        "schema_version": "1",
        "run_id": run_id,
        "command": "tiny_train",
        "status": "running",
        "code_revision": code_revision(root),
        "dependency_lock_sha256": _file_sha256(root / "uv.lock"),
        "config_sha256": config_hash,
        "data_sha256": data_hash,
        "tokenizer_sha256": tokenizer_hash,
        "seeds": {"model": model_seed, "data": data_seed},
        "dependencies": dependency_versions(),
        "hardware": hardware_snapshot(),
        "python": sys.version,
        "precision": "float32",
        "started_at": started_at,
        "finished_at": None,
        "actual_resource_use": None,
        "termination_reason": None,
        "parent_run_id": parent_run_id,
        "parent_checkpoint": parent_checkpoint,
        "checkpoints": [],
        "metrics": {},
    }
