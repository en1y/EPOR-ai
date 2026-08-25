"""Local capability inspection used by the CLI and control API."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import shutil
import socket
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import psutil


@dataclass(frozen=True, slots=True)
class SystemCapabilities:
    python: str
    platform: str
    architecture: str
    hostname: str
    cpu_logical: int
    memory_bytes: int
    uv_available: bool
    git_available: bool
    torch_available: bool
    torch_version: str | None
    cuda_available: bool
    rocm_reported: bool
    revision: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_revision(root: Path) -> str | None:
    head = root / ".git" / "HEAD"
    try:
        value = head.read_text(encoding="utf-8").strip()
        if value.startswith("ref: "):
            ref = root / ".git" / value.removeprefix("ref: ")
            return ref.read_text(encoding="utf-8").strip()[:40]
        return value[:40]
    except OSError:
        return None


def probe_system(root: Path | None = None) -> SystemCapabilities:
    """Collect facts without starting accelerators or contacting the network."""

    torch_version = _distribution_version("torch")
    cuda_available = False
    rocm_reported = False
    if torch_version is not None:
        try:
            import torch

            cuda_available = bool(torch.cuda.is_available())
            rocm_reported = bool(getattr(torch.version, "hip", None))
        except (ImportError, OSError, RuntimeError):
            pass

    project_root = (root or Path.cwd()).resolve()
    return SystemCapabilities(
        python=platform.python_version(),
        platform=platform.platform(),
        architecture=platform.machine(),
        hostname=socket.gethostname(),
        cpu_logical=psutil.cpu_count(logical=True) or 1,
        memory_bytes=psutil.virtual_memory().total,
        uv_available=shutil.which("uv") is not None,
        git_available=shutil.which("git") is not None,
        torch_available=torch_version is not None,
        torch_version=torch_version,
        cuda_available=cuda_available,
        rocm_reported=rocm_reported,
        revision=_git_revision(project_root),
    )


def safe_environment() -> dict[str, str]:
    """Return a small allowlist of non-secret environment facts."""

    keys = ("LANG", "LC_ALL", "OMP_NUM_THREADS", "TOKENIZERS_PARALLELISM")
    return {key: os.environ[key] for key in keys if key in os.environ}
