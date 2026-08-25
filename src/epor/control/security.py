"""Path containment and conservative secret redaction helpers."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


class PathOutsideRootError(ValueError):
    """Raised when a user-controlled path escapes its configured root."""


_SENSITIVE_KEY = re.compile(
    r"(?:authorization|cookie|credential|password|passwd|secret|token|api[_-]?key|private[_-]?key)",
    re.IGNORECASE,
)
_INLINE_SECRET_PATTERNS = (
    re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]{8,}"),
    re.compile(
        r"(?i)((?:api[_-]?key|password|secret|token)\s*[=:]\s*)"
        r"[^\s,;]{4,}"
    ),
    re.compile(r"\b(?:sk|rk|pk)-[A-Za-z0-9_-]{12,}\b"),
)


def contained_path(
    root: Path,
    candidate: str | Path,
    *,
    must_exist: bool = False,
) -> Path:
    """Resolve ``candidate`` beneath ``root``, rejecting traversal and symlinks.

    Relative paths are interpreted relative to the root.  ``Path.resolve`` is
    used even for not-yet-created paths so an existing symlink in any parent
    cannot be used to escape containment.
    """

    resolved_root = root.expanduser().resolve()
    supplied = Path(candidate).expanduser()
    resolved = (resolved_root / supplied if not supplied.is_absolute() else supplied).resolve(
        strict=must_exist
    )
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise PathOutsideRootError(f"path must remain beneath {resolved_root}") from exc
    if must_exist and not resolved.exists():
        raise FileNotFoundError(resolved)
    return resolved


def contained_in_any(
    roots: Sequence[Path], candidate: str | Path, *, must_exist: bool = False
) -> Path:
    """Resolve a path beneath one of a small, explicit set of roots."""

    failures: list[Exception] = []
    supplied = Path(candidate).expanduser()
    for root in roots:
        try:
            return contained_path(root, supplied, must_exist=must_exist)
        except (PathOutsideRootError, FileNotFoundError) as exc:
            failures.append(exc)
        # A relative path is interpreted against each permitted root.
    if must_exist and any(isinstance(exc, FileNotFoundError) for exc in failures):
        raise FileNotFoundError(candidate)
    raise PathOutsideRootError("path must remain beneath an allowed runtime root")


def redact_text(value: str) -> str:
    """Remove common inline credential forms without echoing their values."""

    redacted = value
    for pattern in _INLINE_SECRET_PATTERNS:
        if pattern.groups:
            redacted = pattern.sub(r"\1[REDACTED]", redacted)
        else:
            redacted = pattern.sub("[REDACTED]", redacted)
    return redacted


def redact(value: Any) -> Any:
    """Recursively redact secret-looking keys and inline string credentials."""

    if isinstance(value, Mapping):
        return {
            str(key): "[REDACTED]" if _SENSITIVE_KEY.search(str(key)) else redact(item)
            for key, item in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [redact(item) for item in value]
    if isinstance(value, str):
        return redact_text(value)
    return value
