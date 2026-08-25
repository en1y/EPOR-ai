"""Configuration for the loopback-only control plane."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from ipaddress import ip_address
from pathlib import Path
from urllib.parse import urlsplit


def _default_project_root() -> Path:
    # src/epor/control/settings.py -> repository root
    return Path(__file__).resolve().parents[3]


def _environment_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class ControlSettings:
    """Validated settings shared by the API and worker.

    The bind host cannot be configured to a non-loopback address.  This is an
    intentional v0.0.1 product boundary rather than a deployment suggestion.
    """

    project_root: Path = field(default_factory=_default_project_root)
    database_path: Path | None = None
    artifact_root: Path | None = None
    research_catalog_path: Path | None = None
    host: str = "127.0.0.1"
    port: int = 8742
    allowed_origin: str = "http://127.0.0.1:5173"
    stale_after_seconds: float = 120.0
    poll_interval_seconds: float = 0.5
    create_schema: bool = True

    def __post_init__(self) -> None:
        project_root = self.project_root.expanduser().resolve()
        database_path = self.database_path or project_root / ".epor" / "control.sqlite3"
        artifact_root = self.artifact_root or project_root / ".epor" / "artifacts"
        research_catalog_path = (
            self.research_catalog_path or project_root / "research" / "catalog.yaml"
        )

        try:
            if not ip_address(self.host).is_loopback:
                raise ValueError("control plane host must be a loopback address")
        except ValueError as exc:
            if str(exc) == "control plane host must be a loopback address":
                raise
            raise ValueError("control plane host must be a numeric loopback address") from exc

        if not (1 <= self.port <= 65535):
            raise ValueError("control plane port must be between 1 and 65535")
        if self.allowed_origin.rstrip("/") != self.allowed_origin:
            raise ValueError("allowed_origin must not have a trailing slash")
        try:
            origin = urlsplit(self.allowed_origin)
            origin_address = ip_address(origin.hostname or "")
            origin_port = origin.port
        except ValueError as exc:
            raise ValueError("allowed_origin must be an exact loopback HTTP origin") from exc
        if (
            origin.scheme != "http"
            or not origin_address.is_loopback
            or origin.username is not None
            or origin.password is not None
            or origin_port is None
            or not 1 <= origin_port <= 65535
            or origin.path
            or origin.query
            or origin.fragment
        ):
            raise ValueError("allowed_origin must be an exact loopback HTTP origin")
        if self.stale_after_seconds <= 0 or self.poll_interval_seconds <= 0:
            raise ValueError("worker timeouts must be positive")

        object.__setattr__(self, "project_root", project_root)
        object.__setattr__(self, "database_path", database_path.expanduser().resolve())
        object.__setattr__(self, "artifact_root", artifact_root.expanduser().resolve())
        object.__setattr__(
            self,
            "research_catalog_path",
            research_catalog_path.expanduser().resolve(),
        )

    @classmethod
    def from_environment(cls) -> ControlSettings:
        """Create settings from the small, documented ``EPOR_CONTROL_*`` set."""

        root = Path(os.getenv("EPOR_PROJECT_ROOT", str(_default_project_root())))
        database = os.getenv("EPOR_CONTROL_DATABASE")
        artifacts = os.getenv("EPOR_CONTROL_ARTIFACT_ROOT")
        catalog = os.getenv("EPOR_RESEARCH_CATALOG")
        return cls(
            project_root=root,
            database_path=Path(database) if database else None,
            artifact_root=Path(artifacts) if artifacts else None,
            research_catalog_path=Path(catalog) if catalog else None,
            host=os.getenv("EPOR_CONTROL_HOST", "127.0.0.1"),
            port=int(os.getenv("EPOR_CONTROL_PORT", "8742")),
            allowed_origin=os.getenv("EPOR_CONTROL_UI_ORIGIN", "http://127.0.0.1:5173"),
            stale_after_seconds=float(os.getenv("EPOR_CONTROL_STALE_AFTER_SECONDS", "120")),
            poll_interval_seconds=float(os.getenv("EPOR_CONTROL_POLL_INTERVAL_SECONDS", "0.5")),
            create_schema=_environment_bool("EPOR_CONTROL_CREATE_SCHEMA", True),
        )

    def prepare_directories(self) -> None:
        """Create only the private runtime directories owned by the console."""

        assert self.database_path is not None
        assert self.artifact_root is not None
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.artifact_root.mkdir(parents=True, exist_ok=True)

    @property
    def uvicorn_kwargs(self) -> dict[str, object]:
        """Safe defaults for the CLI's eventual ``uvicorn.run`` call."""

        return {"host": self.host, "port": self.port, "access_log": False}
