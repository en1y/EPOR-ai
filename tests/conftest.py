from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest

from epor.control.api import create_app
from epor.control.authority import Actor, AuthorityStore
from epor.control.settings import ControlSettings

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True, slots=True)
class Harness:
    """A loopback control app with a bootstrapped owner, isolated per test.

    ``safety_root`` never points at the project's own ``.epor``: a fixture must
    not read, overwrite, or authenticate against the developer's real owner
    credential.
    """

    settings: ControlSettings
    app: Any
    store: AuthorityStore
    owner_token: str
    owner: Actor

    def client(self, **kwargs: Any) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.app),
            base_url="http://control.test",
            **kwargs,
        )

    def as_owner(self) -> httpx.AsyncClient:
        return self.client(headers={"Authorization": f"Bearer {self.owner_token}"})

    def as_bearer(self, token: str) -> httpx.AsyncClient:
        return self.client(headers={"Authorization": f"Bearer {token}"})

    def delegate(self, *scopes: str, days: int = 1, label: str = "operator") -> str:
        issued = self.store.create_delegation(
            label=label,
            scopes=list(scopes),
            expires_at=datetime.now(UTC) + timedelta(days=days),
            by=self.owner,
        )
        return issued.token


@pytest.fixture
def harness(tmp_path: Path) -> Harness:
    settings = ControlSettings(
        project_root=PROJECT_ROOT,
        database_path=tmp_path / "control.sqlite3",
        artifact_root=tmp_path / "artifacts",
        safety_root=tmp_path / "safety",
        data_root=tmp_path / "data",
        research_catalog_path=PROJECT_ROOT / "research" / "catalog.yaml",
        poll_interval_seconds=0.005,
    )
    settings.prepare_directories()
    assert settings.safety_root is not None
    store = AuthorityStore(settings.safety_root)
    issued = store.bootstrap_owner()
    return Harness(
        settings=settings,
        app=create_app(settings),
        store=store,
        owner_token=issued.token,
        owner=issued.principal.actor(),
    )


def run(scenario: Callable[[], Coroutine[Any, Any, None]]) -> None:
    """Drive one async API scenario from a synchronous test."""

    asyncio.run(scenario())
