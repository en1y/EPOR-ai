from __future__ import annotations

from pathlib import Path

from epor import __version__
from epor.system import probe_system, safe_environment


def test_version_is_v001() -> None:
    assert __version__ == "0.0.1"


def test_probe_is_read_only_and_serializable(tmp_path: Path) -> None:
    result = probe_system(tmp_path)
    payload = result.to_dict()
    assert payload["cpu_logical"] >= 1
    assert payload["memory_bytes"] > 0
    assert isinstance(payload["python"], str)
    assert payload["revision"] is None


def test_safe_environment_excludes_secrets(monkeypatch: object) -> None:
    # pytest's fixture is intentionally kept untyped to avoid a hard pytest type API.
    monkeypatch.setenv("LANG", "C.UTF-8")  # type: ignore[attr-defined]
    monkeypatch.setenv("EPOR_TOKEN", "secret")  # type: ignore[attr-defined]
    environment = safe_environment()
    assert environment["LANG"] == "C.UTF-8"
    assert "EPOR_TOKEN" not in environment
