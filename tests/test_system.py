from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from epor import __version__
from epor.system import _git_revision, probe_system, safe_environment


def test_version_is_v001() -> None:
    assert __version__ == "0.0.1"


def test_probe_is_read_only_and_serializable(tmp_path: Path) -> None:
    result = probe_system(tmp_path)
    payload = result.to_dict()
    assert payload["cpu_logical"] >= 1
    assert payload["memory_bytes"] > 0
    assert isinstance(payload["python"], str)
    assert payload["revision"] is None


@pytest.mark.skipif(shutil.which("git") is None, reason="Git is required for this test")
def test_probe_reads_revision_from_a_linked_worktree(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    linked_worktree = tmp_path / "linked-worktree"
    repository.mkdir()

    def git(*arguments: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(repository), *arguments],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip()

    git("init", "--initial-branch=main")
    (repository / "fixture.txt").write_text("linked worktree fixture\n", encoding="utf-8")
    git("add", "fixture.txt")
    git(
        "-c",
        "user.name=EPOR test",
        "-c",
        "user.email=epor-test@example.invalid",
        "commit",
        "-m",
        "Create fixture commit",
    )
    expected_revision = git("rev-parse", "HEAD")
    git("worktree", "add", "--detach", str(linked_worktree), "HEAD")

    assert (linked_worktree / ".git").is_file()
    assert probe_system(linked_worktree).revision == expected_revision


def test_git_revision_probe_is_bounded_and_side_effect_free(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observed: dict[str, Any] = {}

    def timeout(*args: object, **kwargs: object) -> None:
        observed["args"] = args
        observed.update(kwargs)
        raise subprocess.TimeoutExpired(cmd="git rev-parse", timeout=5)

    monkeypatch.setattr("epor.system.subprocess.run", timeout)

    assert _git_revision(tmp_path) is None
    assert observed["args"] == (
        ["git", "-C", str(tmp_path.resolve()), "rev-parse", "--verify", "HEAD"],
    )
    assert observed["check"] is True
    assert observed["capture_output"] is True
    assert observed["text"] is True
    assert observed["timeout"] == 5


def test_safe_environment_excludes_secrets(monkeypatch: object) -> None:
    # pytest's fixture is intentionally kept untyped to avoid a hard pytest type API.
    monkeypatch.setenv("LANG", "C.UTF-8")  # type: ignore[attr-defined]
    monkeypatch.setenv("EPOR_TOKEN", "secret")  # type: ignore[attr-defined]
    environment = safe_environment()
    assert environment["LANG"] == "C.UTF-8"
    assert "EPOR_TOKEN" not in environment
