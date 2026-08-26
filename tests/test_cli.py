from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from epor.cli import app

runner = CliRunner()


def test_version_command() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == "0.0.1"


def test_doctor_json_is_machine_readable() -> None:
    result = runner.invoke(app, ["doctor", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["cpu_logical"] >= 1
    assert payload["memory_bytes"] > 0
    assert "environment" not in payload


def test_future_command_fails_closed() -> None:
    result = runner.invoke(app, ["data", "ingest"])
    assert result.exit_code == 2
    assert "v0.0.3" in result.stdout


def test_resume_exposes_original_corpus_option() -> None:
    result = runner.invoke(app, ["train", "resume", "--help"])
    assert result.exit_code == 0
    assert "--corpus" in result.stdout


def test_reference_cli_rejects_paths_outside_project_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    monkeypatch.setenv("EPOR_PROJECT_ROOT", str(project_root))
    outside_config = tmp_path / "outside.yaml"
    outside_config.write_text("not: loaded\n", encoding="utf-8")
    result = runner.invoke(
        app,
        ["train", "pretrain", str(outside_config), "--output", "runs/blocked"],
    )

    assert result.exit_code == 2
    assert "reference command refused" in result.stdout
    assert "path must remain beneath" in result.stdout
    assert not (project_root / "runs/blocked").exists()


def test_reference_cli_rejects_output_outside_project_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = tmp_path / "project"
    config = project_root / "configs/models/epor-tiny.yaml"
    corpus = project_root / "fixtures/tiny_corpus.txt"
    config.parent.mkdir(parents=True)
    corpus.parent.mkdir(parents=True)
    config.write_bytes(Path("configs/models/epor-tiny.yaml").read_bytes())
    corpus.write_bytes(Path("fixtures/tiny_corpus.txt").read_bytes())
    monkeypatch.setenv("EPOR_PROJECT_ROOT", str(project_root))
    result = runner.invoke(
        app,
        [
            "train",
            "pretrain",
            "configs/models/epor-tiny.yaml",
            "--output",
            str(tmp_path / "outside-run"),
        ],
    )

    assert result.exit_code == 2
    assert "reference command refused" in result.stdout
    assert "output path" in result.stdout
    assert not (tmp_path / "outside-run").exists()


def test_reference_cli_exposes_the_shared_step_limit() -> None:
    result = runner.invoke(
        app,
        [
            "train",
            "pretrain",
            "configs/models/epor-tiny.yaml",
            "--output",
            "runs/blocked",
            "--max-steps",
            "1001",
        ],
    )

    assert result.exit_code == 2
    assert "1000" in f"{result.stdout}{result.stderr}"


def test_cli_execution_is_gated_by_the_safety_covenant() -> None:
    from epor import actions

    # Withdrawing a declaration must stop the command, not fall through to work.
    withdrawn = actions.DECLARED_ACTIONS.pop("generate")
    try:
        result = runner.invoke(app, ["generate", "fixtures/tiny_corpus.txt", "hello"])
    finally:
        actions.DECLARED_ACTIONS["generate"] = withdrawn

    assert result.exit_code == 3
    assert "safety covenant" in result.stdout
    assert "Protect people" in result.stdout


@pytest.mark.parametrize(
    ("action_id", "arguments"),
    [
        ("control_api_start", ["api"]),
        ("job_worker_start", ["worker", "--once"]),
        ("local_ui_start", ["ui"]),
    ],
)
def test_local_service_startup_is_gated_by_the_safety_covenant(
    action_id: str,
    arguments: list[str],
) -> None:
    from epor import actions

    withdrawn = actions.DECLARED_ACTIONS.pop(action_id)
    try:
        result = runner.invoke(app, arguments)
    finally:
        actions.DECLARED_ACTIONS[action_id] = withdrawn

    assert result.exit_code == 3
    assert "safety covenant" in result.stdout
    assert "Protect people" in result.stdout
