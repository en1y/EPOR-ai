from __future__ import annotations

import json

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
