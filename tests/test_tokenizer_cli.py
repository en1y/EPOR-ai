"""The tokenizer CLI: covenant-gated, idempotent, and honest about failed gates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from click import unstyle
from conftest import TOKENIZER_FIXTURE_VOCAB, make_tokenizer_build
from typer.testing import CliRunner

from epor.cli import app
from epor.control.authority import AuthorityStore

runner = CliRunner()


@pytest.fixture
def cli_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("EPOR_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setenv("EPOR_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("EPOR_CONTROL_SAFETY_ROOT", str(tmp_path / "safety"))
    monkeypatch.setenv("EPOR_CONTROL_ARTIFACT_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("EPOR_CONTROL_DATABASE", str(tmp_path / "control.sqlite3"))
    return tmp_path


def _json(arguments: list[str]) -> dict[str, Any]:
    result = runner.invoke(app, arguments)
    assert result.exit_code == 0, result.output
    return json.loads(result.stdout)


def test_tokenizer_commands_are_executable_typed_surfaces() -> None:
    result = runner.invoke(app, ["tokenizer", "--help"])
    assert result.exit_code == 0
    output = unstyle(result.stdout)
    assert all(command in output for command in ("train", "shard", "evaluate"))


def test_train_shard_evaluate_is_idempotent_and_audited(cli_root: Path) -> None:
    build = make_tokenizer_build(cli_root)
    train_args = [
        "tokenizer",
        "train",
        build.dataset_id,
        build.build_id,
        "--vocab-size",
        str(TOKENIZER_FIXTURE_VOCAB),
    ]
    trained = _json(train_args)
    assert trained["declared_vocab_size"] == TOKENIZER_FIXTURE_VOCAB
    assert trained["learned_vocab_size"] <= TOKENIZER_FIXTURE_VOCAB
    assert _json(train_args)["tokenizer_id"] == trained["tokenizer_id"]

    shard_args = [
        "tokenizer",
        "shard",
        build.dataset_id,
        build.build_id,
        trained["tokenizer_id"],
        "--max-shard-tokens",
        "4096",
    ]
    first = _json(shard_args)
    again = _json(shard_args)
    assert first["created"] is True and again["created"] is False
    assert again["tokenized_id"] == first["tokenized_id"]
    assert first["total_documents"] == len(build.documents)

    evaluate_args = ["tokenizer", "evaluate", trained["tokenizer_id"], build.dataset_id]
    report = _json([*evaluate_args, build.build_id])
    assert report["split"] == "validation"
    assert report["in_sample"] is False
    assert report["round_trip_failures"] == 0
    assert "not target-scale" in report["scope"]
    assert _json([*evaluate_args, build.build_id])["report_id"] == report["report_id"]

    in_sample = runner.invoke(app, [*evaluate_args, build.build_id, "--split", "train"])
    assert in_sample.exit_code == 0
    assert json.loads(in_sample.stdout)["in_sample"] is True
    assert "in-sample" in in_sample.stderr

    actions = [
        record["action_id"]
        for record in AuthorityStore(cli_root / "safety").records()
        if record.get("kind") == "safety.decision"
    ]
    assert {"tokenizer_train", "tokenizer_shard", "tokenizer_evaluate"} <= set(actions)


def test_round_trip_failures_fail_the_evaluate_gate(
    cli_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from epor.tokenizer import codec

    build = make_tokenizer_build(cli_root)
    trained = _json(
        [
            "tokenizer",
            "train",
            build.dataset_id,
            build.build_id,
            "--vocab-size",
            str(TOKENIZER_FIXTURE_VOCAB),
        ]
    )
    monkeypatch.setattr(codec.EporTokenizer, "decode_bytes", lambda *_args, **_kwargs: b"corrupted")
    result = runner.invoke(
        app, ["tokenizer", "evaluate", trained["tokenizer_id"], build.dataset_id, build.build_id]
    )
    assert result.exit_code == 1
    assert json.loads(result.stdout)["round_trip_failures"] > 0
    assert "round-trip gate failed" in result.stderr


@pytest.mark.parametrize(
    "arguments",
    [
        ["tokenizer", "train", "missing-v1", "0" * 64, "--vocab-size", "1024"],
        ["tokenizer", "shard", "missing-v1", "0" * 64, "0" * 64],
        ["tokenizer", "evaluate", "not-a-hash", "missing-v1", "0" * 64],
    ],
)
def test_missing_inputs_exit_cleanly(cli_root: Path, arguments: list[str]) -> None:
    result = runner.invoke(app, arguments)
    assert result.exit_code == 2
    assert "tokenizer command refused" in result.stdout
    assert result.exception is None or isinstance(result.exception, SystemExit)


@pytest.mark.parametrize(
    ("action_id", "arguments"),
    [
        ("tokenizer_train", ["train", "a", "b"]),
        ("tokenizer_shard", ["shard", "a", "b", "c"]),
        ("tokenizer_evaluate", ["evaluate", "a", "b", "c"]),
    ],
)
def test_tokenizer_commands_are_gated_by_the_safety_covenant(
    cli_root: Path, action_id: str, arguments: list[str]
) -> None:
    from epor import actions

    withdrawn = actions.DECLARED_ACTIONS.pop(action_id)
    try:
        result = runner.invoke(app, ["tokenizer", *arguments])
    finally:
        actions.DECLARED_ACTIONS[action_id] = withdrawn

    assert result.exit_code == 3
    assert "safety covenant" in result.stdout
    assert not list((cli_root / "data" / "tokenizers").glob("*"))
