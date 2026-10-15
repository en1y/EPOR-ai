"""The v0.0.8 context CLI end to end: covenant gates, exit codes, reports, status."""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("torch")

from click import unstyle
from conftest import TokenShardFixture, make_token_shards
from typer.testing import CliRunner

from epor.cli import app

runner = CliRunner()
REPOSITORY = Path(__file__).resolve().parents[1]


def _environment(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    monkeypatch.setenv("EPOR_PROJECT_ROOT", str(root))
    monkeypatch.setenv("EPOR_DATA_ROOT", str(root / "data"))
    monkeypatch.setenv("EPOR_CONTROL_SAFETY_ROOT", str(root / "safety"))
    monkeypatch.setenv("EPOR_CONTROL_ARTIFACT_ROOT", str(root / "artifacts"))
    monkeypatch.setenv("EPOR_CONTROL_DATABASE", str(root / "control.sqlite3"))
    (root / "configs" / "models").mkdir(parents=True, exist_ok=True)
    for name in ("epor-tiny.yaml", "epor-beta.yaml"):
        shutil.copyfile(
            REPOSITORY / "configs" / "models" / name, root / "configs" / "models" / name
        )


def _json(arguments: list[str], *, code: int = 0) -> Any:
    result = runner.invoke(app, arguments)
    assert result.exit_code == code, result.output
    return json.loads(result.stdout)


class Lab:
    def __init__(self, root: Path, shards: TokenShardFixture) -> None:
        self.root = root
        self.shards = shards


@pytest.fixture(scope="module")
def lab(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Lab]:
    root = tmp_path_factory.mktemp("context-cli")
    with pytest.MonkeyPatch.context() as monkeypatch:
        _environment(monkeypatch, root)
        shards = make_token_shards(root, max_shard_tokens=4096)
        _json(
            [
                "train",
                "kernel",
                "configs/models/epor-tiny.yaml",
                "--tokenized-id",
                shards.tokenized_id,
                "--output",
                "runs/base",
                "--max-steps",
                "2",
                "--vocab-from-build",
                "--precision",
                "fp32",
                "--seq-len",
                "64",
                "--tokens-per-batch",
                "128",
                "--validation-split",
                "none",
            ]
        )
        _json(
            [
                "context",
                "extend",
                "runs/base",
                "--target-context",
                "384",
                "--output",
                "runs/ext",
                "--max-steps",
                "2",
                "--precision",
                "fp32",
                "--seq-len",
                "384",
                "--tokens-per-batch",
                "768",
                "--memory-budget-gib",
                "2",
            ]
        )
        yield Lab(root, shards)


@pytest.fixture
def env(lab: Lab, monkeypatch: pytest.MonkeyPatch) -> Lab:
    _environment(monkeypatch, lab.root)
    return lab


def test_context_commands_are_typed_surfaces() -> None:
    output = unstyle(runner.invoke(app, ["context", "--help"]).stdout)
    for name in (
        "extend",
        "retrofit",
        "g6",
        "profile",
        "feasibility",
        "sweep",
        "project",
        "certify",
    ):
        assert name in output


def test_extension_keeps_configured_trained_and_validated_distinct(env: Lab) -> None:
    status = _json(["context", "status", "runs/ext"])
    assert (status["configured"], status["trained"], status["validated"]) == (384, 384, 0)
    assert status["rope_scaling"]["type"] == "yarn"
    assert status["rope_scaling"]["original_max_context"] == 64
    base = _json(["context", "status", "runs/base"])
    assert (base["configured"], base["trained"], base["validated"]) == (128, 64, 0)
    manifest = json.loads((env.root / "runs" / "ext" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["memory_guard"]["budget_bytes"] == 2 * 1024**3


def test_g6_profile_and_a_refused_certification(env: Lab) -> None:
    g6 = _json(
        [
            "context",
            "g6",
            "runs/ext",
            "--length",
            "256",
            "--length",
            "352",
            "--items-per-cell",
            "1",
            "--split",
            "train",
        ]
    )
    assert g6["lengths"] == [256, 352]
    assert set(g6["retrieval"]["passkey"]["256"]) == {"beginning", "middle", "end"}
    assert "validates nothing" in g6["note"]
    profile = _json(["context", "profile", "runs/ext", "--length", "256", "--decode-tokens", "4"])
    [row] = profile["lengths"]
    assert row["status"] == "ok" and row["ttft_seconds"] > 0
    refused = runner.invoke(
        app,
        [
            "context",
            "certify",
            "runs/ext",
            "--length",
            "352",
            "--g6-report",
            g6["report_id"],
            "--baseline-report",
            g6["report_id"],
            "--base-length",
            "256",
            "--profile-report",
            profile["report_id"],
        ],
    )
    assert refused.exit_code == 3, refused.output
    text = unstyle(refused.stdout)
    assert "validated context not raised" in text and "pre-extension checkpoint" in text
    assert _json(["context", "status", "runs/ext"])["validated"] == 0


def test_feasibility_sweep_projection_retrofit_and_listing(env: Lab) -> None:
    feasible = _json(
        [
            "context",
            "feasibility",
            "configs/models/epor-tiny.yaml",
            "--variant",
            "full",
            "--variant",
            "local",
            "--min-length",
            "128",
            "--max-length",
            "256",
            "--window",
            "64",
        ]
    )
    assert feasible["feasible_maximum"]["full"] == {"train": 256, "inference": 256}
    assert feasible["feasible_maximum"]["local"]["inference"] == 256
    sweep = _json(
        [
            "context",
            "sweep",
            "configs/models/epor-tiny.yaml",
            "--axis",
            "window",
            "--value",
            "32",
            "--tokenized-id",
            env.shards.tokenized_id,
            "--output",
            "runs/sweep",
            "--seq-len",
            "64",
            "--tokens-per-batch",
            "128",
            "--max-steps",
            "1",
            "--eval-length",
            "256",
            "--items-per-cell",
            "1",
            "--vocab-from-build",
        ]
    )
    assert sweep["control"]["name"] == "control-full-attention" and len(sweep["points"]) == 1
    projection = _json(
        [
            "context",
            "project",
            "configs/models/epor-beta.yaml",
            "--feasibility-report",
            feasible["report_id"],
        ]
    )
    assert [stage["length"] for stage in projection["stages"]] == [32_768, 131_072, 262_144]
    retrofit = _json(
        [
            "context",
            "retrofit",
            "runs/ext",
            "--output",
            "runs/retro",
            "--max-steps",
            "1",
            "--top-k",
            "8",
            "--precision",
            "fp32",
            "--seq-len",
            "128",
            "--tokens-per-batch",
            "128",
        ]
    )
    assert retrofit["stage"] == "warmup" and "no benefit claimed" in retrofit["scope"]
    # A run stopped before its first checkpoint has trained nothing at its length.
    stopped = env.root / "runs" / "stopped"
    stopped.mkdir()
    (stopped / "manifest.json").write_text((env.root / "runs/ext/manifest.json").read_text())
    summary = _json(["context", "reports"])
    locations = {run["location"]: run for run in summary["runs"]}
    assert locations["runs/ext"]["trained"] == 384 and locations["runs/ext"]["validated"] == 0
    assert locations["runs/stopped"]["configured"] == 384
    assert locations["runs/stopped"]["trained"] is None
    assert "runs/sweep/control-full-attention" in locations
    assert summary["feasibility"] and summary["projection"] and summary["sweep"]
    assert summary["integrity_errors"] == []


def test_certification_is_refused_without_its_declaration(
    env: Lab, monkeypatch: pytest.MonkeyPatch
) -> None:
    import epor.actions

    without = dict(epor.actions.DECLARED_ACTIONS)
    del without["context_certify"]
    monkeypatch.setattr(epor.actions, "DECLARED_ACTIONS", without)
    result = runner.invoke(
        app,
        [
            "context",
            "certify",
            "runs/ext",
            "--length",
            "256",
            "--g6-report",
            "0" * 64,
            "--baseline-report",
            "0" * 64,
            "--base-length",
            "64",
            "--profile-report",
            "0" * 64,
        ],
    )
    assert result.exit_code == 3
    assert "refused by the EPOR safety covenant" in unstyle(result.stdout)
