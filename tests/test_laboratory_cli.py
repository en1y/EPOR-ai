"""The v0.0.6 evaluation and scaling-laboratory CLI: covenant gates, exit codes, reports."""

from __future__ import annotations

import json
import shutil
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("torch")

from click import unstyle
from conftest import TokenShardFixture, make_domain_shards, make_token_shards
from typer.testing import CliRunner

from epor.cli import app

runner = CliRunner()
REPOSITORY = Path(__file__).resolve().parents[1]
TINY = REPOSITORY / "configs" / "models" / "epor-tiny.yaml"
LADDER = REPOSITORY / "configs" / "models" / "ladder"


def _environment(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    monkeypatch.setenv("EPOR_PROJECT_ROOT", str(root))
    monkeypatch.setenv("EPOR_DATA_ROOT", str(root / "data"))
    monkeypatch.setenv("EPOR_CONTROL_SAFETY_ROOT", str(root / "safety"))
    monkeypatch.setenv("EPOR_CONTROL_ARTIFACT_ROOT", str(root / "artifacts"))
    monkeypatch.setenv("EPOR_CONTROL_DATABASE", str(root / "control.sqlite3"))
    (root / "configs" / "models" / "ladder").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(TINY, root / "configs" / "models" / "epor-tiny.yaml")
    for rung in LADDER.glob("*.yaml"):
        shutil.copyfile(rung, root / "configs" / "models" / "ladder" / rung.name)


def _invoke(arguments: list[str]) -> Any:
    return runner.invoke(app, arguments)


def _json(arguments: list[str]) -> Any:
    result = _invoke(arguments)
    assert result.exit_code == 0, result.output
    return json.loads(result.stdout)


class Lab:
    def __init__(self, root: Path, shards: TokenShardFixture) -> None:
        self.root = root
        self.shards = shards

    @property
    def reports(self) -> Path:
        return self.root / "data" / "evaluation-reports"


@pytest.fixture(scope="module")
def lab(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Lab]:
    root = tmp_path_factory.mktemp("lab-cli")
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
                "runs/kernel",
                "--max-steps",
                "2",
                "--vocab-from-build",
                "--precision",
                "fp32",
                "--seq-len",
                "16",
                "--tokens-per-batch",
                "32",
                "--checkpoint-every",
                "2",
                "--validation-split",
                "none",
            ]
        )
        yield Lab(root, shards)


@pytest.fixture
def env(lab: Lab, monkeypatch: pytest.MonkeyPatch) -> Lab:
    _environment(monkeypatch, lab.root)
    return lab


def test_laboratory_commands_are_typed_surfaces() -> None:
    evaluation = unstyle(_invoke(["eval", "--help"]).stdout)
    for name in ("suite", "contamination", "compare", "reports"):
        assert name in evaluation
    scale = unstyle(_invoke(["scale", "--help"]).stdout)
    for name in ("ladder", "profile", "sweep", "mixture", "fit", "project", "reports"):
        assert name in scale


def test_suite_skips_code_by_default_and_stores_a_verified_report(env: Lab) -> None:
    payload = _json(["eval", "suite", "runs/kernel", "--suite", "fixtures"])
    assert payload["scope"] == "harness-fixture"
    assert "Not a benchmark" in payload["claim_boundary"]
    assert payload["code_executed"] is False
    assert {"task_id": "fixture-code", "reason": "code execution not enabled"} in payload[
        "skipped_tasks"
    ]
    assert payload["sandbox"] == []
    assert all(row["scope"] == "harness-fixture" for row in payload["primary_metrics"])
    stored = env.reports / f"{payload['report_id']}.json"
    assert stored.is_file()

    listing = _json(["eval", "reports"])
    assert payload["report_id"] in {item["report_id"] for item in listing["reports"]}
    assert listing["integrity_errors"] == []
    full = _json(["eval", "reports", payload["report_id"]])
    assert full["report_id"] == payload["report_id"]

    safety = _json(["eval", "suite", "runs/kernel", "--suite", "safety"])
    assert safety["suite_id"] != payload["suite_id"]
    comparison = _json(["eval", "compare", payload["report_id"], safety["report_id"]])
    assert "reports come from different suites or suite versions" in comparison["warnings"]
    same = _json(["eval", "compare", payload["report_id"], payload["report_id"], "--seed", "3"])
    assert all(delta["delta"] in (0.0, None) for delta in same["metric_deltas"])


def test_a_tampered_report_is_named_not_listed(env: Lab) -> None:
    payload = _json(["eval", "suite", "runs/kernel", "--suite", "safety"])
    stored = env.reports / f"{payload['report_id']}.json"
    original = stored.read_bytes()
    try:
        stored.chmod(0o600)
        stored.write_bytes(original.replace(b'"harness-fixture"', b'"proxy-scale"', 1))
        listing = _invoke(["eval", "reports"])
        assert listing.exit_code == 2
        document = json.loads(listing.stdout)
        assert payload["report_id"] not in {item["report_id"] for item in document["reports"]}
        assert any(payload["report_id"] in error for error in document["integrity_errors"])
        assert (
            _invoke(["eval", "compare", payload["report_id"], payload["report_id"]]).exit_code == 2
        )
    finally:
        stored.write_bytes(original)


def test_code_execution_is_its_own_covenant_action(
    env: Lab, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The network opt-in means nothing without code execution.
    refused = _invoke(["eval", "suite", "runs/kernel", "--allow-unisolated-network"])
    assert refused.exit_code == 2
    assert "--allow-code-execution" in unstyle(refused.stdout)

    import epor.actions

    before = set(env.reports.glob("*.json"))
    without = dict(epor.actions.DECLARED_ACTIONS)
    del without["eval_code_execution"]
    monkeypatch.setattr(epor.actions, "DECLARED_ACTIONS", without)
    gated = _invoke(["eval", "suite", "runs/kernel", "--allow-code-execution"])
    assert gated.exit_code == 3
    assert "refused by the EPOR safety covenant" in unstyle(gated.stdout)
    assert set(env.reports.glob("*.json")) == before


@pytest.mark.skipif(sys.platform != "linux", reason="the sandbox is Linux-only")
def test_code_execution_records_the_sandbox_it_achieved(env: Lab) -> None:
    from epor.evaluation import CodeSandbox

    isolation = CodeSandbox().probe()
    arguments = ["eval", "suite", "runs/kernel", "--allow-code-execution"]
    if not isolation.network_isolated:
        refused = _invoke(arguments)
        assert refused.exit_code == 3
        assert "allow_unisolated_network" in unstyle(refused.stdout)
        arguments.append("--allow-unisolated-network")
    payload = _json(arguments)
    assert payload["code_executed"] is True
    assert [row["task_id"] for row in payload["sandbox"]] == ["fixture-code"]
    assert payload["sandbox"][0]["network_isolated"] == isolation.network_isolated


def test_contamination_scans_items_without_a_model(env: Lab) -> None:
    payload = _json(["eval", "contamination", "--training-data", env.shards.tokenized_id])
    assert payload["model_kind"] == "data-only"
    assert "lower bound" in payload["note"]
    [row] = payload["contamination"]
    assert row["assessed_items"] >= 1
    assert row["flagged_items"] is not None


def test_operator_paths_and_sources_are_contained_and_registered(env: Lab, tmp_path: Path) -> None:
    outside = tmp_path / "suite.yaml"
    outside.write_text("id: x\n", encoding="utf-8")
    escaped = _invoke(["eval", "suite", "runs/kernel", "--suite", str(outside)])
    assert escaped.exit_code == 2
    assert _invoke(["eval", "suite", str(tmp_path)]).exit_code == 2
    assert _invoke(["eval", "suite", "runs/kernel", "--source", "no-equals"]).exit_code == 2

    benchmark = env.root / "benchmarks" / "mc.jsonl"
    benchmark.parent.mkdir(exist_ok=True)
    benchmark.write_text(
        json.dumps({"question": "1+1?", "choices": ["1", "2"], "answer": 1}) + "\n",
        encoding="utf-8",
    )
    suite = env.root / "benchmarks" / "suite.yaml"
    suite.write_text(
        "id: operator-suite\nversion: 1\nscope: proxy-scale\ntitle: Operator suite\n"
        "tasks:\n  - id: mc\n    type: multiple-choice\n    source:\n"
        "      source_id: unregistered-benchmark\n      format: multiple-choice-jsonl\n",
        encoding="utf-8",
    )
    unregistered = _invoke(
        [
            "eval",
            "suite",
            "runs/kernel",
            "--suite",
            "benchmarks/suite.yaml",
            "--source",
            "unregistered-benchmark=benchmarks/mc.jsonl",
        ]
    )
    assert unregistered.exit_code == 3
    assert _invoke(["eval", "compare", "0" * 64, "1" * 64]).exit_code == 2


def test_cancellation_exits_130(env: Lab, monkeypatch: pytest.MonkeyPatch) -> None:
    import epor.evaluation

    def cancelled(*_args: Any, **_kwargs: Any) -> Any:
        raise epor.evaluation.EvaluationCancelledError("evaluation cancellation requested")

    monkeypatch.setattr(epor.evaluation, "run_suite", cancelled)
    assert _invoke(["eval", "suite", "runs/kernel"]).exit_code == 130


def test_ladder_profile_sweep_fit_and_projection(env: Lab) -> None:
    ladder = _json(["scale", "ladder"])
    status = {row["slug"]: row["execution"] for row in ladder["rungs"]}
    assert status == {
        "epor-ladder-10m": "local",
        "epor-ladder-100m": "projection-only",
        "epor-ladder-300m": "projection-only",
        "epor-ladder-1b": "projection-only",
    }

    profile = _json(
        [
            "scale",
            "profile",
            "configs/models/epor-tiny.yaml",
            "--vocab-from-build",
            env.shards.tokenized_id,
            "--batch-size",
            "1",
            "--seq-len",
            "16",
            "--warmup-steps",
            "0",
            "--timed-steps",
            "1",
        ]
    )
    assert profile["kind"] == "hardware-profile"
    assert profile["scope"] == "host-measurement"
    assert profile["mfu_status"] == "unknown" and profile["mfu"] is None
    assert profile["measured"]["tokens_per_second"] > 0
    assert (
        _invoke(
            [
                "scale",
                "profile",
                "configs/models/ladder/epor-ladder-100m.yaml",
                "--timed-steps",
                "1",
            ]
        ).exit_code
        == 2
    )

    sweep = _json(
        [
            "scale",
            "sweep",
            "--tokenized-id",
            env.shards.tokenized_id,
            "--rung",
            "configs/models/epor-tiny.yaml",
            "--rung",
            "configs/models/ladder/epor-ladder-100m.yaml",
            "--budget",
            "32",
            "--budget",
            "64",
            "--seq-len",
            "16",
            "--tokens-per-batch",
            "32",
            "--precision",
            "fp32",
            "--validation-split",
            "none",
            "--output",
            "runs/sweep",
        ]
    )
    assert sweep["kind"] == "scaling-sweep"
    assert sweep["completed_points"] == 2 and sweep["projection_only_points"] == 2
    statuses = {(p["rung"], p["status"]) for p in sweep["points"]}
    assert ("epor-ladder-100m", "projection-only") in statuses
    # Rerunning into the same directory resumes every completed point.
    again = _json(
        [
            "scale",
            "sweep",
            "--tokenized-id",
            env.shards.tokenized_id,
            "--rung",
            "configs/models/epor-tiny.yaml",
            "--rung",
            "configs/models/ladder/epor-ladder-100m.yaml",
            "--budget",
            "32",
            "--budget",
            "64",
            "--seq-len",
            "16",
            "--tokens-per-batch",
            "32",
            "--precision",
            "fp32",
            "--validation-split",
            "none",
            "--output",
            "runs/sweep",
        ]
    )
    assert again["report_id"] == sweep["report_id"]

    fit = _json(["scale", "fit", sweep["report_id"]])
    assert fit["kind"] == "scaling-fit"
    assert fit["fit_status"]["loss_vs_parameters"] == "insufficient-points"
    assert fit["warning_count"] >= 1
    assert _invoke(["scale", "fit", profile["report_id"]]).exit_code == 2

    inputs = env.root / "projection.yaml"
    shutil.copyfile(REPOSITORY / "configs" / "scaling" / "projection-inputs.example.yaml", inputs)
    projection = _json(
        [
            "scale",
            "project",
            "--inputs",
            "projection.yaml",
            "--target",
            "configs/models/ladder/epor-ladder-1b.yaml",
            "--fit",
            fit["report_id"],
        ]
    )
    assert projection["scope"] == "projection"
    assert projection["approval"]["required"] is True
    assert projection["approval_required"] is True
    assert "hardware.peak_flops_per_device" in projection["unknown_inputs"]
    assert projection["results"]["cost"]["total_with_reserve"]["value"] is None
    assert (
        _invoke(
            [
                "scale",
                "project",
                "--inputs",
                "projection.yaml",
                "--target",
                "configs/models/epor-tiny.yaml",
                "--fit",
                sweep["report_id"],
            ]
        ).exit_code
        == 2
    )

    listing = _json(["scale", "reports"])
    kinds = {item["kind"] for item in listing["reports"]}
    assert {"hardware-profile", "scaling-sweep", "scaling-fit", "compute-projection"} <= kinds
    shown = _json(["scale", "reports", projection["report_id"]])
    assert shown[0]["report_id"] == projection["report_id"]


def test_mixture_ablation_trains_one_model_per_mixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _environment(monkeypatch, tmp_path)
    build = make_domain_shards(tmp_path)
    arguments = [
        "scale",
        "mixture",
        "configs/models/epor-tiny.yaml",
        "--tokenized-id",
        build.tokenized_id,
        "--mixture",
        "balanced:code=1,general=1",
        "--mixture",
        "code-only:code=1,general=0@3",
        "--max-steps",
        "2",
        "--evaluation-max-steps",
        "1",
        "--seq-len",
        "16",
        "--tokens-per-batch",
        "32",
        "--precision",
        "fp32",
        "--output",
        "runs/mixture",
    ]
    payload = _json(arguments)
    assert payload["kind"] == "mixture-ablation"
    assert [item["mixture_id"] for item in payload["mixtures"]] == ["balanced", "code-only"]
    assert _invoke([*arguments[:5], "--mixture", "bad", *arguments[9:]]).exit_code == 2
