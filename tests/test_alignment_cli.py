"""The v0.0.7 alignment CLI end to end: covenant gates, exit codes, audits, reports."""

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
TINY = REPOSITORY / "configs" / "models" / "epor-tiny.yaml"
SFT_ARGS = ["--precision", "fp32", "--seq-len", "64", "--tokens-per-batch", "128"]


def _environment(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    monkeypatch.setenv("EPOR_PROJECT_ROOT", str(root))
    monkeypatch.setenv("EPOR_DATA_ROOT", str(root / "data"))
    monkeypatch.setenv("EPOR_CONTROL_SAFETY_ROOT", str(root / "safety"))
    monkeypatch.setenv("EPOR_CONTROL_ARTIFACT_ROOT", str(root / "artifacts"))
    monkeypatch.setenv("EPOR_CONTROL_DATABASE", str(root / "control.sqlite3"))
    (root / "configs" / "models").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(TINY, root / "configs" / "models" / "epor-tiny.yaml")


def _invoke(arguments: list[str]) -> Any:
    return runner.invoke(app, arguments)


def _json(arguments: list[str], *, code: int = 0) -> Any:
    result = _invoke(arguments)
    assert result.exit_code == code, result.output
    return json.loads(result.stdout)


class Pipeline:
    def __init__(self, root: Path, shards: TokenShardFixture) -> None:
        self.root = root
        self.shards = shards
        self.values: dict[str, Any] = {}


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Pipeline]:
    root = tmp_path_factory.mktemp("align-cli")
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
                "32",
                "--tokens-per-batch",
                "64",
                "--validation-split",
                "none",
            ]
        )
        state = Pipeline(root, shards)
        seeded = _json(["align", "fixtures"])
        state.values["examples"] = seeded["examples"]
        sft = _json(
            ["align", "dataset", "sft", "--checkpoint", "runs/base", "--include", "fixtures"]
        )
        state.values["sft_dataset"] = sft["dataset_id"]
        _json(
            [
                "train",
                "sft",
                "runs/base",
                "--dataset",
                sft["dataset_id"],
                "--output",
                "runs/sft",
                "--max-steps",
                "2",
                *SFT_ARGS,
            ]
        )
        yield state


@pytest.fixture
def env(pipeline: Pipeline, monkeypatch: pytest.MonkeyPatch) -> Pipeline:
    _environment(monkeypatch, pipeline.root)
    return pipeline


def test_alignment_commands_are_typed_surfaces() -> None:
    train = unstyle(_invoke(["train", "--help"]).stdout)
    for name in ("sft", "preference", "grpo"):
        assert name in train
    align = unstyle(_invoke(["align", "--help"]).stdout)
    for name in ("fixtures", "teacher", "critique", "judge", "dataset", "audit", "monitor", "g10"):
        assert name in align
    grpo = _invoke(["train", "grpo"])
    assert grpo.exit_code == 2 and "v0.5.0" in grpo.stdout


def test_sft_through_the_cli_records_its_dataset_and_parent(env: Pipeline) -> None:
    manifest = json.loads((env.root / "runs" / "sft" / "manifest.json").read_text())
    assert manifest["command"] == "kernel_train"
    assert manifest["chat_dataset_id"] == env.values["sft_dataset"]
    assert manifest["init_checkpoint"]["path"].endswith("step-00000002")
    refused = _invoke(
        ["train", "sft", "runs/base", "--recipe", "configs/models/epor-tiny.yaml", "--dataset", "x"]
    )
    assert refused.exit_code == 2 and "not both" in unstyle(refused.stdout)


def test_teacher_sampling_critique_judging_audit_and_dpo(env: Pipeline) -> None:
    sampled = _json(["align", "teacher", "sample", "runs/base", "--max-new-tokens", "6"])
    assert len(sampled["examples"]) == 8
    assert sampled["teacher"]["kind"] == "epor-checkpoint"
    critique = _json(
        [
            "align",
            "critique",
            "runs/sft",
            "--example",
            env.values["examples"][0],
            "--example",
            env.values["examples"][1],
            "--example",
            env.values["examples"][2],
            "--obligation",
            "no-harmful-output",
            "--max-new-tokens",
            "6",
        ]
    )
    assert len(critique["candidates"]) == 3
    bad = _invoke(
        [
            "align",
            "critique",
            "runs/sft",
            "--example",
            env.values["examples"][0],
            "--obligation",
            "x",
        ]
    )
    assert bad.exit_code == 2 and "not covenant obligation keys" in unstyle(bad.stdout)

    judged = _json(
        [
            "align",
            "judge",
            "runs/sft",
            "--subject",
            "preference",
            "--pair",
            f"{critique['candidates'][0]['example']}:{critique['candidates'][0]['revision']}",
            "--pair",
            f"{critique['candidates'][1]['example']}:{critique['candidates'][1]['revision']}",
            "--pair",
            f"{critique['candidates'][2]['example']}:{critique['candidates'][2]['revision']}",
            "--min-margin",
            "0.0",
        ]
    )
    run = judged["judging_run"]
    sampled_items = _json(["align", "audit", "sample", run, "--per-stratum", "3"])
    assert sampled_items["queued"]
    pending = _json(["align", "audit", "list", "--pending", "--run", run])
    assert {item["state"] for item in pending} == {"pending"}

    # Nothing judged trains before its audit is complete.
    before = _json(
        [
            "align",
            "dataset",
            "preference",
            "--checkpoint",
            "runs/sft",
            "--include",
            "fixtures",
            "--include",
            "ai-judge",
        ]
    )
    assert before["candidates_admitted"] == 0
    assert before["excluded"].get("candidate-not-admitted", 0) == len(judged["preferences"])

    for item in pending:
        verdict = _json(
            [
                "align",
                "audit",
                "verdict",
                item["id"],
                "accept" if item["judge_accepted"] else "reject",
            ]
        )
        assert verdict["verdict"] in {"accept", "reject"}
    calibration = _json(["align", "audit", "calibration", run])
    assert calibration["pending"] == 0 and calibration["disagreement_rate"] == 0.0

    preference = _json(
        [
            "align",
            "dataset",
            "preference",
            "--checkpoint",
            "runs/sft",
            "--include",
            "fixtures",
            "--include",
            "ai-judge",
            "--validation-percent",
            "35",
        ]
    )
    env.values["preference_dataset"] = preference["dataset_id"]
    dpo = _json(
        [
            "train",
            "preference",
            "runs/sft",
            "--dataset",
            preference["dataset_id"],
            "--output",
            "runs/dpo",
            "--max-steps",
            "2",
            "--pairs-per-step",
            "2",
        ]
    )
    assert dpo["steps_completed"] == 2 and "not evidence of alignment" in dpo["scope"]

    monitor = _invoke(
        [
            "align",
            "monitor",
            "runs/dpo",
            "--reference",
            "runs/sft",
            "--dataset",
            preference["dataset_id"],
            "--max-new-tokens",
            "6",
        ]
    )
    assert monitor.exit_code in {0, 1}, monitor.output
    report = json.loads(monitor.stdout)
    assert set(report["monitors"]) == {"kl_per_token", "proxy_minus_audited", "length_ratio"}
    assert report["monitors"]["proxy_minus_audited"]["audited_pairs"] >= 1

    g10 = _invoke(
        ["align", "g10", "--before", "runs/sft", "--after", "runs/dpo", "--max-new-tokens", "6"]
    )
    gate = json.loads(g10.stdout)["gate"]
    assert g10.exit_code == {"blocked": 1, "escalated": 4, "no-regression-observed": 0}[gate]
    summary = _json(["align", "reports"])
    assert summary["g10"] and summary["overoptimization"]
    assert summary["judging_runs"][0]["audited"] >= 1
    assert summary["integrity_errors"] == []


def test_verdicts_are_refused_without_the_audit_declaration(
    env: Pipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    import epor.actions

    log = env.root / "safety" / "alignment-audit.jsonl"
    before = log.read_bytes() if log.exists() else b""
    without = dict(epor.actions.DECLARED_ACTIONS)
    del without["align_audit"]
    monkeypatch.setattr(epor.actions, "DECLARED_ACTIONS", without)
    result = _invoke(["align", "audit", "verdict", "any-item", "accept"])
    assert result.exit_code == 3
    assert "refused by the EPOR safety covenant" in unstyle(result.stdout)
    assert (log.read_bytes() if log.exists() else b"") == before


def test_unknown_audit_items_and_outside_paths_are_refused(env: Pipeline, tmp_path: Path) -> None:
    missing = _invoke(["align", "audit", "verdict", "no-such-item", "accept"])
    assert missing.exit_code == 2 and "not found" in unstyle(missing.stdout)
    outside = tmp_path / "teacher.yaml"
    outside.write_text("id: x\n", encoding="utf-8")
    refused = _invoke(["align", "teacher", "register", str(outside)])
    assert refused.exit_code == 2
    unregistered = env.root / "teacher.yaml"
    unregistered.write_text(
        "schema_version: 1\nid: t\nsource_id: nowhere\nprovider: p\nmodel: m\n"
        "model_version: v\nprompt_template_id: t\nprompt_template_sha256: "
        + "a"
        * 64
        + "\nsampling: {decoding: external}\nrights: {license_id: l, "
        "training_on_outputs_permitted: true, permission_basis: b}\n",
        encoding="utf-8",
    )
    rights = _invoke(["align", "teacher", "register", "teacher.yaml"])
    assert rights.exit_code == 3 and "no data-engine registration" in unstyle(rights.stdout)
