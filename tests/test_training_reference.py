from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from epor.training import (  # noqa: E402
    evaluate,
    generate,
    load_checkpoint,
    pretrain,
    resume_pretraining,
)
from epor.training.data import sha256_file  # noqa: E402

TINY_CONFIG = Path("configs/models/epor-tiny.yaml")


def test_checkpoint_resume_is_bit_exact(tmp_path: Path) -> None:
    uninterrupted = pretrain(
        TINY_CONFIG,
        tmp_path / "full",
        max_steps=4,
        batch_size=2,
        sequence_length=8,
        checkpoint_every=2,
    )
    first_half = pretrain(
        TINY_CONFIG,
        tmp_path / "split",
        max_steps=2,
        batch_size=2,
        sequence_length=8,
        checkpoint_every=2,
    )
    resumed = pretrain(
        TINY_CONFIG,
        tmp_path / "split",
        max_steps=4,
        resume_from=first_half.checkpoint_path,
    )

    expected = load_checkpoint(uninterrupted.checkpoint_path)
    actual = load_checkpoint(resumed.checkpoint_path)
    assert first_half.losses + resumed.losses == uninterrupted.losses
    assert expected["step"] == actual["step"] == 4
    assert expected["data_state"]["cursor"] == actual["data_state"]["cursor"]
    assert torch.equal(expected["torch_rng_state"], actual["torch_rng_state"])
    for name, tensor in expected["model_state"].items():
        assert torch.equal(tensor, actual["model_state"][name]), name


def test_manifest_evaluation_and_generation_artifacts(tmp_path: Path) -> None:
    result = pretrain(
        TINY_CONFIG,
        tmp_path / "run",
        max_steps=1,
        sequence_length=8,
    )
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "succeeded"
    assert manifest["config_sha256"]
    assert manifest["data_sha256"]
    assert manifest["tokenizer_sha256"]
    assert manifest["precision"] == "float32"
    assert manifest["dependency_lock_sha256"]
    assert manifest["termination_reason"] == "completed"
    assert manifest["actual_resource_use"]["wall_seconds"] >= 0
    assert manifest["actual_resource_use"]["maximum_rss_bytes"] > 0
    assert manifest["checkpoints"][0]["sha256"]
    assert result.manifest_path.parent.name == "manifests"
    checkpoint = load_checkpoint(result.checkpoint_path)
    assert checkpoint["model_config"]["trained_max_context"] == 8

    evaluation = evaluate(result.checkpoint_path, max_batches=1)
    assert evaluation.loss > 0
    assert evaluation.perplexity > 1
    assert evaluation.report_path.is_file()

    generated = generate(result.checkpoint_path, "EPOR", max_new_tokens=2)
    assert generated.text.startswith("EPOR")
    assert len(generated.new_token_ids) <= 2


def test_new_run_refuses_to_overwrite_an_existing_output(tmp_path: Path) -> None:
    output = tmp_path / "run"
    first = pretrain(TINY_CONFIG, output, max_steps=1, sequence_length=8)
    manifest = json.loads(first.manifest_path.read_text(encoding="utf-8"))
    expected_digest = manifest["checkpoints"][0]["sha256"]

    with pytest.raises(FileExistsError, match="must not already exist"):
        pretrain(TINY_CONFIG, output, max_steps=1, sequence_length=8)

    assert sha256_file(first.checkpoint_path) == expected_digest


def test_branch_resume_preserves_parent_manifest_artifacts(tmp_path: Path) -> None:
    output = tmp_path / "lineage"
    parent = pretrain(
        TINY_CONFIG,
        output,
        max_steps=4,
        sequence_length=8,
        checkpoint_every=2,
    )
    parent_manifest = json.loads(parent.manifest_path.read_text(encoding="utf-8"))
    parent_artifacts = {item["path"]: item["sha256"] for item in parent_manifest["checkpoints"]}
    earlier_checkpoint = output / next(iter(parent_artifacts))

    child = pretrain(
        TINY_CONFIG,
        output,
        max_steps=3,
        sequence_length=8,
        checkpoint_every=2,
        resume_from=earlier_checkpoint,
    )

    assert child.checkpoint_path.parent.name == child.run_id
    for relative_path, digest in parent_artifacts.items():
        assert sha256_file(output / relative_path) == digest


def test_noop_resume_is_rejected_without_replacing_current_lineage(tmp_path: Path) -> None:
    output = tmp_path / "lineage"
    parent = pretrain(TINY_CONFIG, output, max_steps=1, sequence_length=8)
    current_manifest = (output / "manifest.json").read_bytes()

    with pytest.raises(ValueError, match="must exceed the checkpoint step"):
        pretrain(
            TINY_CONFIG,
            output,
            max_steps=1,
            resume_from=parent.checkpoint_path,
        )

    assert (output / "manifest.json").read_bytes() == current_manifest
    resumed = pretrain(
        TINY_CONFIG,
        output,
        max_steps=2,
        resume_from=parent.checkpoint_path,
    )
    assert resumed.steps_completed == 2


def test_resume_helper_preserves_a_custom_corpus(tmp_path: Path) -> None:
    corpus = tmp_path / "custom-corpus.txt"
    corpus.write_text("custom deterministic corpus\n" * 4, encoding="utf-8")
    output = tmp_path / "custom-lineage"
    parent = pretrain(
        TINY_CONFIG,
        output,
        max_steps=1,
        sequence_length=8,
        corpus_path=corpus,
    )

    resumed = resume_pretraining(
        str(TINY_CONFIG),
        str(output),
        str(parent.checkpoint_path),
        max_steps=2,
        corpus_path=corpus,
    )

    assert resumed.steps_completed == 2
