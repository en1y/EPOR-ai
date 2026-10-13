"""Exact single-process resume, compatibility refusal, cancellation, and run manifests (G3)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from conftest import TokenShardFixture, make_token_shards  # noqa: E402
from kernel_support import kernel_config, kernel_settings, same_weights  # noqa: E402

from epor.training import (  # noqa: E402
    CheckpointError,
    OptimizerSettings,
    TrainingCancelled,
    latest_checkpoint,
    pretrain_packed,
    read_kernel_checkpoint,
)
from epor.training.kernel import final_model_state  # noqa: E402


@pytest.fixture(scope="module")
def shards(tmp_path_factory: pytest.TempPathFactory) -> TokenShardFixture:
    return make_token_shards(tmp_path_factory.mktemp("kernel-resume"))


@pytest.mark.parametrize(
    ("precision", "features"),
    [("fp32", {}), ("bf16", {"qk_norm": True, "mtp_depth": 1})],
)
def test_resume_reproduces_uninterrupted_losses_and_weights(
    shards: TokenShardFixture, tmp_path: Path, precision: str, features: dict[str, object]
) -> None:
    config = kernel_config(shards.vocab_size, **features)
    settings = kernel_settings(
        shards.tokenized_id, precision=precision, tokens_per_batch=64, micro_batch_size=2
    )
    full = pretrain_packed(config, shards.data_root, tmp_path / "full", settings, max_steps=6)
    first = pretrain_packed(config, shards.data_root, tmp_path / "split", settings, max_steps=4)
    # A fresh model and optimizer in the same lineage, with activation
    # checkpointing switched on: a runtime-only setting may change on resume.
    resumed = pretrain_packed(
        config,
        shards.data_root,
        tmp_path / "split",
        settings.model_copy(update={"activation_checkpointing": True}),
        max_steps=6,
        resume_from=tmp_path / "split",
    )
    assert resumed.start_step == 4
    assert first.losses + resumed.losses == full.losses
    assert same_weights(final_model_state(tmp_path / "full"), final_model_state(tmp_path / "split"))
    full_state = read_kernel_checkpoint(full.checkpoint_path)
    split_state = read_kernel_checkpoint(resumed.checkpoint_path)
    assert full_state["data_state"] == split_state["data_state"]
    assert full_state["tokens_seen"] == split_state["tokens_seen"] == 6 * 64
    assert resumed.lineage_id == first.lineage_id


def test_resume_refuses_trajectory_and_model_changes(
    shards: TokenShardFixture, tmp_path: Path
) -> None:
    config = kernel_config(shards.vocab_size)
    settings = kernel_settings(shards.tokenized_id)
    run = tmp_path / "run"
    pretrain_packed(config, shards.data_root, run, settings, max_steps=2)

    changed_optimizer = settings.model_copy(
        update={"optimizer": OptimizerSettings(learning_rate=1e-3, warmup_steps=2, total_steps=8)}
    )
    with pytest.raises(CheckpointError, match="optimizer"):
        pretrain_packed(
            config, shards.data_root, run, changed_optimizer, max_steps=4, resume_from=run
        )
    with pytest.raises(CheckpointError, match="data_seed"):
        pretrain_packed(
            config,
            shards.data_root,
            run,
            settings.model_copy(update={"data_seed": 99}),
            max_steps=4,
            resume_from=run,
        )
    with pytest.raises(CheckpointError, match="precision"):
        pretrain_packed(
            config,
            shards.data_root,
            run,
            settings.model_copy(update={"precision": "bf16"}),
            max_steps=4,
            resume_from=run,
        )
    with pytest.raises(CheckpointError, match="model config"):
        pretrain_packed(
            kernel_config(shards.vocab_size, qk_norm=True),
            shards.data_root,
            run,
            settings,
            max_steps=4,
            resume_from=run,
        )
    with pytest.raises(ValueError, match="exceed the checkpoint step"):
        pretrain_packed(config, shards.data_root, run, settings, max_steps=2, resume_from=run)
    with pytest.raises(FileExistsError):
        pretrain_packed(config, shards.data_root, run, settings, max_steps=2)
    # Runtime-only settings may change.
    resumed = pretrain_packed(
        config,
        shards.data_root,
        run,
        settings.model_copy(update={"checkpoint_every": 1, "keep_checkpoints": 2}),
        max_steps=4,
        resume_from=run,
    )
    assert resumed.steps_completed == 4
    kept = sorted(path.name for path in (run / "checkpoints").glob("step-*"))
    assert kept == ["step-00000003", "step-00000004"]


def test_checkpoint_integrity_is_verified(shards: TokenShardFixture, tmp_path: Path) -> None:
    config = kernel_config(shards.vocab_size)
    settings = kernel_settings(shards.tokenized_id)
    run = tmp_path / "run"
    result = pretrain_packed(config, shards.data_root, run, settings, max_steps=2)
    assert latest_checkpoint(run) == result.checkpoint_path
    state_file = next((result.checkpoint_path / "state").glob("*.distcp"))
    state_file.write_bytes(state_file.read_bytes() + b"tampered")
    with pytest.raises(CheckpointError, match="recorded hashes"):
        read_kernel_checkpoint(run)
    with pytest.raises(CheckpointError):
        pretrain_packed(config, shards.data_root, run, settings, max_steps=4, resume_from=run)


def test_cancellation_checkpoints_and_resumes_exactly(
    shards: TokenShardFixture, tmp_path: Path
) -> None:
    config = kernel_config(shards.vocab_size)
    settings = kernel_settings(shards.tokenized_id, checkpoint_every=4)
    full = pretrain_packed(config, shards.data_root, tmp_path / "full", settings, max_steps=5)

    completed: list[int] = []
    run = tmp_path / "cancelled"
    with pytest.raises(TrainingCancelled):
        pretrain_packed(
            config,
            shards.data_root,
            run,
            settings,
            max_steps=5,
            should_cancel=lambda: len(completed) >= 3,
            on_step=lambda step, total, loss: completed.append(step),
        )
    assert completed == [1, 2, 3]
    # Step 3 is not a scheduled checkpoint, so cancellation saved it.
    assert read_kernel_checkpoint(run)["step"] == 3
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "cancelled"
    assert manifest["termination_reason"] == "cancelled"
    resumed = pretrain_packed(config, shards.data_root, run, settings, max_steps=5, resume_from=run)
    assert resumed.losses == full.losses[3:]
    assert same_weights(final_model_state(tmp_path / "full"), final_model_state(run))

    # A request before the first step stops immediately without a checkpoint.
    with pytest.raises(TrainingCancelled):
        pretrain_packed(
            config,
            shards.data_root,
            tmp_path / "early",
            settings,
            max_steps=5,
            should_cancel=lambda: True,
        )
    assert latest_checkpoint(tmp_path / "early") is None
