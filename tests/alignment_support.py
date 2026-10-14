"""Shared alignment-test fixtures: a tokenized build, a tiny base checkpoint, a store."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from conftest import TokenShardFixture, make_token_shards
from kernel_support import kernel_config, kernel_settings

from epor.alignment.records import AlignmentStore
from epor.training import OptimizerSettings, pretrain_packed
from epor.training.kernel import CheckpointRef, PackedTrainingSettings
from epor.training.run_checkpoint import latest_checkpoint

BASE_STEPS = 4


@dataclass(frozen=True)
class AlignmentFixture:
    shards: TokenShardFixture
    config: Any
    base_run: Path
    base_checkpoint: Path
    base_ref: CheckpointRef
    store: AlignmentStore
    tokenizer: Any


def checkpoint_ref(directory: Path) -> CheckpointRef:
    from epor.alignment.model import checkpoint_identity

    return CheckpointRef(path=str(directory), checkpoint_sha256=checkpoint_identity(directory))


def make_alignment_fixture(root: Path) -> AlignmentFixture:
    from epor.tokenizer import load_tokenizer

    shards = make_token_shards(root)
    config = kernel_config(shards.vocab_size)
    run = root / "base"
    pretrain_packed(
        config,
        shards.data_root,
        run,
        kernel_settings(shards.tokenized_id, seq_len=32, tokens_per_batch=64),
        max_steps=BASE_STEPS,
    )
    checkpoint = latest_checkpoint(run)
    assert checkpoint is not None
    return AlignmentFixture(
        shards=shards,
        config=config,
        base_run=run,
        base_checkpoint=checkpoint,
        base_ref=checkpoint_ref(checkpoint),
        store=AlignmentStore(shards.data_root),
        tokenizer=load_tokenizer(shards.data_root, shards.tokenizer_id),
    )


def sft_settings(
    fixture: AlignmentFixture, dataset_id: str, **overrides: Any
) -> PackedTrainingSettings:
    values: dict[str, Any] = {
        "tokenized_id": fixture.shards.tokenized_id,
        "chat_dataset_id": dataset_id,
        "init_checkpoint": fixture.base_ref,
        "seq_len": 64,
        "tokens_per_batch": 128,
        "micro_batch_size": 1,
        "precision": "fp32",
        "checkpoint_every": 2,
        "validation_split": "validation",
        "optimizer": OptimizerSettings(learning_rate=3e-3, warmup_steps=1, total_steps=6),
        **overrides,
    }
    return PackedTrainingSettings.model_validate(values)
