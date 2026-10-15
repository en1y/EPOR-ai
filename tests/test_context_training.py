"""Progressive context extension and the sparse retrofit on the training kernel."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from conftest import TokenShardFixture, make_token_shards  # noqa: E402
from kernel_support import kernel_config, kernel_settings, same_weights  # noqa: E402

from epor.context.extension import extension_config, retrofit_config  # noqa: E402
from epor.models.config import ModelConfig, SparseAttention  # noqa: E402
from epor.training import pretrain_packed  # noqa: E402
from epor.training.kernel import CheckpointRef, final_model_state, trained_context  # noqa: E402
from epor.training.run_checkpoint import (  # noqa: E402
    CheckpointError,
    latest_checkpoint,
    read_kernel_checkpoint,
)

BASE_LEN = 32
LONG_LEN = 128


@pytest.fixture(scope="module")
def shards(tmp_path_factory: pytest.TempPathFactory) -> TokenShardFixture:
    return make_token_shards(tmp_path_factory.mktemp("context-training"), max_shard_tokens=4096)


@pytest.fixture(scope="module")
def base(shards: TokenShardFixture, tmp_path_factory: pytest.TempPathFactory) -> Path:
    run = tmp_path_factory.mktemp("base") / "run"
    pretrain_packed(
        kernel_config(shards.vocab_size),
        shards.data_root,
        run,
        kernel_settings(shards.tokenized_id, seq_len=BASE_LEN, tokens_per_batch=64),
        max_steps=2,
    )
    return run


def _ref(run: Path) -> CheckpointRef:
    from epor.alignment.model import checkpoint_identity

    directory = latest_checkpoint(run)
    assert directory is not None
    return CheckpointRef(path=str(directory), checkpoint_sha256=checkpoint_identity(directory))


def _extended(shards: TokenShardFixture) -> ModelConfig:
    return extension_config(
        kernel_config(shards.vocab_size), 256, trained_context=BASE_LEN, method="yarn"
    )


def test_extension_config_scales_from_the_trained_not_the_configured_context(
    shards: TokenShardFixture,
) -> None:
    config = _extended(shards)
    assert config.configured_max_context == 256
    assert config.rope_scaling is not None
    assert (config.rope_scaling.factor, config.rope_scaling.original_max_context) == (8, 32)
    with pytest.raises(ValueError, match="must exceed"):
        extension_config(config, 16, trained_context=BASE_LEN)


def test_extension_continues_from_the_parent_records_trained_context_and_resumes_exactly(
    shards: TokenShardFixture, base: Path, tmp_path: Path
) -> None:
    settings = kernel_settings(
        shards.tokenized_id,
        seq_len=LONG_LEN,
        tokens_per_batch=LONG_LEN * 2,
        init_checkpoint=_ref(base),
    )
    config = _extended(shards)
    full = pretrain_packed(config, shards.data_root, tmp_path / "full", settings, max_steps=4)
    trainer = read_kernel_checkpoint(tmp_path / "full")
    assert trained_context(trainer) == LONG_LEN
    manifest = json.loads((tmp_path / "full" / "manifest.json").read_text())
    assert manifest["context"]["configured"] == 256 and manifest["context"]["trained"] == LONG_LEN
    assert manifest["context"]["extension_of"] == kernel_config(shards.vocab_size).sha256

    pretrain_packed(config, shards.data_root, tmp_path / "split", settings, max_steps=2)
    resumed = pretrain_packed(
        config,
        shards.data_root,
        tmp_path / "split",
        settings,
        max_steps=4,
        resume_from=tmp_path / "split",
    )
    assert resumed.losses == full.losses[2:]
    assert same_weights(final_model_state(tmp_path / "split"), final_model_state(tmp_path / "full"))

    # The parent's weights are the starting point, not a fresh initialization.
    frozen = settings.model_copy(
        update={"optimizer": settings.optimizer.model_copy(update={"learning_rate": 1e-9})}
    )
    pretrain_packed(config, shards.data_root, tmp_path / "near", frozen, max_steps=1)
    near, parent = final_model_state(tmp_path / "near"), final_model_state(base)
    assert max(float((near[k] - parent[k]).abs().max()) for k in parent) < 1e-5


def test_an_init_checkpoint_from_a_different_model_is_refused(
    shards: TokenShardFixture, base: Path, tmp_path: Path
) -> None:
    other = kernel_config(shards.vocab_size, d_ff=128)
    settings = kernel_settings(shards.tokenized_id, init_checkpoint=_ref(base))
    with pytest.raises(CheckpointError, match=r"differs from this run's in .*d_ff"):
        pretrain_packed(other, shards.data_root, tmp_path / "x", settings, max_steps=1)


def test_sparse_retrofit_warms_up_the_indexer_then_selects(
    shards: TokenShardFixture, base: Path, tmp_path: Path
) -> None:
    parent = kernel_config(shards.vocab_size)
    warmup = retrofit_config(parent, SparseAttention(top_k=8, select=False))
    settings = kernel_settings(shards.tokenized_id, init_checkpoint=_ref(base))
    pretrain_packed(warmup, shards.data_root, tmp_path / "warmup", settings, max_steps=2)
    metrics = [
        json.loads(line)
        for line in (tmp_path / "warmup" / "metrics.jsonl").read_text().splitlines()
    ]
    assert all(item["index_loss"] >= 0 for item in metrics)
    selected = retrofit_config(parent, SparseAttention(top_k=8, select=True))
    sparse = pretrain_packed(
        selected,
        shards.data_root,
        tmp_path / "selected",
        settings.model_copy(update={"init_checkpoint": _ref(tmp_path / "warmup")}),
        max_steps=2,
    )
    assert len(sparse.losses) == 2
    with pytest.raises(ValueError, match="activation checkpointing"):
        pretrain_packed(
            selected,
            shards.data_root,
            tmp_path / "refused",
            settings.model_copy(update={"activation_checkpointing": True}),
            max_steps=1,
        )


def test_long_document_selection(shards: TokenShardFixture, tmp_path: Path) -> None:
    from epor.training.shard_data import ShardCorpus

    corpus = ShardCorpus.open(shards.data_root, shards.tokenized_id, "train")
    longest = max(corpus.pair_counts)
    settings = kernel_settings(shards.tokenized_id, min_document_pairs=longest)
    pretrain_packed(
        kernel_config(shards.vocab_size), shards.data_root, tmp_path / "long", settings, max_steps=1
    )
    manifest = json.loads((tmp_path / "long" / "manifest.json").read_text())
    assert manifest["long_documents"]["documents"] == corpus.pair_counts.count(longest)
    with pytest.raises(ValueError, match="no document"):
        pretrain_packed(
            kernel_config(shards.vocab_size),
            shards.data_root,
            tmp_path / "none",
            settings.model_copy(update={"min_document_pairs": longest + 1}),
            max_steps=1,
        )
    assert "min_document_pairs" not in kernel_settings(shards.tokenized_id).trajectory()
