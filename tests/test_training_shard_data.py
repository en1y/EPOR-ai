"""Verified, memory-mapped reads of token shards for training (v0.0.5 data path)."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from conftest import TokenShardFixture, make_token_shards  # noqa: E402

from epor.models import ModelConfig, load_model_config  # noqa: E402
from epor.tokenizer import (  # noqa: E402
    BOS_ID,
    EOS_ID,
    TokenizedBuildError,
    iter_tokenized_documents,
)
from epor.tokenizer.shards import SHARD_DIR  # noqa: E402
from epor.training.shard_data import ShardCorpus  # noqa: E402

TINY_CONFIG = Path("configs/models/epor-tiny.yaml")


def tiny_config(vocab_size: int, **overrides: Any) -> ModelConfig:
    base = load_model_config(TINY_CONFIG)
    return ModelConfig.model_validate(
        {**base.model_dump(mode="json"), "vocab_size": vocab_size, **overrides}
    )


@pytest.fixture(scope="module")
def shards(tmp_path_factory: pytest.TempPathFactory) -> TokenShardFixture:
    return make_token_shards(tmp_path_factory.mktemp("shard-data"))


def test_documents_match_the_tokenizer_reader_including_multi_part(
    shards: TokenShardFixture,
) -> None:
    config = tiny_config(shards.vocab_size)
    for split in ("train", "validation"):
        corpus = ShardCorpus.open(
            shards.data_root,
            shards.tokenized_id,
            split,
            model_config=config,
            tokenizer_id=shards.tokenizer_id,
        )
        expected = list(iter_tokenized_documents(shards.data_root, shards.tokenized_id, split))
        assert corpus.document_ids == tuple(document_id for document_id, _ in expected)
        for index, (_, content) in enumerate(expected):
            framed = corpus.framed_tokens(index).tolist()
            assert framed == [BOS_ID, *content, EOS_ID]
            assert corpus.pair_counts[index] == len(content) + 1
    train = ShardCorpus.open(shards.data_root, shards.tokenized_id, "train")
    # The 64-token shard limit must have split at least one document across shards.
    assert any(count + 1 > 64 for count in train.pair_counts)
    assert train.max_context is None


def test_pairs_are_shifted_framed_tokens(shards: TokenShardFixture) -> None:
    corpus = ShardCorpus.open(shards.data_root, shards.tokenized_id, "train")
    document = max(range(corpus.document_count), key=lambda item: corpus.pair_counts[item])
    framed = corpus.framed_tokens(document)
    # A window straddling a shard boundary of a multi-part document.
    inputs, targets = corpus.pairs(document, 60, 10)
    assert inputs.tolist() == framed[60:70].tolist()
    assert targets.tolist() == framed[61:71].tolist()
    last = corpus.pair_counts[document] - 1
    tail_inputs, tail_targets = corpus.pairs(document, last, 1)
    assert tail_targets.tolist() == [EOS_ID]
    assert tail_inputs.tolist() != [EOS_ID]
    with pytest.raises(IndexError):
        corpus.pairs(document, last, 2)


def test_refuses_vocabulary_and_special_id_mismatch(shards: TokenShardFixture) -> None:
    with pytest.raises(TokenizedBuildError, match="vocab"):
        ShardCorpus.open(
            shards.data_root,
            shards.tokenized_id,
            "train",
            model_config=tiny_config(shards.vocab_size * 2),
        )
    with pytest.raises(TokenizedBuildError, match="reserved"):
        ShardCorpus.open(
            shards.data_root,
            shards.tokenized_id,
            "train",
            model_config=tiny_config(shards.vocab_size, bos_token_id=15, eos_token_id=16),
        )
    with pytest.raises(TokenizedBuildError, match="different tokenizer"):
        ShardCorpus.open(shards.data_root, shards.tokenized_id, "train", tokenizer_id="0" * 64)


def test_refuses_a_tampered_shard(shards: TokenShardFixture, tmp_path: Path) -> None:
    root = tmp_path / "data"
    shutil.copytree(shards.data_root, root)
    shard = next((root / "tokenized" / shards.tokenized_id / SHARD_DIR).iterdir())
    shard.chmod(0o600)
    payload = bytearray(shard.read_bytes())
    payload[2] ^= 1
    shard.write_bytes(bytes(payload))
    with pytest.raises(TokenizedBuildError, match="hash"):
        ShardCorpus.open(root, shards.tokenized_id, "train")
    with pytest.raises(TokenizedBuildError, match="not found"):
        ShardCorpus.open(root, "f" * 64, "train")
