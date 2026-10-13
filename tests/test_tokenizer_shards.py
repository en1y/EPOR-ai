"""Content-addressed, boundary-preserving token shards and removal impact."""

from __future__ import annotations

import shutil
from array import array
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from conftest import TOKENIZER_FIXTURE_VOCAB, VALIDATION_MARKER, make_tokenizer_build

from epor.data.models import DatasetBuild
from epor.data.service import DataEngine
from epor.data.store import DataStore, sha256_bytes
from epor.tokenizer import (
    DOC_END_ID,
    DOC_START_ID,
    TokenizedBuild,
    TokenizedBuildError,
    affected_tokenized_builds,
    build_token_shards,
    iter_split_documents,
    iter_tokenized_documents,
    load_tokenizer,
    train_tokenizer,
    verify_tokenized_build,
)
from epor.tokenizer.corpus import Split
from epor.tokenizer.shards import (
    INDEX_DIR,
    SHARD_DIR,
    SHARD_SUFFIX,
    SPLITS,
    ShardIndex,
    SplitShards,
)

SMALL_SHARD = 64


@dataclass(frozen=True)
class Corpus:
    root: Path
    build: DatasetBuild
    tokenizer_id: str

    @property
    def data(self) -> Path:
        return self.root / "data"

    def copy(self, destination: Path) -> Corpus:
        shutil.copytree(self.root, destination)
        return Corpus(destination, self.build, self.tokenizer_id)


@pytest.fixture(scope="module")
def corpus(tmp_path_factory: pytest.TempPathFactory) -> Corpus:
    root = tmp_path_factory.mktemp("shards")
    build = make_tokenizer_build(root)
    trained = train_tokenizer(
        root / "data", build.dataset_id, build.build_id, vocab_size=TOKENIZER_FIXTURE_VOCAB
    )
    return Corpus(root, build, trained.tokenizer_id)


@pytest.fixture
def fresh(corpus: Corpus, tmp_path: Path) -> Corpus:
    """A private copy for tests that write shards or tombstones."""

    return corpus.copy(tmp_path / "copy")


def _shard(corpus: Corpus, **kwargs: Any) -> TokenizedBuild:
    return build_token_shards(
        corpus.data, corpus.build.dataset_id, corpus.build.build_id, corpus.tokenizer_id, **kwargs
    )


def _shard_files(directory: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(directory)): path.read_bytes()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def _raw_shards(
    directory: Path, manifest_splits: dict[Split, SplitShards], split: Split
) -> Iterator[tuple[ShardIndex, list[int]]]:
    for record in manifest_splits[split].shards:
        index = ShardIndex.model_validate_json(
            (directory / INDEX_DIR / f"{record.index_sha256}.json").read_bytes()
        )
        tokens = array("H")
        tokens.frombytes((directory / SHARD_DIR / f"{record.sha256}{SHARD_SUFFIX}").read_bytes())
        yield index, tokens.tolist()


def test_repeated_builds_are_identical_and_idempotent(fresh: Corpus, tmp_path: Path) -> None:
    independent = fresh.copy(tmp_path / "independent")
    first = _shard(fresh)
    again = _shard(fresh)
    other = _shard(independent)

    assert first.created and not again.created
    assert again.tokenized_id == first.tokenized_id == other.tokenized_id
    assert again.manifest == first.manifest == other.manifest
    assert _shard_files(first.directory) == _shard_files(other.directory)
    assert sha256_bytes((first.directory / "manifest.json").read_bytes()) == first.tokenized_id
    assert first.directory == DataStore(fresh.data).tokenized_path(first.tokenized_id)
    assert not any((fresh.data / "tmp").iterdir())

    manifest = first.manifest
    assert manifest.tokenizer_id == fresh.tokenizer_id
    assert manifest.build_id == fresh.build.build_id
    for split in SPLITS:
        assert manifest.splits[split].split_sha256 == fresh.build.artifacts[f"splits/{split}.jsonl"]
    assert manifest.total_tokens == sum(item.token_count for item in manifest.splits.values())
    assert manifest.excluded_document_count == 0 and manifest.applied_tombstone_ids == []
    assert verify_tokenized_build(fresh.data, first.tokenized_id) == manifest


def test_different_inputs_give_different_ids(fresh: Corpus, tmp_path: Path) -> None:
    baseline = _shard(fresh).tokenized_id
    smaller = train_tokenizer(
        fresh.data, fresh.build.dataset_id, fresh.build.build_id, vocab_size=600
    ).tokenizer_id
    other_tokenizer = build_token_shards(
        fresh.data, fresh.build.dataset_id, fresh.build.build_id, smaller
    )
    other_limit = _shard(fresh, max_shard_tokens=SMALL_SHARD)

    root = tmp_path / "other-build"
    build = make_tokenizer_build(root, extra_train_text="Another document changes the build.\n")
    # Same tokenizer artifact, different build: copy it into the other store.
    shutil.copytree(
        DataStore(fresh.data).tokenizer_path(fresh.tokenizer_id),
        DataStore(root / "data").tokenizer_path(fresh.tokenizer_id),
    )
    other_build_id = build_token_shards(
        root / "data", build.dataset_id, build.build_id, fresh.tokenizer_id
    ).tokenized_id

    ids = {baseline, other_tokenizer.tokenized_id, other_limit.tokenized_id, other_build_id}
    assert len(ids) == 4


def test_decoded_spans_reproduce_source_text(fresh: Corpus) -> None:
    built = _shard(fresh)
    tokenizer = load_tokenizer(fresh.data, fresh.tokenizer_id)
    store = DataStore(fresh.data)
    for split in SPLITS:
        source = [
            (doc.document_id, doc.text) for doc in iter_split_documents(store, fresh.build, split)
        ]
        # Through the raw index: each span minus its boundary tokens.
        raw = []
        for index, tokens in _raw_shards(built.directory, built.manifest.splits, split):
            for entry in index.entries:
                assert entry.part_count == 1
                span = tokens[entry.offset : entry.offset + entry.length]
                assert span[0] == DOC_START_ID and span[-1] == DOC_END_ID
                raw.append((entry.document_id, tokenizer.decode_bytes(span[1:-1]).decode()))
        assert raw == source
        decoded = [
            (document_id, tokenizer.decode_bytes(ids).decode())
            for document_id, ids in iter_tokenized_documents(fresh.data, built.tokenized_id, split)
        ]
        assert decoded == source


def test_ids_fit_the_vocabulary_and_reserved_ids_are_only_boundaries(fresh: Corpus) -> None:
    built = _shard(fresh)
    vocab = load_tokenizer(fresh.data, fresh.tokenizer_id).vocab_size
    for split in SPLITS:
        for index, tokens in _raw_shards(built.directory, built.manifest.splits, split):
            assert all(token < 65_536 and token < vocab for token in tokens)
            boundaries = {entry.offset for entry in index.entries} | {
                entry.offset + entry.length - 1 for entry in index.entries
            }
            reserved = {position for position, token in enumerate(tokens) if token < 256}
            assert reserved == boundaries
            assert all(tokens[position] in (DOC_START_ID, DOC_END_ID) for position in reserved)


def test_documents_appear_only_in_their_own_split(fresh: Corpus) -> None:
    built = _shard(fresh)
    seen: dict[str, set[str]] = {}
    for split in SPLITS:
        seen[split] = {
            entry.document_id
            for index, _ in _raw_shards(built.directory, built.manifest.splits, split)
            for entry in index.entries
        }
        expected = {item.document_id for item in fresh.build.documents if item.split == split}
        assert seen[split] == expected
        assert built.manifest.splits[split].document_count == len(expected)
    assert seen["train"] and seen["validation"]
    assert not seen["train"] & seen["validation"]


def test_shard_limit_and_oversized_documents(fresh: Corpus) -> None:
    built = _shard(fresh, max_shard_tokens=SMALL_SHARD)
    tokenizer = load_tokenizer(fresh.data, fresh.tokenizer_id)
    store = DataStore(fresh.data)
    continued = 0
    for split in SPLITS:
        shards = list(_raw_shards(built.directory, built.manifest.splits, split))
        for index, tokens in shards:
            assert 0 < len(tokens) <= SMALL_SHARD
            for entry in index.entries:
                if entry.part_count > 1:
                    continued += 1
                    # Pieces of an oversized document each fill their shard.
                    if entry.part < entry.part_count - 1:
                        assert entry.offset + entry.length == SMALL_SHARD
                    if entry.part > 0:
                        assert entry.offset == 0
        source = [doc.text for doc in iter_split_documents(store, fresh.build, split)]
        decoded = [
            tokenizer.decode_bytes(ids).decode()
            for _, ids in iter_tokenized_documents(fresh.data, built.tokenized_id, split)
        ]
        assert decoded == source
    assert continued > 0
    verify_tokenized_build(fresh.data, built.tokenized_id)

    with pytest.raises(ValueError, match="max_shard_tokens"):
        _shard(fresh, max_shard_tokens=3)


def test_small_documents_are_never_split(fresh: Corpus) -> None:
    built = _shard(fresh, max_shard_tokens=100_000)
    for split in SPLITS:
        shards = list(_raw_shards(built.directory, built.manifest.splits, split))
        assert len(shards) <= 1
        assert all(entry.part_count == 1 for index, _ in shards for entry in index.entries)


def test_tombstoned_documents_are_excluded_and_affected_builds_found(fresh: Corpus) -> None:
    before = _shard(fresh)
    store = DataStore(fresh.data)
    marker = next(
        record for record in store.documents() if VALIDATION_MARKER in record.text
    ).document_id
    unrelated = "0" * 64

    affected = affected_tokenized_builds(fresh.data, marker)
    assert [(item.tokenized_id, item.split) for item in affected] == [
        (before.tokenized_id, "validation")
    ]
    assert affected[0].shard_sha256s == tuple(
        item.sha256 for item in before.manifest.splits["validation"].shards
    )
    assert affected_tokenized_builds(fresh.data, unrelated) == []

    tombstone = DataEngine(fresh.data).remove(
        target_kind="document_id",
        target=marker,
        reason="Fixture removal request.",
        requested_by="tests",
    )
    after = _shard(fresh)
    assert after.tokenized_id != before.tokenized_id
    assert after.manifest.excluded_document_count == 1
    assert after.manifest.splits["validation"].excluded_document_count == 1
    assert after.manifest.splits["validation"].document_count == (
        before.manifest.splits["validation"].document_count - 1
    )
    assert after.manifest.applied_tombstone_ids == [tombstone.id]
    remaining = {
        document_id
        for split in SPLITS
        for document_id, _ in iter_tokenized_documents(fresh.data, after.tokenized_id, split)
    }
    assert marker not in remaining
    assert [item.tokenized_id for item in affected_tokenized_builds(fresh.data, marker)] == [
        before.tokenized_id
    ]


def test_source_tombstone_uses_the_data_engine_rule(fresh: Corpus) -> None:
    DataEngine(fresh.data).remove(
        target_kind="source_id",
        target="train-corpus",
        reason="Fixture source withdrawal.",
        requested_by="tests",
    )
    built = _shard(fresh)
    train = built.manifest.splits["train"]
    assert train.document_count == 0 and train.shards == []
    assert train.excluded_document_count == 3
    verify_tokenized_build(fresh.data, built.tokenized_id)


def test_tampering_is_detected(fresh: Corpus) -> None:
    built = _shard(fresh)
    record = built.manifest.splits["train"].shards[0]
    shard = built.directory / SHARD_DIR / f"{record.sha256}{SHARD_SUFFIX}"
    original = shard.read_bytes()
    shard.write_bytes(bytes([original[0] ^ 1]) + original[1:])
    with pytest.raises(TokenizedBuildError, match="shard"):
        verify_tokenized_build(fresh.data, built.tokenized_id)
    # A rebuild refuses to vouch for the tampered artifact at the same address.
    with pytest.raises(TokenizedBuildError):
        _shard(fresh)
    shard.write_bytes(original)

    index = built.directory / INDEX_DIR / f"{record.index_sha256}.json"
    index_bytes = index.read_bytes()
    index.write_bytes(index_bytes.replace(b'"offset":0', b'"offset":1', 1))
    with pytest.raises(TokenizedBuildError, match="index"):
        verify_tokenized_build(fresh.data, built.tokenized_id)
    index.write_bytes(index_bytes)

    manifest = built.directory / "manifest.json"
    manifest.write_bytes(manifest.read_bytes() + b" ")
    with pytest.raises(TokenizedBuildError, match="content address"):
        verify_tokenized_build(fresh.data, built.tokenized_id)


def test_cancellation_leaves_no_partial_output(fresh: Corpus) -> None:
    class Cancelled(RuntimeError):
        pass

    calls = 0

    def check() -> None:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise Cancelled

    with pytest.raises(Cancelled):
        _shard(fresh, check_cancelled=check)
    assert not any((fresh.data / "tokenized").iterdir())
    assert not any((fresh.data / "tmp").iterdir())
