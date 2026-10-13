"""Deterministic, leakage-free tokenizer training and tamper-evident artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import (
    TOKENIZER_FIXTURE_VOCAB,
    VALIDATION_MARKER,
    make_tokenizer_build,
)

from epor.data.models import DatasetBuild
from epor.data.store import DataStore, DataStoreError, canonical_json_bytes, sha256_bytes
from epor.tokenizer import (
    CorpusError,
    TokenizerArtifactError,
    TrainedTokenizer,
    load_tokenizer,
    read_artifact,
    train_tokenizer,
)
from epor.tokenizer.artifact import MANIFEST_FILE, TOKENIZER_FILE, TokenizerManifest
from epor.tokenizer.spec import FIRST_MERGE_ID


def _train(
    root: Path,
    *,
    marker_split: str | None = "validation",
    extra_train_text: str | None = None,
) -> tuple[DatasetBuild, TrainedTokenizer]:
    build = make_tokenizer_build(root, marker_split=marker_split, extra_train_text=extra_train_text)
    return build, train_tokenizer(
        root / "data", build.dataset_id, build.build_id, vocab_size=TOKENIZER_FIXTURE_VOCAB
    )


def _vocab(directory: Path) -> dict[str, int]:
    return json.loads((directory / TOKENIZER_FILE).read_bytes())["model"]["vocab"]


def test_repeated_training_is_identical_and_idempotent(tmp_path: Path) -> None:
    build, first = _train(tmp_path / "a")
    again = train_tokenizer(
        tmp_path / "a" / "data",
        build.dataset_id,
        build.build_id,
        vocab_size=TOKENIZER_FIXTURE_VOCAB,
    )
    _, independent = _train(tmp_path / "b")

    assert again == first
    assert independent.tokenizer_id == first.tokenizer_id
    assert independent.manifest == first.manifest
    for name in (TOKENIZER_FILE, MANIFEST_FILE):
        assert (independent.directory / name).read_bytes() == (first.directory / name).read_bytes()
    assert first.directory == DataStore(tmp_path / "a" / "data").tokenizer_path(first.tokenizer_id)
    assert sha256_bytes((first.directory / MANIFEST_FILE).read_bytes()) == first.tokenizer_id


def test_manifest_records_provenance_and_settings(tmp_path: Path) -> None:
    build, trained = _train(tmp_path)
    manifest = trained.manifest
    train_docs = [item for item in build.documents if item.split == "train"]

    assert manifest.training.dataset_id == build.dataset_id
    assert manifest.training.build_id == build.build_id
    assert manifest.training.split == "train"
    assert manifest.training.split_sha256 == build.artifacts["splits/train.jsonl"]
    assert manifest.training.document_count == len(train_docs) == 3
    assert manifest.training.byte_count > 0
    assert manifest.trainer.parallelism == "single-threaded"
    assert manifest.trainer.pre_tokenizer["use_regex"] is True
    assert manifest.trainer.normalizer is None
    assert manifest.tokenizer_json_sha256 == sha256_bytes(
        (trained.directory / TOKENIZER_FILE).read_bytes()
    )


def test_changing_inputs_or_settings_changes_the_manifest(tmp_path: Path) -> None:
    _, base = _train(tmp_path / "base")
    _, extra = _train(tmp_path / "extra", extra_train_text="An additional training document. " * 9)
    build = make_tokenizer_build(tmp_path / "settings")
    other_size = train_tokenizer(
        tmp_path / "settings" / "data", build.dataset_id, build.build_id, vocab_size=900
    )

    assert extra.tokenizer_id != base.tokenizer_id
    assert extra.manifest.training.build_id != base.manifest.training.build_id
    assert other_size.tokenizer_id != base.tokenizer_id


def test_training_reads_only_the_train_split(tmp_path: Path) -> None:
    held_out_build, held_out = _train(tmp_path / "held-out", marker_split="validation")
    _, absent = _train(tmp_path / "absent", marker_split=None)
    _, leaked = _train(tmp_path / "leaked", marker_split="train")

    marker_doc = [item for item in held_out_build.documents if item.source_id == "marker-corpus"]
    assert [item.split for item in marker_doc] == ["validation"]
    # A validation-only document contributes nothing: the learned tokenizer is
    # byte-identical to one trained without that document at all.
    assert held_out.manifest.tokenizer_json_sha256 == absent.manifest.tokenizer_json_sha256
    assert held_out.manifest.training.document_count == 3
    assert not any("qzv" in token for token in _vocab(held_out.directory))
    # Control: the same marker in the train split is learned, so the check above
    # would catch leakage.
    assert any(VALIDATION_MARKER in token for token in _vocab(leaked.directory))


def test_small_corpus_reports_learned_size_honestly(tmp_path: Path) -> None:
    build = make_tokenizer_build(tmp_path)
    trained = train_tokenizer(tmp_path / "data", build.dataset_id, build.build_id)
    manifest = trained.manifest

    assert manifest.vocab_size == 65_536
    assert FIRST_MERGE_ID < manifest.learned_vocab_size < manifest.vocab_size
    assert manifest.merge_count == manifest.learned_vocab_size - FIRST_MERGE_ID
    assert len(_vocab(trained.directory)) == manifest.learned_vocab_size
    loaded = load_tokenizer(tmp_path / "data", trained.tokenizer_id)
    assert loaded.vocab_size == 65_536
    assert loaded.learned_vocab_size == manifest.learned_vocab_size


def test_manifest_rejects_inflated_learned_size(tmp_path: Path) -> None:
    _, trained = _train(tmp_path)
    payload = trained.manifest.model_dump(mode="json")
    payload["learned_vocab_size"] = payload["vocab_size"] + 1
    with pytest.raises(ValueError, match="cannot exceed"):
        TokenizerManifest.model_validate(payload)


@pytest.mark.parametrize("target", [TOKENIZER_FILE, MANIFEST_FILE])
def test_tampered_artifacts_are_refused(tmp_path: Path, target: str) -> None:
    _, trained = _train(tmp_path)
    path = trained.directory / target
    document = json.loads(path.read_bytes())
    if target == TOKENIZER_FILE:
        document["model"]["merges"] = document["model"]["merges"][:-1]
    else:
        document["training"]["document_count"] += 1
    path.write_bytes(canonical_json_bytes(document))

    with pytest.raises(TokenizerArtifactError):
        load_tokenizer(tmp_path / "data", trained.tokenizer_id)


def test_rehashed_artifact_with_drifted_reserved_ids_is_refused(tmp_path: Path) -> None:
    _, trained = _train(tmp_path)
    document = json.loads((trained.directory / TOKENIZER_FILE).read_bytes())
    vocab = document["model"]["vocab"]
    vocab["<|bos|>"], vocab["<|eos|>"] = vocab["<|eos|>"], vocab["<|bos|>"]
    forged_json = canonical_json_bytes(document)
    forged_manifest = trained.manifest.model_copy(
        update={"tokenizer_json_sha256": sha256_bytes(forged_json)}
    )
    manifest_bytes = canonical_json_bytes(forged_manifest)
    forged = tmp_path / "forged" / sha256_bytes(manifest_bytes)
    forged.mkdir(parents=True)
    (forged / TOKENIZER_FILE).write_bytes(forged_json)
    (forged / MANIFEST_FILE).write_bytes(manifest_bytes)

    with pytest.raises(TokenizerArtifactError, match="reserved ID block"):
        read_artifact(forged)


def test_existing_artifact_address_cannot_be_overwritten(tmp_path: Path) -> None:
    _, trained = _train(tmp_path)
    store = DataStore(tmp_path / "data")
    with pytest.raises(DataStoreError, match="immutable"):
        store.write_tokenizer_file(trained.tokenizer_id, TOKENIZER_FILE, b"{}")
    with pytest.raises(DataStoreError, match="SHA-256"):
        store.tokenizer_path("../builds")
    with pytest.raises(TokenizerArtifactError, match="not found"):
        load_tokenizer(store, "0" * 64)


def test_tampered_train_split_is_refused(tmp_path: Path) -> None:
    build = make_tokenizer_build(tmp_path)
    store = DataStore(tmp_path / "data")
    shard = store.build_path(build.dataset_id, build.build_id) / "splits/train.jsonl"
    shard.write_bytes(shard.read_bytes() + b"\n")
    with pytest.raises(CorpusError, match="does not match the build manifest"):
        train_tokenizer(store, build.dataset_id, build.build_id, vocab_size=TOKENIZER_FIXTURE_VOCAB)


def test_training_honours_cancellation_and_limits(tmp_path: Path) -> None:
    build = make_tokenizer_build(tmp_path)

    class Cancelled(RuntimeError):
        pass

    def cancel() -> None:
        raise Cancelled

    with pytest.raises(Cancelled):
        train_tokenizer(tmp_path / "data", build.dataset_id, build.build_id, check_cancelled=cancel)
    with pytest.raises(ValueError, match="at least"):
        train_tokenizer(tmp_path / "data", build.dataset_id, build.build_id, vocab_size=512)
    with pytest.raises(CorpusError, match="not found"):
        train_tokenizer(tmp_path / "data", build.dataset_id, "0" * 64)
    assert not list((tmp_path / "data" / "tokenizers").iterdir())
