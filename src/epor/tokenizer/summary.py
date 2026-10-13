"""Read-only summary of stored tokenizers, token shards, and fertility reports.

Every entry is read through a loader that checks its content address and
canonical form, so the summary never reports an artifact whose bytes no longer
match its ID. A tokenizer is read in full (``read_artifact`` re-validates the
layout); a tokenized build is checked at the manifest level only, because
re-hashing every shard on each page load would scale with the corpus. Use
``verify_tokenized_build`` for the full shard check. An artifact that fails
its check is listed under ``integrity_errors`` instead of being summarized.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from epor.data.store import DataStore, DataStoreError

from .artifact import TokenizerArtifactError, TokenizerManifest, read_artifact
from .evaluate import FertilityReport, load_fertility_report
from .shards import TokenizedBuildError, TokenizedManifest, load_tokenized_manifest

_READ_ERRORS = (TokenizerArtifactError, TokenizedBuildError, DataStoreError, OSError, ValueError)


def describe_tokenizer(tokenizer_id: str, manifest: TokenizerManifest) -> dict[str, Any]:
    """Operator-facing tokenizer facts; declared and learned sizes stay distinct."""

    return {
        "tokenizer_id": tokenizer_id,
        "tokenizer_version": manifest.tokenizer_version,
        "reserved_spec_version": manifest.reserved_spec_version,
        "declared_vocab_size": manifest.vocab_size,
        "learned_vocab_size": manifest.learned_vocab_size,
        "merge_count": manifest.merge_count,
        "dataset_id": manifest.training.dataset_id,
        "build_id": manifest.training.build_id,
        "train_split_sha256": manifest.training.split_sha256,
        "train_documents": manifest.training.document_count,
        "train_bytes": manifest.training.byte_count,
        "min_frequency": manifest.trainer.min_frequency,
        "library_version": manifest.trainer.library_version,
    }


def describe_tokenized(tokenized_id: str, manifest: TokenizedManifest) -> dict[str, Any]:
    return {
        "tokenized_id": tokenized_id,
        "tokenizer_id": manifest.tokenizer_id,
        "dataset_id": manifest.dataset_id,
        "build_id": manifest.build_id,
        "max_shard_tokens": manifest.max_shard_tokens,
        "total_tokens": manifest.total_tokens,
        "total_documents": manifest.total_documents,
        "excluded_document_count": manifest.excluded_document_count,
        "applied_tombstone_ids": manifest.applied_tombstone_ids,
        "shard_count": sum(len(item.shards) for item in manifest.splits.values()),
        "splits": {
            split: {
                "document_count": item.document_count,
                "excluded_document_count": item.excluded_document_count,
                "token_count": item.token_count,
                "shard_count": len(item.shards),
            }
            for split, item in manifest.splits.items()
        },
    }


def describe_report(report: FertilityReport) -> dict[str, Any]:
    return {"report_id": report.report_id, **report.model_dump(mode="json")}


def tokenizer_summary(store: DataStore | Path) -> dict[str, Any]:
    data_store = store if isinstance(store, DataStore) else DataStore(store)
    errors: list[str] = []

    tokenizers: list[dict[str, Any]] = []
    for tokenizer_id in data_store.tokenizer_ids():
        try:
            _, manifest, _ = read_artifact(data_store.tokenizer_path(tokenizer_id))
        except _READ_ERRORS as exc:
            errors.append(f"tokenizer {tokenizer_id}: {exc}")
            continue
        tokenizers.append(describe_tokenizer(tokenizer_id, manifest))

    tokenized: list[dict[str, Any]] = []
    for tokenized_id in data_store.tokenized_ids():
        try:
            shards = load_tokenized_manifest(data_store, tokenized_id)
        except _READ_ERRORS as exc:
            errors.append(f"tokenized build {tokenized_id}: {exc}")
            continue
        tokenized.append(describe_tokenized(tokenized_id, shards))

    reports: list[dict[str, Any]] = []
    for report_id in data_store.evaluation_ids():
        try:
            report = load_fertility_report(data_store, report_id)
        except _READ_ERRORS as exc:
            errors.append(f"fertility report {report_id}: {exc}")
            continue
        reports.append(describe_report(report))

    return {
        "schema_version": 1,
        "tokenizers": tokenizers,
        "tokenized_builds": tokenized,
        "fertility_reports": reports,
        "integrity_errors": errors,
    }
