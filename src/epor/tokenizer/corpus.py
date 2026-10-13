"""Verified, split-scoped reads of a data-engine build.

Tokenizer training must see only the ``train`` split. These helpers read one
split shard, verify it against the immutable build manifest, and refuse any row
that the manifest does not assign to that split, so a corrupted or edited shard
cannot leak validation or test text into training.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal

from epor.data.models import DatasetBuild
from epor.data.store import DataStore, canonical_json_bytes, sha256_bytes, sha256_file

Split = Literal["train", "validation", "test"]


class CorpusError(RuntimeError):
    """A build or split shard does not match its immutable manifest."""


@dataclass(frozen=True, slots=True)
class SplitDocument:
    document_id: str
    source_id: str
    relative_input_name: str
    source_order: int
    text: str


def load_build(store: DataStore, dataset_id: str, build_id: str) -> DatasetBuild:
    """Load a build manifest and verify that it is addressed by its own build ID."""

    path = store.build_path(dataset_id, build_id) / "manifest.json"
    try:
        payload = path.read_bytes()
    except FileNotFoundError as exc:
        raise CorpusError(f"dataset build not found: {dataset_id}/{build_id}") from exc
    build = DatasetBuild.model_validate_json(payload)
    if build.dataset_id != dataset_id or build.build_id != build_id:
        raise CorpusError("build manifest does not match its storage address")
    if canonical_json_bytes(build) != payload:
        raise CorpusError("build manifest is not in canonical form")
    return build


def split_artifact_name(split: Split) -> str:
    return f"splits/{split}.jsonl"


def split_sha256(build: DatasetBuild, split: Split) -> str:
    try:
        return build.artifacts[split_artifact_name(split)]
    except KeyError as exc:
        raise CorpusError(f"build {build.build_id} has no {split} split artifact") from exc


def iter_split_documents(
    store: DataStore,
    build: DatasetBuild,
    split: Split,
) -> Iterator[SplitDocument]:
    """Yield one split's documents in the build's deterministic row order."""

    path = store.build_path(build.dataset_id, build.build_id) / split_artifact_name(split)
    expected = split_sha256(build, split)
    if not path.is_file() or sha256_file(path) != expected:
        raise CorpusError(f"{split} split shard does not match the build manifest")
    allowed = {item.document_id for item in build.documents if item.split == split}
    seen: set[str] = set()
    with path.open("rb") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            document_id = row["document_id"]
            if document_id not in allowed or document_id in seen:
                raise CorpusError(f"{split} shard row is not assigned to that split once")
            text = row["text"]
            if sha256_bytes(text.encode("utf-8")) != row["normalized_sha256"]:
                raise CorpusError(f"{split} shard row text does not match its hash")
            seen.add(document_id)
            yield SplitDocument(
                document_id=document_id,
                source_id=row["source_id"],
                relative_input_name=row["relative_input_name"],
                source_order=row["source_order"],
                text=text,
            )
    if seen != allowed:
        raise CorpusError(f"{split} shard is missing documents listed in the manifest")
