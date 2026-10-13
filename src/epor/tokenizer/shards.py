"""Content-addressed token shards that preserve document boundaries.

A tokenized build pins one tokenizer artifact to one data-engine build and
stores every split's documents as fixed-width token arrays:

* **Format.** Each shard is a flat array of little-endian ``uint16`` token IDs
  (the 65,536-entry vocabulary fits exactly). Every document is written as
  ``<|doc_start|>`` + content tokens + ``<|doc_end|>``; those two reserved IDs
  are the only reserved IDs that ever appear in a shard. BOS/EOS are left to
  the training loader, which frames sequences, not documents.
* **Packing.** Documents are packed greedily in the build's stable split order
  and never split across shards, unless one framed document alone exceeds
  ``max_shard_tokens``. Such a document is cut into consecutive pieces that
  each fill a shard; its index entries record ``part``/``part_count`` so the
  continuation is explicit. ``<|doc_start|>`` sits only in part 0 and
  ``<|doc_end|>`` only in the last part.
* **Index.** Every shard has a canonical JSON index listing, per document
  span, ``document_id``, token ``offset``, token ``length`` (boundary tokens
  included), ``part`` and ``part_count``, so boundaries are recoverable without
  scanning token values.
* **Addressing.** Shards are stored as ``shards/<sha256>.u16`` and indexes as
  ``indexes/<sha256>.json``; the manifest pins both hashes, the tokenizer and
  reserved-spec hashes, the build and per-split ``split_sha256``, the shard
  size limit, and the removal state. The tokenized build ID is the SHA-256 of
  the canonical manifest and the build lives at
  ``<data_root>/tokenized/<tokenized_id>/``. Nothing time- or host-dependent is
  recorded, so identical inputs give an identical ID and a rebuild is a no-op.
* **Atomicity.** The whole build is written to a staging directory under
  ``<data_root>/tmp`` and renamed into place only when complete; failure or
  cancellation removes the staging directory and leaves no partial artifact.
* **Removal.** Documents covered by a tombstone at shard time (including
  tombstones recorded after the parent build) are excluded with the data
  engine's own matching rule; the manifest records the excluded count and the
  tombstone IDs that applied. ``affected_tokenized_builds`` answers which
  stored tokenized builds and shards contain a given document, so a later
  removal can identify what must be rebuilt.
"""

from __future__ import annotations

import os
import sys
from array import array
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

from pydantic import Field, ValidationError

from epor.data.models import FrozenSchema, Identifier, Sha256
from epor.data.store import DataStore, canonical_json_bytes, sha256_bytes

from .artifact import TokenizerArtifactError
from .codec import load_tokenizer
from .corpus import (
    Split,
    iter_split_documents,
    load_build,
    split_sha256,
    tombstoned_documents,
)
from .spec import (
    DOC_END_ID,
    DOC_START_ID,
    RESERVED_COUNT,
    RESERVED_SPEC_SHA256,
    RESERVED_SPEC_VERSION,
)

SPLITS: Final[tuple[Split, ...]] = ("train", "validation", "test")
TOKEN_ID_LIMIT: Final = 1 << 16
DEFAULT_MAX_SHARD_TOKENS: Final = 1 << 24  # 16,777,216 tokens = 32 MiB per shard
MIN_MAX_SHARD_TOKENS: Final = 4
MANIFEST_FILE: Final = "manifest.json"
SHARD_DIR: Final = "shards"
INDEX_DIR: Final = "indexes"
SHARD_SUFFIX: Final = ".u16"


class TokenizedBuildError(RuntimeError):
    """A tokenized build is missing, tampered with, or violates its contract."""


class TokenFormat(FrozenSchema):
    dtype: Literal["uint16"] = "uint16"
    byteorder: Literal["little"] = "little"
    document_start_id: Literal[15] = 15
    document_end_id: Literal[16] = 16


class ShardIndexEntry(FrozenSchema):
    document_id: Sha256
    offset: int = Field(ge=0)
    length: int = Field(ge=1, description="Span length in tokens, boundary tokens included.")
    part: int = Field(ge=0)
    part_count: int = Field(ge=1)


class ShardIndex(FrozenSchema):
    schema_version: Literal[1] = 1
    split: Split
    ordinal: int = Field(ge=0)
    shard_sha256: Sha256
    token_count: int = Field(ge=1)
    entries: list[ShardIndexEntry] = Field(min_length=1)


class ShardRecord(FrozenSchema):
    sha256: Sha256
    index_sha256: Sha256
    token_count: int = Field(ge=1)
    document_count: int = Field(ge=1, description="Document spans (index entries) in the shard.")


class SplitShards(FrozenSchema):
    split_sha256: Sha256
    document_count: int = Field(ge=0, description="Documents written, after exclusions.")
    excluded_document_count: int = Field(ge=0)
    token_count: int = Field(ge=0)
    shards: list[ShardRecord]


class TokenizedManifest(FrozenSchema):
    schema_version: Literal[1] = 1
    kind: Literal["epor-token-shards"] = "epor-token-shards"
    format: TokenFormat = Field(default_factory=TokenFormat)
    tokenizer_id: Sha256
    tokenizer_json_sha256: Sha256
    reserved_spec_version: str
    reserved_spec_sha256: Sha256
    vocab_size: int = Field(ge=1, le=TOKEN_ID_LIMIT)
    dataset_id: Identifier
    build_id: Sha256
    max_shard_tokens: int = Field(ge=MIN_MAX_SHARD_TOKENS)
    splits: dict[Split, SplitShards]
    total_tokens: int = Field(ge=0)
    total_documents: int = Field(ge=0)
    excluded_document_count: int = Field(ge=0)
    applied_tombstone_ids: list[Sha256]


@dataclass(frozen=True, slots=True)
class TokenizedBuild:
    tokenized_id: str
    manifest: TokenizedManifest
    directory: Path
    created: bool


@dataclass(frozen=True, slots=True)
class AffectedShards:
    tokenized_id: str
    dataset_id: str
    build_id: str
    tokenizer_id: str
    split: Split
    shard_sha256s: tuple[str, ...]


def _to_bytes(tokens: array[int]) -> bytes:
    if sys.byteorder != "little":
        tokens = array("H", tokens)
        tokens.byteswap()
    return tokens.tobytes()


def _from_bytes(payload: bytes) -> array[int]:
    tokens = array("H")
    tokens.frombytes(payload)
    if sys.byteorder != "little":
        tokens.byteswap()
    return tokens


def _write_file(path: Path, payload: bytes) -> None:
    if path.exists():
        # Content-addressed: an existing file with this name has these bytes.
        return
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o600)


class _SplitWriter:
    def __init__(self, staging: Path, split: Split, max_tokens: int) -> None:
        self._staging = staging
        self._split: Split = split
        self._max = max_tokens
        self._tokens: array[int] = array("H")
        self._entries: list[ShardIndexEntry] = []
        self.shards: list[ShardRecord] = []
        self.token_count = 0

    def _append(self, document_id: str, piece: list[int], part: int, part_count: int) -> None:
        self._entries.append(
            ShardIndexEntry(
                document_id=document_id,
                offset=len(self._tokens),
                length=len(piece),
                part=part,
                part_count=part_count,
            )
        )
        self._tokens.extend(piece)
        if len(self._tokens) >= self._max:
            self.flush()

    def add(self, document_id: str, framed: list[int]) -> None:
        if self._tokens and len(self._tokens) + len(framed) > self._max:
            self.flush()
        pieces = [framed[start : start + self._max] for start in range(0, len(framed), self._max)]
        for part, piece in enumerate(pieces):
            self._append(document_id, piece, part, len(pieces))

    def flush(self) -> None:
        if not self._tokens:
            return
        payload = _to_bytes(self._tokens)
        shard_sha256 = sha256_bytes(payload)
        index = ShardIndex(
            split=self._split,
            ordinal=len(self.shards),
            shard_sha256=shard_sha256,
            token_count=len(self._tokens),
            entries=self._entries,
        )
        index_bytes = canonical_json_bytes(index)
        index_sha256 = sha256_bytes(index_bytes)
        _write_file(self._staging / SHARD_DIR / f"{shard_sha256}{SHARD_SUFFIX}", payload)
        _write_file(self._staging / INDEX_DIR / f"{index_sha256}.json", index_bytes)
        self.shards.append(
            ShardRecord(
                sha256=shard_sha256,
                index_sha256=index_sha256,
                token_count=len(self._tokens),
                document_count=len(self._entries),
            )
        )
        self.token_count += len(self._tokens)
        self._tokens = array("H")
        self._entries = []


def build_token_shards(
    store: DataStore | Path,
    dataset_id: str,
    build_id: str,
    tokenizer_id: str,
    *,
    max_shard_tokens: int = DEFAULT_MAX_SHARD_TOKENS,
    check_cancelled: Callable[[], None] | None = None,
) -> TokenizedBuild:
    """Encode every split of a build into boundary-preserving token shards."""

    if max_shard_tokens < MIN_MAX_SHARD_TOKENS:
        raise ValueError(f"max_shard_tokens must be at least {MIN_MAX_SHARD_TOKENS}")
    data_store = store if isinstance(store, DataStore) else DataStore(store)
    build = load_build(data_store, dataset_id, build_id)
    tokenizer = load_tokenizer(data_store, tokenizer_id)
    limit = min(tokenizer.vocab_size, TOKEN_ID_LIMIT)
    if tokenizer.learned_vocab_size > limit:
        raise TokenizedBuildError("tokenizer vocabulary does not fit uint16 token shards")
    excluded, applied = tombstoned_documents(data_store, build)

    with data_store.staging_directory("tokenized-") as staging:
        (staging / SHARD_DIR).mkdir(mode=0o700)
        (staging / INDEX_DIR).mkdir(mode=0o700)
        splits: dict[Split, SplitShards] = {}
        for split in SPLITS:
            writer = _SplitWriter(staging, split, max_shard_tokens)
            written = 0
            skipped = 0
            for document in iter_split_documents(data_store, build, split):
                if check_cancelled is not None:
                    check_cancelled()
                if document.document_id in excluded:
                    skipped += 1
                    continue
                content = tokenizer.encode(document.text)
                if any(not RESERVED_COUNT <= token < limit for token in content):
                    raise TokenizerArtifactError("encoding produced an ID outside the shard range")
                writer.add(document.document_id, [DOC_START_ID, *content, DOC_END_ID])
                written += 1
            writer.flush()
            splits[split] = SplitShards(
                split_sha256=split_sha256(build, split),
                document_count=written,
                excluded_document_count=skipped,
                token_count=writer.token_count,
                shards=writer.shards,
            )
        if check_cancelled is not None:
            check_cancelled()

        manifest = TokenizedManifest(
            tokenizer_id=tokenizer.tokenizer_id,
            tokenizer_json_sha256=tokenizer.manifest.tokenizer_json_sha256,
            reserved_spec_version=RESERVED_SPEC_VERSION,
            reserved_spec_sha256=RESERVED_SPEC_SHA256,
            vocab_size=tokenizer.vocab_size,
            dataset_id=build.dataset_id,
            build_id=build.build_id,
            max_shard_tokens=max_shard_tokens,
            splits=splits,
            total_tokens=sum(item.token_count for item in splits.values()),
            total_documents=sum(item.document_count for item in splits.values()),
            excluded_document_count=sum(item.excluded_document_count for item in splits.values()),
            applied_tombstone_ids=sorted(applied),
        )
        manifest_bytes = canonical_json_bytes(manifest)
        tokenized_id = sha256_bytes(manifest_bytes)
        _write_file(staging / MANIFEST_FILE, manifest_bytes)
        target = data_store.tokenized_path(tokenized_id)
        created = data_store.publish_directory(staging, target)

    if not created:
        # Same ID means same manifest; prove the stored shards still match it.
        verify_tokenized_build(data_store, tokenized_id)
    return TokenizedBuild(
        tokenized_id=tokenized_id, manifest=manifest, directory=target, created=created
    )


def _read_manifest(store: DataStore, tokenized_id: str) -> tuple[TokenizedManifest, Path]:
    directory = store.tokenized_path(tokenized_id)
    try:
        payload = (directory / MANIFEST_FILE).read_bytes()
    except FileNotFoundError as exc:
        raise TokenizedBuildError(f"tokenized build not found: {tokenized_id}") from exc
    if sha256_bytes(payload) != tokenized_id:
        raise TokenizedBuildError("tokenized manifest does not match its content address")
    try:
        manifest = TokenizedManifest.model_validate_json(payload)
    except ValidationError as exc:
        raise TokenizedBuildError("tokenized manifest is invalid") from exc
    if canonical_json_bytes(manifest) != payload:
        raise TokenizedBuildError("tokenized manifest is not in canonical form")
    if set(manifest.splits) != set(SPLITS):
        raise TokenizedBuildError("tokenized manifest must describe every split")
    return manifest, directory


def _read_index(directory: Path, record: ShardRecord) -> ShardIndex:
    path = directory / INDEX_DIR / f"{record.index_sha256}.json"
    try:
        payload = path.read_bytes()
    except FileNotFoundError as exc:
        raise TokenizedBuildError(f"shard index is missing: {record.index_sha256}") from exc
    if sha256_bytes(payload) != record.index_sha256:
        raise TokenizedBuildError(f"shard index does not match its hash: {record.index_sha256}")
    try:
        index = ShardIndex.model_validate_json(payload)
    except ValidationError as exc:
        raise TokenizedBuildError("shard index is invalid") from exc
    if canonical_json_bytes(index) != payload:
        raise TokenizedBuildError("shard index is not in canonical form")
    return index


def _read_shard(directory: Path, record: ShardRecord) -> array[int]:
    path = directory / SHARD_DIR / f"{record.sha256}{SHARD_SUFFIX}"
    try:
        payload = path.read_bytes()
    except FileNotFoundError as exc:
        raise TokenizedBuildError(f"token shard is missing: {record.sha256}") from exc
    if len(payload) != 2 * record.token_count or sha256_bytes(payload) != record.sha256:
        raise TokenizedBuildError(f"token shard does not match its hash: {record.sha256}")
    return _from_bytes(payload)


def _iter_split(
    store: DataStore, tokenized_id: str, split: Split
) -> Iterator[tuple[ShardRecord, ShardIndex, array[int]]]:
    manifest, directory = _read_manifest(store, tokenized_id)
    for ordinal, record in enumerate(manifest.splits[split].shards):
        index = _read_index(directory, record)
        if (
            index.split != split
            or index.ordinal != ordinal
            or index.shard_sha256 != record.sha256
            or index.token_count != record.token_count
            or len(index.entries) != record.document_count
        ):
            raise TokenizedBuildError("shard index disagrees with the manifest")
        yield record, index, _read_shard(directory, record)


def iter_tokenized_documents(
    store: DataStore | Path, tokenized_id: str, split: Split
) -> Iterator[tuple[str, list[int]]]:
    """Yield ``(document_id, content_tokens)`` in shard order, boundaries stripped.

    Continued documents are reassembled from their parts. Every shard and index
    read is hash-verified, but call ``verify_tokenized_build`` for the full
    structural check.
    """

    data_store = store if isinstance(store, DataStore) else DataStore(store)
    pending: tuple[str, list[int]] | None = None
    for _, index, tokens in _iter_split(data_store, tokenized_id, split):
        for entry in index.entries:
            span = tokens[entry.offset : entry.offset + entry.length].tolist()
            if entry.part == 0:
                if pending is not None:
                    raise TokenizedBuildError("document continuation is incomplete")
                pending = (entry.document_id, [])
            elif pending is None or pending[0] != entry.document_id:
                raise TokenizedBuildError("document continuation is out of order")
            pending[1].extend(span)
            if entry.part == entry.part_count - 1:
                document_id, framed = pending
                if framed[:1] != [DOC_START_ID] or framed[-1:] != [DOC_END_ID]:
                    raise TokenizedBuildError("document is missing its boundary tokens")
                yield document_id, framed[1:-1]
                pending = None
    if pending is not None:
        raise TokenizedBuildError("split ends inside a continued document")


def verify_tokenized_build(store: DataStore | Path, tokenized_id: str) -> TokenizedManifest:
    """Fully verify a stored tokenized build and return its manifest.

    Checks the manifest address and canonical form, every shard and index hash,
    index/manifest agreement, contiguous spans that cover each shard exactly,
    explicit and ordered continuations, token IDs below the declared
    vocabulary, and that reserved IDs appear only as document boundaries.
    """

    data_store = store if isinstance(store, DataStore) else DataStore(store)
    manifest, _ = _read_manifest(data_store, tokenized_id)
    if (
        manifest.reserved_spec_sha256 != RESERVED_SPEC_SHA256
        or manifest.reserved_spec_version != RESERVED_SPEC_VERSION
    ):
        raise TokenizedBuildError("tokenized build uses a different reserved-ID spec")
    for split in SPLITS:
        summary = manifest.splits[split]
        documents = 0
        tokens_total = 0
        open_part: tuple[str, int, int] | None = None
        for record, index, tokens in _iter_split(data_store, tokenized_id, split):
            tokens_total += len(tokens)
            if tokens and max(tokens) >= manifest.vocab_size:
                raise TokenizedBuildError("token shard holds an ID outside the vocabulary")
            boundaries = 0
            position = 0
            for entry in index.entries:
                if entry.offset != position or entry.part >= entry.part_count:
                    raise TokenizedBuildError("shard index spans are not contiguous")
                position += entry.length
                if entry.part == 0:
                    if open_part is not None:
                        raise TokenizedBuildError("document continuation is incomplete")
                elif open_part != (entry.document_id, entry.part - 1, entry.part_count):
                    raise TokenizedBuildError("document continuation is out of order")
                open_part = (entry.document_id, entry.part, entry.part_count)
                if entry.part == 0:
                    documents += 1
                    boundaries += 1
                    if tokens[entry.offset] != DOC_START_ID:
                        raise TokenizedBuildError("document span does not open with doc_start")
                if entry.part == entry.part_count - 1:
                    boundaries += 1
                    if tokens[entry.offset + entry.length - 1] != DOC_END_ID:
                        raise TokenizedBuildError("document span does not close with doc_end")
                    open_part = None
            if position != record.token_count:
                raise TokenizedBuildError("shard index does not cover the shard exactly")
            # Little-endian uint16: an ID below 256 has a zero high byte.
            if _to_bytes(tokens)[1::2].count(0) != boundaries:
                raise TokenizedBuildError("reserved IDs appear outside document boundaries")
        if open_part is not None:
            raise TokenizedBuildError("split ends inside a continued document")
        if documents != summary.document_count or tokens_total != summary.token_count:
            raise TokenizedBuildError("split totals disagree with the shards")
    if manifest.total_tokens != sum(item.token_count for item in manifest.splits.values()):
        raise TokenizedBuildError("manifest total tokens disagree with its splits")
    return manifest


def load_tokenized_manifest(store: DataStore | Path, tokenized_id: str) -> TokenizedManifest:
    """Read a tokenized manifest, checking only its address and canonical form."""

    data_store = store if isinstance(store, DataStore) else DataStore(store)
    return _read_manifest(data_store, tokenized_id)[0]


def affected_tokenized_builds(store: DataStore | Path, document_id: str) -> list[AffectedShards]:
    """Every stored tokenized build, split, and shard that contains a document.

    Used by removal handling: a tombstone resolves to document IDs, and each
    tokenized build listed here must be rebuilt (the rebuild excludes the
    tombstoned document) and its old shards retired.
    """

    data_store = store if isinstance(store, DataStore) else DataStore(store)
    affected: list[AffectedShards] = []
    for tokenized_id in data_store.tokenized_ids():
        manifest, directory = _read_manifest(data_store, tokenized_id)
        for split in SPLITS:
            hits = tuple(
                record.sha256
                for record in manifest.splits[split].shards
                if any(
                    entry.document_id == document_id
                    for entry in _read_index(directory, record).entries
                )
            )
            if hits:
                affected.append(
                    AffectedShards(
                        tokenized_id=tokenized_id,
                        dataset_id=manifest.dataset_id,
                        build_id=manifest.build_id,
                        tokenizer_id=manifest.tokenizer_id,
                        split=split,
                        shard_sha256s=hits,
                    )
                )
    return affected
