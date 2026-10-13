"""Content-addressed tokenizer artifacts and their strict layout contract.

An artifact is a directory ``<data_root>/tokenizers/<tokenizer_id>/`` holding
``tokenizer.json`` (the ``tokenizers`` library serialization, written as
canonical JSON) and ``manifest.json``. ``tokenizer_id`` is the SHA-256 of the
canonical manifest, and the manifest pins the SHA-256 of ``tokenizer.json`` and
of the reserved-ID spec, so any edit to either file, or any drift of the
reserved table, is detected on load. Files are written through the data store's
immutable writer: re-writing identical bytes is a no-op and different bytes at
the same address are refused.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final, Literal

from pydantic import Field, model_validator

from epor.data.models import FrozenSchema, Identifier, Sha256
from epor.data.store import DataStore, canonical_json_bytes, sha256_bytes

from .spec import (
    BYTE_COUNT,
    BYTE_OFFSET,
    FIRST_MERGE_ID,
    MIN_VOCAB_SIZE,
    RESERVED_SPEC_SHA256,
    RESERVED_SPEC_VERSION,
    RESERVED_TOKENS,
)

TOKENIZER_KIND: Final = "byte-level-bpe"
TOKENIZER_VERSION: Final = "epor-bpe-v1"
TOKENIZER_FILE: Final = "tokenizer.json"
MANIFEST_FILE: Final = "manifest.json"

# The exact pre-tokenizer every EPOR tokenizer uses. ``use_regex`` keeps the
# GPT-2 split pattern so merges never cross word/punctuation/whitespace-run
# boundaries; that also means a learned merge can never spell a reserved
# ``<|...|>`` string, because ``<|`` and ``|>`` always split from the name.
PRE_TOKENIZER: Final[dict[str, Any]] = {
    "type": "ByteLevel",
    "add_prefix_space": False,
    "trim_offsets": False,
    "use_regex": True,
}


class TokenizerArtifactError(RuntimeError):
    """A tokenizer artifact is missing, tampered with, or violates the layout."""


def _bytes_to_unicode() -> tuple[str, ...]:
    """The byte-level alphabet: the printable stand-in character for each byte."""

    printable = [
        *range(ord("!"), ord("~") + 1),
        *range(ord("¡"), ord("¬") + 1),
        *range(ord("®"), ord("ÿ") + 1),
    ]
    mapping: dict[int, int] = {byte: byte for byte in printable}
    extra = 0
    for byte in range(256):
        if byte not in mapping:
            mapping[byte] = 256 + extra
            extra += 1
    return tuple(chr(mapping[byte]) for byte in range(256))


BYTE_TO_UNICODE: Final = _bytes_to_unicode()
UNICODE_TO_BYTE: Final[dict[str, int]] = {char: byte for byte, char in enumerate(BYTE_TO_UNICODE)}


class TrainingProvenance(FrozenSchema):
    dataset_id: Identifier
    build_id: Sha256
    split: Literal["train"] = "train"
    split_sha256: Sha256
    document_count: int = Field(ge=1)
    byte_count: int = Field(ge=0)


class TrainerSettings(FrozenSchema):
    library: Literal["tokenizers"] = "tokenizers"
    library_version: str = Field(min_length=1, max_length=40)
    model: Literal["BPE"] = "BPE"
    pre_tokenizer: dict[str, Any] = Field(default_factory=lambda: dict(PRE_TOKENIZER))
    normalizer: None = None
    initial_alphabet: Literal["byte-level-256"] = "byte-level-256"
    min_frequency: int = Field(ge=1)
    parallelism: Literal["single-threaded"] = "single-threaded"


class TokenizerManifest(FrozenSchema):
    schema_version: Literal[1] = 1
    kind: Literal["byte-level-bpe"] = TOKENIZER_KIND
    tokenizer_version: Literal["epor-bpe-v1"] = TOKENIZER_VERSION
    reserved_spec_version: str
    reserved_spec_sha256: Sha256
    vocab_size: int = Field(ge=MIN_VOCAB_SIZE, description="Declared (target) vocabulary size.")
    learned_vocab_size: int = Field(
        ge=FIRST_MERGE_ID,
        description="Reserved + byte + merges actually learned; may be below vocab_size.",
    )
    merge_count: int = Field(ge=0)
    training: TrainingProvenance
    trainer: TrainerSettings
    tokenizer_json_sha256: Sha256

    @model_validator(mode="after")
    def sizes_are_honest(self) -> TokenizerManifest:
        if self.learned_vocab_size > self.vocab_size:
            raise ValueError("learned vocabulary cannot exceed the declared vocabulary size")
        if self.merge_count != self.learned_vocab_size - FIRST_MERGE_ID:
            raise ValueError("merge count must equal the learned IDs above the byte block")
        return self


def manifest_id(manifest: TokenizerManifest) -> str:
    return sha256_bytes(canonical_json_bytes(manifest))


def merge_pair(entry: Any) -> tuple[str, str]:
    if isinstance(entry, str):
        left, separator, right = entry.partition(" ")
        if not separator:
            raise TokenizerArtifactError("malformed merge entry")
        return left, right
    if isinstance(entry, list) and len(entry) == 2 and all(isinstance(x, str) for x in entry):
        return entry[0], entry[1]
    raise TokenizerArtifactError("malformed merge entry")


def validate_layout(document: dict[str, Any], *, vocab_size: int) -> int:
    """Check a ``tokenizer.json`` document against the fixed layout.

    Returns the learned vocabulary size. Enforces: no normalizer or post-processor,
    the exact byte-level pre-tokenizer, the reserved table at IDs 0-255 as
    special added tokens, byte ``b`` at ID ``256 + b``, and merge ``i`` producing
    ID ``512 + i`` with contiguous IDs and no extra tokens.
    """

    if document.get("normalizer") is not None:
        raise TokenizerArtifactError("tokenizer must not normalize input")
    if document.get("post_processor") is not None:
        raise TokenizerArtifactError("tokenizer must not insert tokens implicitly")
    if document.get("pre_tokenizer") != PRE_TOKENIZER:
        raise TokenizerArtifactError("tokenizer pre-tokenizer differs from the EPOR contract")
    model = document.get("model") or {}
    if (
        model.get("type") != "BPE"
        or model.get("dropout") is not None
        or model.get("unk_token") is not None
        or model.get("continuing_subword_prefix") is not None
        or model.get("end_of_word_suffix") is not None
    ):
        raise TokenizerArtifactError("tokenizer model must be deterministic byte-level BPE")

    added = document.get("added_tokens") or []
    expected_added = [(item.id, item.content) for item in RESERVED_TOKENS]
    if [(item.get("id"), item.get("content")) for item in added] != expected_added or not all(
        item.get("special") is True and item.get("normalized") is False for item in added
    ):
        raise TokenizerArtifactError("reserved special tokens drifted from the spec")

    vocab: dict[str, int] = model.get("vocab") or {}
    merges = [merge_pair(entry) for entry in model.get("merges") or []]
    learned = len(vocab)
    if learned > vocab_size or learned != FIRST_MERGE_ID + len(merges):
        raise TokenizerArtifactError("vocabulary size does not match merges and layout")
    by_id = {token_id: token for token, token_id in vocab.items()}
    if sorted(by_id) != list(range(learned)):
        raise TokenizerArtifactError("vocabulary IDs must be unique and contiguous")
    for item in RESERVED_TOKENS:
        if by_id[item.id] != item.content:
            raise TokenizerArtifactError("reserved ID block drifted from the spec")
    for byte in range(BYTE_COUNT):
        if by_id[BYTE_OFFSET + byte] != BYTE_TO_UNICODE[byte]:
            raise TokenizerArtifactError("byte block must map byte b to ID 256 + b")
    for index, (left, right) in enumerate(merges):
        if by_id[FIRST_MERGE_ID + index] != left + right:
            raise TokenizerArtifactError("merge order does not match learned IDs")
        if (
            vocab.get(left, learned) >= FIRST_MERGE_ID + index
            or vocab.get(right, learned) >= FIRST_MERGE_ID + index
        ):
            raise TokenizerArtifactError("merge references a token learned later")
        if vocab[left] < BYTE_OFFSET or vocab[right] < BYTE_OFFSET:
            raise TokenizerArtifactError("merges must never involve reserved tokens")
    return learned


def write_artifact(
    store: DataStore,
    manifest: TokenizerManifest,
    tokenizer_json: bytes,
) -> tuple[str, Path]:
    """Write ``tokenizer.json`` then the committing manifest; idempotent."""

    if sha256_bytes(tokenizer_json) != manifest.tokenizer_json_sha256:
        raise TokenizerArtifactError("manifest does not pin these tokenizer bytes")
    tokenizer_id = manifest_id(manifest)
    store.write_tokenizer_file(tokenizer_id, TOKENIZER_FILE, tokenizer_json)
    store.write_tokenizer_file(tokenizer_id, MANIFEST_FILE, canonical_json_bytes(manifest))
    return tokenizer_id, store.tokenizer_path(tokenizer_id)


def read_artifact(directory: Path) -> tuple[str, TokenizerManifest, bytes]:
    """Read and fully verify an artifact directory; refuse anything tampered."""

    try:
        manifest_bytes = (directory / MANIFEST_FILE).read_bytes()
        tokenizer_json = (directory / TOKENIZER_FILE).read_bytes()
    except FileNotFoundError as exc:
        raise TokenizerArtifactError(f"tokenizer artifact is incomplete: {directory}") from exc
    tokenizer_id = sha256_bytes(manifest_bytes)
    if directory.name != tokenizer_id:
        raise TokenizerArtifactError("tokenizer manifest does not match its content address")
    try:
        manifest = TokenizerManifest.model_validate_json(manifest_bytes)
    except ValueError as exc:
        raise TokenizerArtifactError("tokenizer manifest is invalid") from exc
    if canonical_json_bytes(manifest) != manifest_bytes:
        raise TokenizerArtifactError("tokenizer manifest is not in canonical form")
    if sha256_bytes(tokenizer_json) != manifest.tokenizer_json_sha256:
        raise TokenizerArtifactError("tokenizer.json does not match its pinned hash")
    if (
        manifest.reserved_spec_sha256 != RESERVED_SPEC_SHA256
        or manifest.reserved_spec_version != RESERVED_SPEC_VERSION
    ):
        raise TokenizerArtifactError("tokenizer was built for a different reserved-ID spec")
    try:
        document = json.loads(tokenizer_json)
    except ValueError as exc:
        raise TokenizerArtifactError("tokenizer.json is not valid JSON") from exc
    learned = validate_layout(document, vocab_size=manifest.vocab_size)
    if learned != manifest.learned_vocab_size:
        raise TokenizerArtifactError("tokenizer.json size disagrees with its manifest")
    return tokenizer_id, manifest, tokenizer_json
