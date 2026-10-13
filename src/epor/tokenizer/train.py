"""Deterministic training of the EPOR byte-level BPE from a data-engine build.

Training reads only the build's ``train`` split (validation and test text never
reaches the trainer), learns merges from scratch with the ``tokenizers``
library, then rewrites the result into the fixed layout from ``spec``: reserved
IDs 0-255, byte ``b`` at ``256 + b``, and merge ``i`` at ``512 + i``. The library
sorts its initial alphabet by character rather than by byte, so the rewrite is
what pins the byte block; merges are stored as token strings and are unaffected.

Determinism: the BPE trainer breaks count ties by pair order, and word counting
merges per-thread maps whose result does not depend on scheduling, so repeated
runs produced byte-identical ``tokenizer.json`` in both threaded and
single-threaded modes in our checks. Training nevertheless runs with library
parallelism disabled (``TOKENIZERS_PARALLELISM=false``) so reproducibility does
not rest on an undocumented property of the parallel path; the setting is
recorded in the manifest. Tokenizer training is a one-off, so the cost is
acceptable.

Removal tombstones recorded after the build are honored by refusal: if any
train-split document is now tombstoned, training stops and the dataset must be
rebuilt first, so a tokenizer's recorded ``split_sha256`` always names exactly
the text it learned from.

If the corpus supports fewer merges than ``vocab_size`` asks for, the manifest
records the smaller ``learned_vocab_size`` next to the declared size. Nothing
is padded and the declared size is never claimed as learned.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tokenizers
from tokenizers import AddedToken, Tokenizer, decoders, models, pre_tokenizers, trainers

from epor.data.store import DataStore, canonical_json_bytes, sha256_bytes

from .artifact import (
    BYTE_TO_UNICODE,
    PRE_TOKENIZER,
    TokenizerArtifactError,
    TokenizerManifest,
    TrainerSettings,
    TrainingProvenance,
    merge_pair,
    validate_layout,
    write_artifact,
)
from .corpus import (
    CorpusError,
    iter_split_documents,
    load_build,
    split_sha256,
    tombstoned_documents,
)
from .spec import (
    BYTE_COUNT,
    BYTE_OFFSET,
    DEFAULT_VOCAB_SIZE,
    FIRST_MERGE_ID,
    MIN_VOCAB_SIZE,
    RESERVED_SPEC_SHA256,
    RESERVED_SPEC_VERSION,
    RESERVED_TOKENS,
)

DEFAULT_MIN_FREQUENCY = 2
_PARALLELISM_ENV = "TOKENIZERS_PARALLELISM"


@dataclass(frozen=True, slots=True)
class TrainedTokenizer:
    tokenizer_id: str
    manifest: TokenizerManifest
    directory: Path


@contextmanager
def _single_threaded() -> Iterator[None]:
    previous = os.environ.get(_PARALLELISM_ENV)
    os.environ[_PARALLELISM_ENV] = "false"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(_PARALLELISM_ENV, None)
        else:
            os.environ[_PARALLELISM_ENV] = previous


def _added_token(content: str) -> AddedToken:
    return AddedToken(
        content, single_word=False, lstrip=False, rstrip=False, normalized=False, special=True
    )


def _canonical_layout(document: dict[str, Any]) -> dict[str, Any]:
    """Renumber a freshly trained tokenizer into the fixed EPOR layout."""

    model = document["model"]
    merges = [merge_pair(entry) for entry in model["merges"]]
    vocab: dict[str, int] = {item.content: item.id for item in RESERVED_TOKENS}
    for byte in range(BYTE_COUNT):
        vocab[BYTE_TO_UNICODE[byte]] = BYTE_OFFSET + byte
    for index, (left, right) in enumerate(merges):
        token = left + right
        if token in vocab:
            raise TokenizerArtifactError("a learned merge collides with an existing token")
        vocab[token] = FIRST_MERGE_ID + index
    if set(vocab) != set(model["vocab"]):
        raise TokenizerArtifactError("trained vocabulary is outside the reserved/byte/merge layout")
    model["vocab"] = vocab
    model["merges"] = [[left, right] for left, right in merges]
    document["added_tokens"] = [
        {
            "id": item.id,
            "content": item.content,
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": True,
        }
        for item in RESERVED_TOKENS
    ]
    document["normalizer"] = None
    document["post_processor"] = None
    document["padding"] = None
    document["truncation"] = None
    return document


def train_tokenizer(
    store: DataStore | Path,
    dataset_id: str,
    build_id: str,
    *,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    min_frequency: int = DEFAULT_MIN_FREQUENCY,
    check_cancelled: Callable[[], None] | None = None,
) -> TrainedTokenizer:
    """Train on one build's train split and store an immutable artifact."""

    if vocab_size < MIN_VOCAB_SIZE:
        raise ValueError(f"vocab_size must be at least {MIN_VOCAB_SIZE}")
    if min_frequency < 1:
        raise ValueError("min_frequency must be at least 1")
    data_store = store if isinstance(store, DataStore) else DataStore(store)
    build = load_build(data_store, dataset_id, build_id)
    train_split_sha256 = split_sha256(build, "train")
    if not any(item.split == "train" for item in build.documents):
        raise CorpusError("build has no training documents")
    # The manifest pins the whole train split by hash, so training cannot
    # silently drop a document removed after the build. Refuse instead: the
    # operator rebuilds the dataset, which applies the tombstone, and trains on
    # that build.
    removed, _ = tombstoned_documents(data_store, build)
    train_ids = {item.document_id for item in build.documents if item.split == "train"}
    if removed & train_ids:
        raise CorpusError(
            "build has training documents removed since it was made; rebuild the dataset "
            "before training a tokenizer"
        )

    stats = {"documents": 0, "bytes": 0}

    def texts() -> Iterator[str]:
        for document in iter_split_documents(data_store, build, "train"):
            if check_cancelled is not None:
                check_cancelled()
            stats["documents"] += 1
            stats["bytes"] += len(document.text.encode("utf-8"))
            yield document.text

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=PRE_TOKENIZER["add_prefix_space"],
        trim_offsets=PRE_TOKENIZER["trim_offsets"],
        use_regex=PRE_TOKENIZER["use_regex"],
    )
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        show_progress=False,
        special_tokens=[_added_token(item.content) for item in RESERVED_TOKENS],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    with _single_threaded():
        tokenizer.train_from_iterator(texts(), trainer=trainer)
    if check_cancelled is not None:
        check_cancelled()

    document = _canonical_layout(json.loads(tokenizer.to_str()))
    learned_vocab_size = validate_layout(document, vocab_size=vocab_size)
    tokenizer_json = canonical_json_bytes(document)
    # The rewritten document must still load in the library it came from.
    Tokenizer.from_str(tokenizer_json.decode("utf-8"))

    manifest = TokenizerManifest(
        reserved_spec_version=RESERVED_SPEC_VERSION,
        reserved_spec_sha256=RESERVED_SPEC_SHA256,
        vocab_size=vocab_size,
        learned_vocab_size=learned_vocab_size,
        merge_count=learned_vocab_size - FIRST_MERGE_ID,
        training=TrainingProvenance(
            dataset_id=build.dataset_id,
            build_id=build.build_id,
            split_sha256=train_split_sha256,
            document_count=stats["documents"],
            byte_count=stats["bytes"],
        ),
        trainer=TrainerSettings(
            library_version=tokenizers.__version__,
            min_frequency=min_frequency,
        ),
        tokenizer_json_sha256=sha256_bytes(tokenizer_json),
    )
    tokenizer_id, directory = write_artifact(data_store, manifest, tokenizer_json)
    return TrainedTokenizer(tokenizer_id=tokenizer_id, manifest=manifest, directory=directory)
