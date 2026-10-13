"""Read-only, verified access to one split of a v0.0.4 tokenized build.

Framing decision
----------------
Shards frame every document as ``<|doc_start|>`` (15) + content +
``<|doc_end|>`` (16).  Those IDs are *storage* framing.  The model sees the
same document framed with its own ``bos_token_id`` / ``eos_token_id``
(``<|bos|>`` 1 / ``<|eos|>`` 2), which ``ShardCorpus.open`` requires to equal
the frozen reserved spec.  A document of ``n`` content tokens therefore
becomes the ``n + 2`` model tokens ``[BOS, c1 .. cn, EOS]`` and yields
``n + 1`` *next-token pairs*: pair ``i`` has input ``framed[i]`` and target
``framed[i + 1]``.  BOS is only ever an input, EOS only ever a target, and the
first content token is learned from BOS alone.  Because the stored framed
document has the same length as the model-framed one, the mapping is the
position-preserving substitution 15 -> BOS, 16 -> EOS, which is safe because
``verify_tokenized_build`` proves reserved IDs occur only as boundaries.

Reading
-------
Shards are memory-mapped privately with ``torch.from_file`` (little-endian
``uint16``; big-endian hosts are refused rather than silently misread).
Multi-part documents are reassembled from their consecutive index entries, so
a document is addressed by its index in the split's stable shard order no
matter how many shards it spans.  ``open`` runs the full
``verify_tokenized_build`` by default; the maps are created afterwards, so a
file swapped *after* verification is outside this check (same as any reader).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from epor.data.store import DataStore
from epor.models.config import ModelConfig
from epor.tokenizer.corpus import Split
from epor.tokenizer.shards import (
    SHARD_DIR,
    SHARD_SUFFIX,
    TokenizedBuildError,
    TokenizedManifest,
    _read_index,  # pyright: ignore[reportPrivateUsage]
    load_tokenized_manifest,
    verify_tokenized_build,
)
from epor.tokenizer.spec import BOS_ID, DOC_END_ID, DOC_START_ID, EOS_ID, PAD_ID


@dataclass(frozen=True, slots=True)
class _Piece:
    shard: int
    offset: int
    length: int


class ShardCorpus:
    """Documents of one split of a verified tokenized build, as next-token pairs."""

    def __init__(
        self,
        *,
        tokenized_id: str,
        manifest: TokenizedManifest,
        split: Split,
        document_ids: tuple[str, ...],
        pieces: tuple[tuple[_Piece, ...], ...],
        shards: tuple[Tensor, ...],
        max_context: int | None,
    ) -> None:
        self.tokenized_id = tokenized_id
        self.manifest = manifest
        self.split: Split = split
        self.document_ids = document_ids
        self._pieces = pieces
        self._shards = shards
        self.max_context = max_context
        # Stored framed length minus one: the number of next-token pairs.
        self.pair_counts: tuple[int, ...] = tuple(
            sum(piece.length for piece in doc) - 1 for doc in pieces
        )

    @property
    def tokenizer_id(self) -> str:
        return self.manifest.tokenizer_id

    @property
    def vocab_size(self) -> int:
        return self.manifest.vocab_size

    @property
    def document_count(self) -> int:
        return len(self._pieces)

    @property
    def pair_count(self) -> int:
        """Total trainable (input, target) pairs in the split."""

        return sum(self.pair_counts)

    @classmethod
    def open(
        cls,
        store: DataStore | Path,
        tokenized_id: str,
        split: Split,
        *,
        model_config: ModelConfig | None = None,
        tokenizer_id: str | None = None,
        verify: bool = True,
    ) -> ShardCorpus:
        """Open a split, refusing unverified builds and model/tokenizer mismatches.

        With ``model_config`` the build's vocabulary must equal the model's and
        the model's pad/BOS/EOS IDs must be the reserved spec's, so every token
        the pipeline emits is a valid embedding row with the intended meaning.
        """

        if sys.byteorder != "little":
            raise TokenizedBuildError("token shards are little-endian; this host is not")
        data_store = store if isinstance(store, DataStore) else DataStore(store)
        manifest = (
            verify_tokenized_build(data_store, tokenized_id)
            if verify
            else load_tokenized_manifest(data_store, tokenized_id)
        )
        if tokenizer_id is not None and manifest.tokenizer_id != tokenizer_id:
            raise TokenizedBuildError("tokenized build was made with a different tokenizer")
        if model_config is not None:
            if manifest.vocab_size != model_config.vocab_size:
                raise TokenizedBuildError(
                    f"tokenizer vocabulary {manifest.vocab_size} does not match the model "
                    f"vocab_size {model_config.vocab_size}"
                )
            special = (
                model_config.pad_token_id,
                model_config.bos_token_id,
                model_config.eos_token_id,
            )
            if special != (PAD_ID, BOS_ID, EOS_ID):
                raise TokenizedBuildError(
                    "model pad/bos/eos token IDs do not match the reserved tokenizer spec"
                )
        directory = data_store.tokenized_path(tokenized_id)
        records = manifest.splits[split].shards
        shards: list[Tensor] = []
        document_ids: list[str] = []
        pieces: list[tuple[_Piece, ...]] = []
        pending: list[_Piece] = []
        for ordinal, record in enumerate(records):
            path = directory / SHARD_DIR / f"{record.sha256}{SHARD_SUFFIX}"
            if not path.is_file() or path.stat().st_size != 2 * record.token_count:
                raise TokenizedBuildError(f"token shard is missing or truncated: {record.sha256}")
            shards.append(
                torch.from_file(
                    str(path), shared=False, size=record.token_count, dtype=torch.uint16
                )
            )
            for entry in _read_index(directory, record).entries:
                if entry.part == 0:
                    document_ids.append(entry.document_id)
                pending.append(_Piece(ordinal, entry.offset, entry.length))
                if entry.part == entry.part_count - 1:
                    pieces.append(tuple(pending))
                    pending = []
        if pending:
            raise TokenizedBuildError("split ends inside a continued document")
        return cls(
            tokenized_id=tokenized_id,
            manifest=manifest,
            split=split,
            document_ids=tuple(document_ids),
            pieces=tuple(pieces),
            shards=tuple(shards),
            max_context=None if model_config is None else model_config.configured_max_context,
        )

    def framed_tokens(self, document: int, start: int = 0, stop: int | None = None) -> Tensor:
        """Model-framed tokens ``[BOS, content.., EOS][start:stop]`` as int64."""

        total = self.pair_counts[document] + 1
        stop = total if stop is None else stop
        if not 0 <= start <= stop <= total:
            raise IndexError("framed token range is outside the document")
        parts: list[Tensor] = []
        base = 0
        for piece in self._pieces[document]:
            low = max(start, base)
            high = min(stop, base + piece.length)
            if low < high:
                begin = piece.offset + low - base
                parts.append(self._shards[piece.shard][begin : begin + high - low])
            base += piece.length
        tokens = torch.cat(parts).to(torch.int64) if parts else torch.empty(0, dtype=torch.int64)
        tokens[tokens == DOC_START_ID] = BOS_ID
        tokens[tokens == DOC_END_ID] = EOS_ID
        return tokens

    def pairs(self, document: int, start: int, length: int) -> tuple[Tensor, Tensor]:
        """Next-token pairs ``start .. start + length`` of a document: (inputs, targets)."""

        if length < 1 or start + length > self.pair_counts[document]:
            raise IndexError("pair range is outside the document")
        framed = self.framed_tokens(document, start, start + length + 1)
        return framed[:-1], framed[1:]
