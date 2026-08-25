"""Offline, deterministic data utilities for reference-model smoke training."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from epor.models.tokenizer import DebugByteTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FIXTURE_CORPUS = PROJECT_ROOT / "fixtures" / "tiny_corpus.txt"


@dataclass(frozen=True, slots=True)
class TokenCorpus:
    """A content-addressed, in-memory token corpus used only for tiny runs."""

    path: Path
    token_ids: Tensor
    sha256: str

    def __post_init__(self) -> None:
        if self.token_ids.ndim != 1:
            raise ValueError("corpus token_ids must be one-dimensional")
        if self.token_ids.dtype != torch.int64:
            raise TypeError("corpus token_ids must use torch.int64")
        if self.token_ids.numel() < 2:
            raise ValueError("corpus must contain at least two token IDs")


def sha256_file(path: str | Path) -> str:
    """Hash a file without loading it all into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_token_corpus(
    path: str | Path = DEFAULT_FIXTURE_CORPUS,
    *,
    tokenizer: DebugByteTokenizer | None = None,
) -> TokenCorpus:
    """Load bytes through the reversible debug tokenizer, entirely offline."""

    corpus_path = Path(path).resolve()
    data = corpus_path.read_bytes()
    active_tokenizer = tokenizer or DebugByteTokenizer()
    ids = active_tokenizer.encode(data, add_bos=True, add_eos=True)
    return TokenCorpus(
        path=corpus_path,
        token_ids=torch.tensor(ids, dtype=torch.int64),
        sha256=hashlib.sha256(data).hexdigest(),
    )


class RandomBatchStream:
    """Deterministic random windows with an explicit serializable cursor/RNG."""

    def __init__(
        self,
        corpus: TokenCorpus,
        *,
        batch_size: int,
        sequence_length: int,
        seed: int,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if sequence_length < 1:
            raise ValueError("sequence_length must be positive")
        if seed < 0:
            raise ValueError("data seed must be non-negative")
        if corpus.token_ids.numel() <= sequence_length:
            raise ValueError("corpus must contain more tokens than the requested sequence length")
        self.corpus = corpus
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(seed)
        self.cursor = 0

    def next_batch(self) -> tuple[Tensor, Tensor]:
        """Return next-token input/target windows and advance exactly once."""

        max_start = self.corpus.token_ids.numel() - self.sequence_length - 1
        starts = torch.randint(
            0,
            max_start + 1,
            (self.batch_size,),
            generator=self.generator,
        )
        inputs = torch.stack(
            [
                self.corpus.token_ids[start : start + self.sequence_length]
                for start in starts.tolist()
            ]
        )
        targets = torch.stack(
            [
                self.corpus.token_ids[start + 1 : start + self.sequence_length + 1]
                for start in starts.tolist()
            ]
        )
        self.cursor += 1
        return inputs, targets

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "1",
            "cursor": self.cursor,
            "generator_state": self.generator.get_state().clone(),
            "corpus_sha256": self.corpus.sha256,
            "batch_size": self.batch_size,
            "sequence_length": self.sequence_length,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("schema_version") != "1":
            raise ValueError("unsupported data-stream state schema")
        if state.get("corpus_sha256") != self.corpus.sha256:
            raise ValueError("resume corpus hash does not match the checkpoint")
        if state.get("batch_size") != self.batch_size:
            raise ValueError("resume batch size does not match the checkpoint")
        if state.get("sequence_length") != self.sequence_length:
            raise ValueError("resume sequence length does not match the checkpoint")
        cursor = state.get("cursor")
        generator_state = state.get("generator_state")
        if not isinstance(cursor, int) or cursor < 0:
            raise ValueError("checkpoint data cursor is invalid")
        if not isinstance(generator_state, Tensor):
            raise ValueError("checkpoint data generator state is invalid")
        self.generator.set_state(generator_state.cpu())
        self.cursor = cursor


def sequential_batches(
    corpus: TokenCorpus,
    *,
    batch_size: int,
    sequence_length: int,
    max_batches: int | None = None,
) -> Iterator[tuple[Tensor, Tensor]]:
    """Yield deterministic non-overlapping validation windows."""

    if batch_size < 1 or sequence_length < 1:
        raise ValueError("batch_size and sequence_length must be positive")
    if max_batches is not None and max_batches < 1:
        raise ValueError("max_batches must be positive when provided")
    starts = list(range(0, corpus.token_ids.numel() - sequence_length, sequence_length))
    for batches_emitted, offset in enumerate(range(0, len(starts), batch_size)):
        if max_batches is not None and batches_emitted >= max_batches:
            break
        selected = starts[offset : offset + batch_size]
        inputs = torch.stack(
            [corpus.token_ids[start : start + sequence_length] for start in selected]
        )
        targets = torch.stack(
            [corpus.token_ids[start + 1 : start + sequence_length + 1] for start in selected]
        )
        yield inputs, targets
