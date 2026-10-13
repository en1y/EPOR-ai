"""Deterministic sequence packing and token-based batching over token shards.

Packing
-------
Every document contributes its next-token pairs (see
:mod:`epor.training.shard_data`: ``[BOS, c1 .. cn, EOS]`` gives ``n + 1``
pairs).  Rows of ``seq_len`` positions are filled greedily, in document
order, by concatenating pair runs; a document that does not fit the space left
in a row continues at position 0 of the next row.  Each run is one contiguous
``segment_ids`` value (numbered ``0, 1, ..`` within its row), so the model's
document-causal attention never crosses documents.

* ``targets[r, t]`` is always the true next token: EOS where the document
  really ends, otherwise the following content token, *including* at a row
  boundary that cuts a document (that target is known, so it is not masked).
  For token shards only padding positions carry ``IGNORE_INDEX``; a chat
  corpus (:mod:`epor.training.chat_data`) also masks every non-assistant
  target, and ``target_tokens`` always counts the unmasked targets.
* **Known limitation:** a continued chunk starts a new segment, so RoPE
  positions restart at 0 and the chunk cannot attend to the document's earlier
  tokens.  No position offset is carried; long documents are trained as
  independent windows, the usual cost of fixed-length packing.

Training stream
---------------
``PackedBatchStream`` is an endless pass over the split.  Epoch ``e`` visits
documents in ``torch.randperm`` order drawn from a dedicated
``torch.Generator`` seeded with a hash of ``(seed, e)``, never the global RNG.
Packing runs straight across epoch boundaries, so a training row is never
padded and ``target_tokens == tokens`` for every training step.  Because the
order is a pure function of ``(seed, epoch)``, the exact resume state is the
cursor ``(epoch, position, offset)``; ``order_sha256`` pins the epoch order so
a PyTorch change to ``randperm`` is refused instead of silently reordering.

Token batching and ranks
------------------------
``tokens_per_batch`` is the *global* optimizer-step budget in positions and
must be a multiple of ``seq_len``: ``rows_per_step = tokens_per_batch //
seq_len``.  Every rank plans all ``rows_per_step`` rows (layout only, no token
reads), takes the contiguous block ``rows[rank * rows_per_rank : (rank + 1) *
rows_per_rank]``, and splits it into ``accumulation_steps`` micro-batches of
``micro_batch_size`` rows.  So the rank-ordered concatenation of all ranks'
micro-batches is exactly the single-process global batch, and the state is
identical on every rank.  Rank, world size, and micro-batch size are not part
of the fingerprint: they never change which rows form a step.

Loss normalization (for the training loop)
------------------------------------------
Normalize by real target tokens of the *global* step, never per micro-batch:
``step_loss = sum(cross_entropy_sum(micro)) / step.global_target_tokens`` over
all micro-batches of all ranks.  With the model's mean loss, a micro-batch's
sum is ``main_loss * micro.target_tokens`` (skip micro-batches whose
``target_tokens`` is 0: their mean is NaN).  Under gradient averaging across
``world_size`` ranks, scale each rank's contribution by ``world_size``.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Final, Literal

import torch
from pydantic import Field, model_validator
from torch import Tensor

from epor.data.models import FrozenSchema, Sha256
from epor.data.store import canonical_json_bytes, sha256_bytes
from epor.models.reference import IGNORE_INDEX
from epor.tokenizer.corpus import Split
from epor.tokenizer.spec import PAD_ID

from .shard_data import TrainingCorpus

PIPELINE_VERSION: Final = "epor-packing-v1"
STATE_SCHEMA_VERSION: Final = 1


class DataPipelineSettings(FrozenSchema):
    """Everything that decides which tokens form each step; record it in run manifests."""

    pipeline_version: Literal["epor-packing-v1"] = PIPELINE_VERSION
    tokenized_id: Sha256
    tokenizer_id: Sha256
    split: Split
    seq_len: int = Field(ge=1)
    tokens_per_batch: int = Field(ge=1)
    seed: int = Field(ge=0)
    shuffle: bool

    @model_validator(mode="after")
    def budget_is_whole_rows(self) -> DataPipelineSettings:
        if self.tokens_per_batch % self.seq_len:
            raise ValueError("tokens_per_batch must be a multiple of seq_len")
        return self

    @property
    def rows_per_step(self) -> int:
        return self.tokens_per_batch // self.seq_len

    @property
    def fingerprint(self) -> str:
        return sha256_bytes(canonical_json_bytes(self))

    @classmethod
    def for_corpus(
        cls,
        corpus: TrainingCorpus,
        *,
        seq_len: int,
        tokens_per_batch: int,
        seed: int,
        shuffle: bool,
    ) -> DataPipelineSettings:
        return cls(
            tokenized_id=corpus.tokenized_id,
            tokenizer_id=corpus.tokenizer_id,
            split=corpus.split,
            seq_len=seq_len,
            tokens_per_batch=tokens_per_batch,
            seed=seed,
            shuffle=shuffle,
        )


@dataclass(frozen=True, slots=True)
class BatchGeometry:
    """How one global step's rows divide over ranks and accumulation micro-batches."""

    rows_per_step: int
    world_size: int
    rank: int
    micro_batch_size: int

    def __post_init__(self) -> None:
        if self.world_size < 1 or not 0 <= self.rank < self.world_size:
            raise ValueError("rank must satisfy 0 <= rank < world_size")
        if self.rows_per_step % self.world_size:
            raise ValueError("tokens_per_batch // seq_len must be divisible by world_size")
        if self.micro_batch_size < 1 or self.rows_per_rank % self.micro_batch_size:
            raise ValueError("micro_batch_size must divide the rows each rank receives")

    @property
    def rows_per_rank(self) -> int:
        return self.rows_per_step // self.world_size

    @property
    def accumulation_steps(self) -> int:
        return self.rows_per_rank // self.micro_batch_size

    def rank_slice(self) -> slice:
        return slice(self.rank * self.rows_per_rank, (self.rank + 1) * self.rows_per_rank)


@dataclass(frozen=True, slots=True)
class RowSegment:
    """Pairs ``start .. start + length`` of corpus document ``document``."""

    document: int
    start: int
    length: int
    ends_document: bool


Row = tuple[RowSegment, ...]


@dataclass(frozen=True, slots=True)
class PackedBatch:
    """One micro-batch: ``[rows, seq_len]`` int64 tensors plus its layout."""

    input_ids: Tensor
    targets: Tensor
    segment_ids: Tensor
    layout: tuple[Row, ...]
    target_tokens: int

    @property
    def tokens(self) -> int:
        return self.input_ids.numel()

    @property
    def padding_tokens(self) -> int:
        return self.tokens - self.target_tokens


@dataclass(frozen=True, slots=True)
class PackedStep:
    """This rank's micro-batches for one optimizer step, with global token counts."""

    step: int
    micro_batches: tuple[PackedBatch, ...]
    global_tokens: int
    global_target_tokens: int

    @property
    def padding_fraction(self) -> float:
        return 1.0 - self.global_target_tokens / self.global_tokens


@dataclass(frozen=True, slots=True)
class _Cursor:
    epoch: int
    position: int
    offset: int


def _epoch_seed(seed: int, epoch: int) -> int:
    digest = hashlib.sha256(f"{PIPELINE_VERSION}:{seed}:{epoch}".encode()).digest()
    return int.from_bytes(digest[:8], "little") >> 1


class _Planner:
    """Row layouts from document lengths alone; reads no tokens."""

    def __init__(self, corpus: TrainingCorpus, seq_len: int, seed: int, shuffle: bool) -> None:
        if corpus.document_count == 0:
            raise ValueError(f"split {corpus.split!r} has no documents")
        self._lengths = corpus.pair_counts
        self._seq_len = seq_len
        self._seed = seed
        self._shuffle = shuffle
        self._orders: dict[int, list[int]] = {}

    def order(self, epoch: int) -> list[int]:
        cached = self._orders.get(epoch)
        if cached is None:
            count = len(self._lengths)
            if self._shuffle:
                generator = torch.Generator(device="cpu")
                generator.manual_seed(_epoch_seed(self._seed, epoch))
                cached = torch.randperm(count, generator=generator).tolist()
            else:
                cached = list(range(count))
            # A row spans at most two epochs only if seq_len is large; keep a few.
            if len(self._orders) >= 4:
                self._orders.pop(min(self._orders))
            self._orders[epoch] = cached
        return cached

    def order_sha256(self, epoch: int) -> str:
        return sha256_bytes(",".join(map(str, self.order(epoch))).encode())

    def document_length(self, cursor: _Cursor) -> int:
        return self._lengths[self.order(cursor.epoch)[cursor.position]]

    def plan_row(self, cursor: _Cursor, *, cycle: bool) -> tuple[Row, _Cursor]:
        count = len(self._lengths)
        epoch, position, offset = cursor.epoch, cursor.position, cursor.offset
        space = self._seq_len
        segments: list[RowSegment] = []
        while space and position < count:
            document = self.order(epoch)[position]
            take = min(self._lengths[document] - offset, space)
            ends = offset + take == self._lengths[document]
            segments.append(RowSegment(document, offset, take, ends))
            space -= take
            offset += take
            if ends:
                position, offset = position + 1, 0
                if position == count and cycle:
                    epoch, position = epoch + 1, 0
        return tuple(segments), _Cursor(epoch, position, offset)


def _materialize(corpus: TrainingCorpus, rows: Sequence[Row], seq_len: int) -> PackedBatch:
    shape = (len(rows), seq_len)
    inputs = torch.full(shape, PAD_ID, dtype=torch.int64)
    targets = torch.full(shape, IGNORE_INDEX, dtype=torch.int64)
    segments = torch.zeros(shape, dtype=torch.int64)
    for row_index, row in enumerate(rows):
        position = 0
        for segment_index, segment in enumerate(row):
            stop = position + segment.length
            row_inputs, row_targets = corpus.pairs(segment.document, segment.start, segment.length)
            inputs[row_index, position:stop] = row_inputs
            targets[row_index, position:stop] = row_targets
            segments[row_index, position:stop] = segment_index
            position = stop
        # Padding is its own run, after every document run of the row.
        segments[row_index, position:] = len(row)
    real = int((targets != IGNORE_INDEX).sum())
    return PackedBatch(
        input_ids=inputs,
        targets=targets,
        segment_ids=segments,
        layout=tuple(rows),
        target_tokens=real,
    )


def _check_context(corpus: TrainingCorpus, seq_len: int) -> None:
    if corpus.max_context is not None and seq_len > corpus.max_context:
        raise ValueError(
            f"seq_len {seq_len} exceeds the model's configured context {corpus.max_context}"
        )


def _build_step(
    corpus: TrainingCorpus,
    rows: list[Row],
    geometry: BatchGeometry,
    seq_len: int,
    step: int,
) -> PackedStep:
    mine = rows[geometry.rank_slice()]
    size = geometry.micro_batch_size
    micro = tuple(
        _materialize(corpus, mine[start : start + size], seq_len)
        for start in range(0, len(mine), size)
    )
    return PackedStep(
        step=step,
        micro_batches=micro,
        global_tokens=len(rows) * seq_len,
        global_target_tokens=sum(
            corpus.target_count(segment.document, segment.start, segment.length)
            for row in rows
            for segment in row
        ),
    )


def _counter(state: dict[str, Any], key: str) -> int:
    value = state.get(key)
    if type(value) is not int or value < 0:
        raise ValueError(f"checkpoint data cursor field {key!r} is invalid")
    return value


def _geometry(
    settings: DataPipelineSettings, rank: int, world_size: int, micro_batch_size: int | None
) -> BatchGeometry:
    if micro_batch_size is None:
        micro_batch_size = settings.rows_per_step // max(world_size, 1)
    return BatchGeometry(
        rows_per_step=settings.rows_per_step,
        world_size=world_size,
        rank=rank,
        micro_batch_size=micro_batch_size,
    )


class PackedBatchStream:
    """Endless, shuffled, exactly resumable training batches for one rank."""

    def __init__(
        self,
        corpus: TrainingCorpus,
        *,
        seq_len: int,
        tokens_per_batch: int,
        seed: int,
        rank: int = 0,
        world_size: int = 1,
        micro_batch_size: int | None = None,
    ) -> None:
        _check_context(corpus, seq_len)
        self.corpus = corpus
        self.settings = DataPipelineSettings.for_corpus(
            corpus, seq_len=seq_len, tokens_per_batch=tokens_per_batch, seed=seed, shuffle=True
        )
        self.geometry = _geometry(self.settings, rank, world_size, micro_batch_size)
        self._planner = _Planner(corpus, seq_len, seed, shuffle=True)
        self._cursor = _Cursor(0, 0, 0)
        self.step = 0

    @property
    def epoch(self) -> int:
        return self._cursor.epoch

    def next_step(self) -> PackedStep:
        rows: list[Row] = []
        cursor = self._cursor
        for _ in range(self.settings.rows_per_step):
            row, cursor = self._planner.plan_row(cursor, cycle=True)
            rows.append(row)
        packed = _build_step(self.corpus, rows, self.geometry, self.settings.seq_len, self.step)
        self._cursor = cursor
        self.step += 1
        return packed

    def __iter__(self) -> Iterator[PackedStep]:
        while True:
            yield self.next_step()

    def state_dict(self) -> dict[str, Any]:
        """Plain-JSON resume state, identical on every rank at a step boundary."""

        return {
            "schema_version": STATE_SCHEMA_VERSION,
            "fingerprint": self.settings.fingerprint,
            "settings": self.settings.model_dump(mode="json"),
            "step": self.step,
            "epoch": self._cursor.epoch,
            "position": self._cursor.position,
            "offset": self._cursor.offset,
            "order_sha256": self._planner.order_sha256(self._cursor.epoch),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("schema_version") != STATE_SCHEMA_VERSION:
            raise ValueError("unsupported data-pipeline state schema")
        if state.get("fingerprint") != self.settings.fingerprint:
            raise ValueError(
                "data-pipeline settings differ from the checkpoint "
                "(tokenized build, split, seq_len, tokens_per_batch, seed, or version)"
            )
        step, epoch, position, offset = (
            _counter(state, key) for key in ("step", "epoch", "position", "offset")
        )
        cursor = _Cursor(epoch, position, offset)
        if position >= self.corpus.document_count or offset >= self._planner.document_length(
            cursor
        ):
            raise ValueError("checkpoint data cursor is outside the corpus")
        if state.get("order_sha256") != self._planner.order_sha256(epoch):
            raise ValueError("document order for the checkpoint epoch cannot be reproduced")
        self._cursor = cursor
        self.step = step


def evaluation_steps(
    corpus: TrainingCorpus,
    *,
    seq_len: int,
    tokens_per_batch: int,
    rank: int = 0,
    world_size: int = 1,
    micro_batch_size: int | None = None,
    max_steps: int | None = None,
) -> Iterator[PackedStep]:
    """One unshuffled pass over a split (e.g. validation), padding only the tail.

    Ranks partition each step exactly like ``PackedBatchStream``; the final
    step's missing rows are all padding (``target_tokens == 0``).
    """

    _check_context(corpus, seq_len)
    settings = DataPipelineSettings.for_corpus(
        corpus, seq_len=seq_len, tokens_per_batch=tokens_per_batch, seed=0, shuffle=False
    )
    geometry = _geometry(settings, rank, world_size, micro_batch_size)
    if max_steps is not None and max_steps < 1:
        raise ValueError("max_steps must be positive when provided")
    planner = _Planner(corpus, seq_len, 0, shuffle=False)
    cursor = _Cursor(0, 0, 0)
    step = 0
    while cursor.position < corpus.document_count and (max_steps is None or step < max_steps):
        rows: list[Row] = []
        for _ in range(settings.rows_per_step):
            row, cursor = planner.plan_row(cursor, cycle=False)
            rows.append(row)
        yield _build_step(corpus, rows, geometry, seq_len, step)
        step += 1
