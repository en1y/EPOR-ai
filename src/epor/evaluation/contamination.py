"""N-gram contamination checks between evaluation items and training data.

What is measured
----------------
For every evaluation item, the share of its units covered by an n-gram that
also occurs in a training document, at two levels:

``text``
    Units are words of the *data-engine build* split text: NFKC, case-folded,
    ``\\w+`` runs (punctuation and whitespace dropped, so case, spacing, and
    punctuation edits do not hide an overlap).  Default ``n = 13``
    **[EPOR-adaptation]** of the common 13-gram word-overlap practice.
``token``
    Units are token IDs of the *tokenized build* shards; item text is encoded
    with the build's own tokenizer.  Default ``n = 13`` **[EPOR-adaptation]**
    (inside the 8-13 token range used in published practice).  Item text is
    encoded standalone, so tokens at its edges can differ from the same words
    in context; interior n-grams still match.

An item with fewer than ``n`` units but at least ``min_ngram`` units is
represented by one whole-item n-gram; shorter items are reported as
``assessed: false`` rather than as clean.  ``coverage`` is the fraction of an
item's units covered by matched n-grams; an item is *flagged* when coverage on
the train split reaches ``threshold`` (default ``0.5``, **[hypothesis]** — a
mid-point between "any collision" and "near-verbatim", to be calibrated with
human review of flagged items).  ``any_match`` (at least one collision) is
reported too.  Validation and test splits can be scanned as well, to show
where an item sits in the build (``<level>.<split>.*`` metrics).

``split-leakage`` mode instead indexes the build's validation and test
documents and measures how much of each is covered by train-split n-grams.

Limits
------
* **A lower bound.** Only verbatim n-gram reuse after normalization is
  detected: paraphrases, translations, reformatted equations, and
  semantically equivalent rewrites are not.
* N-grams are 64-bit truncated BLAKE2b hashes (stable across processes,
  unlike Python's ``hash()``); a collision can produce a rare false match.
* Memory is bounded by the evaluation items (or held-out documents in
  split-leakage mode, capped by ``max_index_ngrams``) plus one training
  document at a time; training data is streamed in its stable split order.

Reports never contain item or training text: only item IDs, SHA-256 hashes of
item text, unit/n-gram counts, matched item offsets, and (hashed) training
document IDs with unit offsets.
"""

from __future__ import annotations

import hashlib
import math
import re
import struct
import unicodedata
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from epor.data.models import Identifier
from epor.data.store import DataStore, sha256_bytes

from .report import ItemResult, MeasurementScope, Metric, TaskResult
from .safety_behaviour import SafetyPreferenceItem, SafetyRubricItem
from .tasks import CodeItem, ExactMatchItem, MathItem, MultipleChoiceItem

ContaminationLevel = Literal["text", "token"]
SplitName = Literal["train", "validation", "test"]
DEFAULT_TEXT_NGRAM = 13
DEFAULT_TOKEN_NGRAM = 13
DEFAULT_MIN_NGRAM = 8
DEFAULT_THRESHOLD = 0.5
DEFAULT_MAX_INDEX_NGRAMS = 2_000_000
CANCEL_POLL_DOCUMENTS = 64
MAX_RECORDED_MATCHES = 3
MAX_RECORDED_OFFSETS = 32
_WORD = re.compile(r"\w+")
_HASH_PERSON = b"epor-ngram-v1"

Unit = str | int


class ContaminationError(RuntimeError):
    """A contamination scan cannot run as configured (e.g. the index is too large)."""


class ContaminationSettings(BaseModel):
    """Contamination task options (YAML key ``contamination`` on a suite task)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: Literal["benchmark-items", "split-leakage"] = "benchmark-items"
    targets: list[Identifier] | None = None
    levels: list[ContaminationLevel] = Field(default_factory=lambda: ["text", "token"])
    splits: list[SplitName] = Field(default_factory=lambda: ["train"])
    text_ngram: int = Field(default=DEFAULT_TEXT_NGRAM, ge=2, le=64)
    token_ngram: int = Field(default=DEFAULT_TOKEN_NGRAM, ge=2, le=64)
    min_ngram: int = Field(default=DEFAULT_MIN_NGRAM, ge=1, le=64)
    threshold: float = Field(default=DEFAULT_THRESHOLD, gt=0.0, le=1.0)
    max_index_ngrams: int = Field(default=DEFAULT_MAX_INDEX_NGRAMS, ge=1)

    @model_validator(mode="after")
    def consistent(self) -> ContaminationSettings:
        if not self.levels or len(set(self.levels)) != len(self.levels):
            raise ValueError("contamination levels must be non-empty and unique")
        if "train" not in self.splits or len(set(self.splits)) != len(self.splits):
            raise ValueError("contamination splits must be unique and include train")
        if self.min_ngram > min(self.text_ngram, self.token_ngram):
            raise ValueError("min_ngram cannot exceed the n-gram sizes")
        if self.mode == "split-leakage" and self.targets is not None:
            raise ValueError("split-leakage mode scans the build itself and takes no targets")
        return self

    def ngram(self, level: ContaminationLevel) -> int:
        return self.text_ngram if level == "text" else self.token_ngram


# ----------------------------------------------------------------------------- units


def text_units(text: str) -> list[str]:
    """Normalized word units: NFKC, case-folded ``\\w+`` runs."""

    return _WORD.findall(unicodedata.normalize("NFKC", text).casefold())


def ngram_hash(units: Sequence[Unit]) -> int:
    """Stable 64-bit BLAKE2b hash of a word or token-ID n-gram."""

    if units and isinstance(units[0], int):
        payload = b"t" + struct.pack(f"<{len(units)}I", *units)
    else:
        payload = b"w" + "\x1f".join(str(unit) for unit in units).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8, person=_HASH_PERSON).digest()
    return int.from_bytes(digest, "little")


@dataclass(frozen=True, slots=True)
class ContaminationItem:
    """An evaluation item's identity and text (the text never reaches a report)."""

    item_id: str
    task_id: str
    text: str

    @property
    def text_sha256(self) -> str:
        return sha256_bytes(self.text.encode("utf-8"))


ScannableItem = (
    MultipleChoiceItem
    | ExactMatchItem
    | MathItem
    | CodeItem
    | SafetyPreferenceItem
    | SafetyRubricItem
)


def item_text(item: ScannableItem) -> str:
    """All text an item shows a model or a grader, joined by newlines."""

    if isinstance(item, MultipleChoiceItem):
        parts = [item.question, *item.choices]
    elif isinstance(item, ExactMatchItem):
        parts = [item.prompt, *item.answers]
    elif isinstance(item, MathItem):
        parts = [item.problem, item.answer]
    elif isinstance(item, CodeItem):
        parts = [item.prompt, item.test]
    elif isinstance(item, SafetyPreferenceItem):
        parts = [item.context, item.safe, item.unsafe]
    else:
        parts = [item.prompt]
    return "\n".join(parts)


def contamination_items(task_id: str, items: Iterable[ScannableItem]) -> list[ContaminationItem]:
    """Wrap one task's items, naming each ``<task_id>/<item_id>``."""

    return [
        ContaminationItem(item_id=f"{task_id}/{item.id}", task_id=task_id, text=item_text(item))
        for item in items
    ]


@dataclass(slots=True)
class _SplitState:
    covered: set[int] = field(default_factory=set)
    matched_starts: set[int] = field(default_factory=set)
    matches: list[tuple[str, int]] = field(default_factory=list)


@dataclass(slots=True)
class _ItemUnits:
    units: int
    ngram: int | None  # None when the item is too short to assess
    total_ngrams: int
    splits: dict[str, _SplitState] = field(default_factory=dict)


class NgramIndex:
    """Hashed n-grams of a bounded set of targets, probed by streaming documents.

    Targets are small (evaluation items or held-out documents); the training
    side is never materialized.  A first-unit filter skips hashing windows that
    cannot start any indexed n-gram.
    """

    def __init__(self, *, max_ngrams: int) -> None:
        self.max_ngrams = max_ngrams
        self._grams: dict[int, dict[int, list[tuple[int, int]]]] = {}
        self._first: dict[int, set[Unit]] = {}
        self.size = 0

    def add(self, target: int, units: Sequence[Unit], n: int) -> int:
        """Index every ``n``-gram of ``units`` for ``target``; returns how many."""

        count = len(units) - n + 1
        if count <= 0:
            return 0
        if self.size + count > self.max_ngrams:
            raise ContaminationError(
                f"contamination index would exceed {self.max_ngrams} n-grams; "
                "raise max_index_ngrams or scan fewer targets"
            )
        by_hash = self._grams.setdefault(n, {})
        first = self._first.setdefault(n, set())
        for start in range(count):
            window = units[start : start + n]
            by_hash.setdefault(ngram_hash(window), []).append((target, start))
            first.add(window[0])
        self.size += count
        return count

    def probe(self, units: Sequence[Unit]) -> Iterator[tuple[int, int, int, int]]:
        """Yield ``(target, target_start, n, document_offset)`` for every hit."""

        for n, by_hash in self._grams.items():
            first = self._first[n]
            for offset in range(len(units) - n + 1):
                if units[offset] not in first:
                    continue
                hits = by_hash.get(ngram_hash(units[offset : offset + n]))
                if hits:
                    for target, start in hits:
                        yield target, start, n, offset


@dataclass(frozen=True, slots=True)
class TrainingCorpus:
    """Streams of a build's split documents at each level, as ``(document_id, units)``."""

    text: Callable[[SplitName], Iterable[tuple[str, Sequence[Unit]]]] | None
    token: Callable[[SplitName], Iterable[tuple[str, Sequence[Unit]]]] | None
    encode: Callable[[str], list[int]] | None
    tokenized_id: str | None = None
    build_id: str | None = None

    def stream(self, level: ContaminationLevel) -> Callable[[SplitName], Iterable[Any]]:
        source = self.text if level == "text" else self.token
        if source is None:
            raise ContaminationError(f"no {level}-level training data is available")
        return source

    def units(self, level: ContaminationLevel, text: str) -> list[Unit]:
        if level == "text":
            return list(text_units(text))
        if self.encode is None:
            raise ContaminationError("token-level checks need the build's tokenizer")
        return list(self.encode(text))


def build_corpus(data_root: str | Path, tokenized_id: str) -> TrainingCorpus:
    """Text (data-engine build) and token (shards) streams of a tokenized build."""

    from epor.tokenizer import (
        iter_split_documents,
        iter_tokenized_documents,
        load_build,
        load_tokenizer,
    )
    from epor.tokenizer.corpus import tombstoned_documents
    from epor.tokenizer.shards import load_tokenized_manifest

    store = DataStore(Path(data_root).expanduser().resolve())
    manifest = load_tokenized_manifest(store, tokenized_id)
    build = load_build(store, manifest.dataset_id, manifest.build_id)
    tokenizer = load_tokenizer(store, manifest.tokenizer_id)
    # Shards exclude tombstoned documents; the text level applies the current
    # tombstones the same way, so both levels describe the same documents.
    excluded, _ = tombstoned_documents(store, build)

    def text(split: SplitName) -> Iterator[tuple[str, list[str]]]:
        for document in iter_split_documents(store, build, split):
            if document.document_id not in excluded:
                yield document.document_id, text_units(document.text)

    def token(split: SplitName) -> Iterator[tuple[str, list[int]]]:
        yield from iter_tokenized_documents(store, tokenized_id, split)

    return TrainingCorpus(
        text=text,
        token=token,
        encode=tokenizer.encode,
        tokenized_id=tokenized_id,
        build_id=manifest.build_id,
    )


def _cancel_check(should_cancel: Callable[[], bool] | None, counter: int) -> None:
    if should_cancel is not None and counter % CANCEL_POLL_DOCUMENTS == 0 and should_cancel():
        from .suites import EvaluationCancelledError

        raise EvaluationCancelledError("evaluation cancellation requested")


def _scan(
    index: NgramIndex,
    states: Sequence[_ItemUnits],
    documents: Iterable[tuple[str, Sequence[Unit]]],
    split: str,
    should_cancel: Callable[[], bool] | None,
) -> int:
    scanned = 0
    for counter, (document_id, units) in enumerate(documents):
        _cancel_check(should_cancel, counter)
        scanned += 1
        for target, start, n, offset in index.probe(units):
            state = states[target].splits[split]
            if start not in state.matched_starts:
                state.matched_starts.add(start)
                state.covered.update(range(start, start + n))
            if len(state.matches) < MAX_RECORDED_MATCHES and all(
                document_id != seen for seen, _ in state.matches
            ):
                state.matches.append((document_id, offset))
    return scanned


def _mean(values: Sequence[float]) -> float | None:
    return math.fsum(values) / len(values) if values else None


def _level_details(state: _ItemUnits, splits: Sequence[str]) -> dict[str, Any]:
    details: dict[str, Any] = {
        "units": state.units,
        "ngram": state.ngram,
        "assessed": state.ngram is not None,
        "total_ngrams": state.total_ngrams,
    }
    for split in splits:
        split_state = state.splits[split]
        coverage = len(split_state.covered) / state.units if state.ngram is not None else None
        details[split] = {
            "coverage": coverage,
            "matched_ngrams": len(split_state.matched_starts),
            "item_offsets": sorted(split_state.matched_starts)[:MAX_RECORDED_OFFSETS],
            "matches": [
                {"document_id": document_id, "unit_offset": offset}
                for document_id, offset in split_state.matches
            ],
        }
    return details


def _new_state(units: int, n: int, min_ngram: int, splits: Sequence[str]) -> _ItemUnits:
    effective = n if units >= n else (units if units >= min_ngram else None)
    total = units - effective + 1 if effective is not None else 0
    return _ItemUnits(
        units=units,
        ngram=effective,
        total_ngrams=total,
        splits={split: _SplitState() for split in splits},
    )


def scan_contamination(
    task_id: str,
    items: Sequence[ContaminationItem],
    corpus: TrainingCorpus,
    *,
    scope: MeasurementScope,
    settings: ContaminationSettings | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> TaskResult:
    """Per-item n-gram coverage of ``items`` by the corpus splits, at each level."""

    config = settings or ContaminationSettings()
    if config.mode != "benchmark-items":
        raise ValueError("scan_contamination runs benchmark-items mode")
    ids = [item.item_id for item in items]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("contamination needs at least one item and unique item IDs")
    level_states: dict[str, list[_ItemUnits]] = {}
    documents_scanned: dict[str, int] = {}
    for level in config.levels:
        stream = corpus.stream(level)
        index = NgramIndex(max_ngrams=config.max_index_ngrams)
        states: list[_ItemUnits] = []
        for position, item in enumerate(items):
            units = corpus.units(level, item.text)
            state = _new_state(len(units), config.ngram(level), config.min_ngram, config.splits)
            if state.ngram is not None:
                index.add(position, units, state.ngram)
            states.append(state)
        for split in config.splits:
            documents_scanned[f"{level}.{split}"] = _scan(
                index, states, stream(split), split, should_cancel
            )
        level_states[level] = states

    results: list[ItemResult] = []
    flagged_union: list[float] = []
    for position, item in enumerate(items):
        flagged = False
        any_assessed = False
        details: dict[str, Any] = {"task_id": item.task_id, "item_sha256": item.text_sha256}
        for level in config.levels:
            state = level_states[level][position]
            level_detail = _level_details(state, config.splits)
            train_coverage = level_detail["train"]["coverage"]
            level_flagged = train_coverage is not None and train_coverage >= config.threshold
            level_detail["flagged"] = level_flagged
            details[level] = level_detail
            flagged = flagged or level_flagged
            any_assessed = any_assessed or state.ngram is not None
        details["flagged"] = flagged
        details["assessed"] = any_assessed
        if any_assessed:
            flagged_union.append(float(flagged))
        results.append(ItemResult(item_id=item.item_id, score=float(flagged), details=details))

    metrics = [
        Metric(
            name="flagged_rate", value=_mean(flagged_union), scope=scope, higher_is_better=False
        ),
        Metric(
            name="flagged_items",
            value=float(sum(flagged_union)),
            scope=scope,
            higher_is_better=False,
        ),
        Metric(
            name="assessed_items",
            value=float(len(flagged_union)),
            scope=scope,
            higher_is_better=True,
        ),
        Metric(name="items", value=float(len(items)), scope=scope, higher_is_better=True),
    ]
    for level in config.levels:
        for split in config.splits:
            coverages: list[float] = []
            for result in results:
                coverage = result.details[level][split]["coverage"]
                if coverage is not None:
                    coverages.append(coverage)
            prefix = f"{level}.{split}"
            metrics += [
                Metric(
                    name=f"{prefix}.flagged_rate",
                    value=_mean([float(c >= config.threshold) for c in coverages]),
                    scope=scope,
                    higher_is_better=False,
                ),
                Metric(
                    name=f"{prefix}.any_match_rate",
                    value=_mean([float(c > 0.0) for c in coverages]),
                    scope=scope,
                    higher_is_better=False,
                ),
                Metric(
                    name=f"{prefix}.mean_coverage",
                    value=_mean(coverages),
                    scope=scope,
                    higher_is_better=False,
                ),
            ]
    return TaskResult(
        task_id=task_id,
        task_type="contamination",
        format="ngram-contamination-v1",
        primary_metric="flagged_rate",
        metrics=metrics,
        items=results,
        settings=_settings_record(config, corpus, documents_scanned),
    )


def scan_split_leakage(
    task_id: str,
    corpus: TrainingCorpus,
    *,
    scope: MeasurementScope,
    settings: ContaminationSettings | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> TaskResult:
    """Coverage of each validation/test document by train-split n-grams.

    Held-out documents are indexed (bounded by ``max_index_ngrams``) and the
    train split is streamed once per level.  Item IDs are
    ``<split>:<document_id>`` (document IDs are content hashes).
    """

    config = settings or ContaminationSettings(mode="split-leakage")
    held_out: list[SplitName] = ["validation", "test"]
    level_states: dict[str, list[_ItemUnits]] = {}
    keys: list[str] = []
    documents_scanned: dict[str, int] = {}
    for level in config.levels:
        stream = corpus.stream(level)
        index = NgramIndex(max_ngrams=config.max_index_ngrams)
        states: list[_ItemUnits] = []
        level_keys: list[str] = []
        for split in held_out:
            for counter, (document_id, units) in enumerate(stream(split)):
                _cancel_check(should_cancel, counter)
                state = _new_state(len(units), config.ngram(level), config.min_ngram, ["train"])
                if state.ngram is not None:
                    index.add(len(states), units, state.ngram)
                states.append(state)
                level_keys.append(f"{split}:{document_id}")
        if keys and keys != level_keys:
            raise ContaminationError(
                "text and token levels list different held-out documents; rebuild the "
                "tokenized build so it reflects the current tombstones"
            )
        keys = level_keys
        documents_scanned[f"{level}.train"] = _scan(
            index, states, stream("train"), "train", should_cancel
        )
        level_states[level] = states

    results: list[ItemResult] = []
    flagged_union: list[float] = []
    for position, key in enumerate(keys):
        flagged = False
        any_assessed = False
        details: dict[str, Any] = {"split": key.split(":", 1)[0]}
        for level in config.levels:
            state = level_states[level][position]
            level_detail = _level_details(state, ["train"])
            coverage = level_detail["train"]["coverage"]
            level_detail["flagged"] = coverage is not None and coverage >= config.threshold
            details[level] = level_detail
            flagged = flagged or level_detail["flagged"]
            any_assessed = any_assessed or state.ngram is not None
        details["flagged"] = flagged
        details["assessed"] = any_assessed
        if any_assessed:
            flagged_union.append(float(flagged))
        results.append(ItemResult(item_id=key, score=float(flagged), details=details))
    metrics = [
        Metric(
            name="flagged_rate", value=_mean(flagged_union), scope=scope, higher_is_better=False
        ),
        Metric(
            name="flagged_items",
            value=float(sum(flagged_union)),
            scope=scope,
            higher_is_better=False,
        ),
        Metric(
            name="assessed_items",
            value=float(len(flagged_union)),
            scope=scope,
            higher_is_better=True,
        ),
        Metric(name="items", value=float(len(keys)), scope=scope, higher_is_better=True),
    ]
    for level in config.levels:
        coverages = [c for r in results if (c := r.details[level]["train"]["coverage"]) is not None]
        metrics.append(
            Metric(
                name=f"{level}.mean_coverage",
                value=_mean(coverages),
                scope=scope,
                higher_is_better=False,
            )
        )
    return TaskResult(
        task_id=task_id,
        task_type="contamination",
        format="split-leakage-v1",
        primary_metric="flagged_rate",
        metrics=metrics,
        items=results,
        settings=_settings_record(config, corpus, documents_scanned),
    )


def _settings_record(
    config: ContaminationSettings, corpus: TrainingCorpus, scanned: dict[str, int]
) -> dict[str, Any]:
    return {
        **config.model_dump(mode="json"),
        "hash": "blake2b-64",
        "text_units": "nfkc-casefold-word",
        "tokenized_id": corpus.tokenized_id,
        "build_id": corpus.build_id,
        "documents_scanned": scanned,
        "lower_bound": True,
        "limitations": (
            "Detects verbatim n-gram reuse after normalization only; paraphrased, "
            "translated, or reformatted contamination is not detected."
        ),
        "threshold_label": "hypothesis",
        "ngram_label": "EPOR-adaptation",
    }


__all__ = [
    "DEFAULT_MIN_NGRAM",
    "DEFAULT_TEXT_NGRAM",
    "DEFAULT_THRESHOLD",
    "DEFAULT_TOKEN_NGRAM",
    "ContaminationError",
    "ContaminationItem",
    "ContaminationLevel",
    "ContaminationSettings",
    "NgramIndex",
    "ScannableItem",
    "TrainingCorpus",
    "build_corpus",
    "contamination_items",
    "item_text",
    "ngram_hash",
    "scan_contamination",
    "scan_split_leakage",
    "text_units",
]
