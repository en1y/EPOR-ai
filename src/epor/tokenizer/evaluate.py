"""Tokenizer fertility measured on one held-out split of a data build.

``evaluate_tokenizer`` encodes every document of a split (``validation`` by
default; never ``train`` unless asked, and an in-sample run is flagged) and
reports, overall and grouped by the build's domain and language labels:
documents, UTF-8 bytes, characters, whitespace-delimited words, content tokens
(document boundary tokens are not counted), tokens/byte, tokens/word,
bytes/token, the byte-fallback rate (share of tokens that are raw byte tokens,
IDs 256-511), and the number of documents whose tokens do not decode back to
the exact source bytes (expected to be 0).

Documents covered by a current tombstone are excluded with the same rule the
token shards use. Ratios are rounded to six decimals so the report is
deterministic canonical JSON. Reports are written immutably to
``<data_root>/evaluations/<report_id>.json``, where ``report_id`` is the SHA-256
of the canonical report; identical inputs reproduce the same report.

The numbers describe the given corpus only. ``scope`` says so in every report:
a fixture or proxy-scale corpus measurement is not a target-scale result.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, get_args

from pydantic import Field, ValidationError

from epor.data.models import FrozenSchema, Identifier, Sha256
from epor.data.store import DataStore, DataStoreError, canonical_json_bytes, sha256_bytes

from .codec import load_tokenizer
from .corpus import (
    CorpusError,
    Split,
    iter_split_documents,
    load_build,
    split_sha256,
    tombstoned_documents,
)
from .spec import BYTE_COUNT, BYTE_OFFSET, RESERVED_SPEC_SHA256

RATIO_DECIMALS: Final = 6
MeasurementScope = Literal[
    "Measured on the named build split only. Values describe this corpus and are not "
    "target-scale results; fixture or proxy-scale corpora do not predict full-scale fertility."
]
MEASUREMENT_SCOPE: Final[MeasurementScope] = get_args(MeasurementScope)[0]


class FertilityMetrics(FrozenSchema):
    documents: int = Field(ge=0)
    bytes: int = Field(ge=0)
    characters: int = Field(ge=0)
    words: int = Field(ge=0)
    tokens: int = Field(ge=0)
    byte_fallback_tokens: int = Field(ge=0)
    tokens_per_byte: float | None
    tokens_per_word: float | None
    bytes_per_token: float | None
    byte_fallback_rate: float | None


class FertilityReport(FrozenSchema):
    schema_version: Literal[1] = 1
    kind: Literal["tokenizer-fertility"] = "tokenizer-fertility"
    scope: MeasurementScope = MEASUREMENT_SCOPE
    tokenizer_id: Sha256
    tokenizer_json_sha256: Sha256
    reserved_spec_sha256: Sha256
    vocab_size: int = Field(ge=1)
    learned_vocab_size: int = Field(ge=1)
    dataset_id: Identifier
    build_id: Sha256
    split: Split
    split_sha256: Sha256
    in_sample: bool = Field(
        description="True when the split is the train split this tokenizer was trained on."
    )
    excluded_document_count: int = Field(ge=0)
    applied_tombstone_ids: list[Sha256]
    round_trip_failures: int = Field(ge=0)
    overall: FertilityMetrics
    by_domain: dict[str, FertilityMetrics]
    by_language: dict[str, FertilityMetrics]

    @property
    def report_id(self) -> str:
        return sha256_bytes(canonical_json_bytes(self))


@dataclass(slots=True)
class _Tally:
    documents: int = 0
    bytes: int = 0
    characters: int = 0
    words: int = 0
    tokens: int = 0
    byte_fallback_tokens: int = 0

    def add(self, other: _Tally) -> None:
        self.documents += other.documents
        self.bytes += other.bytes
        self.characters += other.characters
        self.words += other.words
        self.tokens += other.tokens
        self.byte_fallback_tokens += other.byte_fallback_tokens

    def metrics(self) -> FertilityMetrics:
        def ratio(numerator: int, denominator: int) -> float | None:
            return round(numerator / denominator, RATIO_DECIMALS) if denominator else None

        return FertilityMetrics(
            documents=self.documents,
            bytes=self.bytes,
            characters=self.characters,
            words=self.words,
            tokens=self.tokens,
            byte_fallback_tokens=self.byte_fallback_tokens,
            tokens_per_byte=ratio(self.tokens, self.bytes),
            tokens_per_word=ratio(self.tokens, self.words),
            bytes_per_token=ratio(self.bytes, self.tokens),
            byte_fallback_rate=ratio(self.byte_fallback_tokens, self.tokens),
        )


def _grouped(tallies: dict[str, _Tally]) -> dict[str, FertilityMetrics]:
    return {key: tallies[key].metrics() for key in sorted(tallies)}


def evaluate_tokenizer(
    store: DataStore | Path,
    tokenizer_id: str,
    dataset_id: str,
    build_id: str,
    *,
    split: Split = "validation",
    check_cancelled: Callable[[], None] | None = None,
) -> FertilityReport:
    """Measure fertility on one split and store the report immutably."""

    data_store = store if isinstance(store, DataStore) else DataStore(store)
    build = load_build(data_store, dataset_id, build_id)
    tokenizer = load_tokenizer(data_store, tokenizer_id)
    excluded, applied = tombstoned_documents(data_store, build)
    labels = {item.document_id: item for item in build.documents}

    overall = _Tally()
    by_domain: defaultdict[str, _Tally] = defaultdict(_Tally)
    by_language: defaultdict[str, _Tally] = defaultdict(_Tally)
    failures = 0
    skipped = 0
    for document in iter_split_documents(data_store, build, split):
        if check_cancelled is not None:
            check_cancelled()
        if document.document_id in excluded:
            skipped += 1
            continue
        encoded = document.text.encode("utf-8")
        ids = tokenizer.encode(document.text)
        if tokenizer.decode_bytes(ids, skip_special_tokens=False) != encoded:
            failures += 1
        tally = _Tally(
            documents=1,
            bytes=len(encoded),
            characters=len(document.text),
            words=len(document.text.split()),
            tokens=len(ids),
            byte_fallback_tokens=sum(
                1 for token in ids if BYTE_OFFSET <= token < BYTE_OFFSET + BYTE_COUNT
            ),
        )
        overall.add(tally)
        label = labels[document.document_id]
        by_domain[label.domain].add(tally)
        by_language[label.language].add(tally)
    if overall.documents == 0:
        raise CorpusError(f"build {build_id} has no {split} documents to evaluate")

    report = FertilityReport(
        tokenizer_id=tokenizer.tokenizer_id,
        tokenizer_json_sha256=tokenizer.manifest.tokenizer_json_sha256,
        reserved_spec_sha256=RESERVED_SPEC_SHA256,
        vocab_size=tokenizer.vocab_size,
        learned_vocab_size=tokenizer.learned_vocab_size,
        dataset_id=build.dataset_id,
        build_id=build.build_id,
        split=split,
        split_sha256=split_sha256(build, split),
        in_sample=split == "train" and tokenizer.manifest.training.build_id == build.build_id,
        excluded_document_count=skipped,
        applied_tombstone_ids=sorted(applied),
        round_trip_failures=failures,
        overall=overall.metrics(),
        by_domain=_grouped(by_domain),
        by_language=_grouped(by_language),
    )
    data_store.write_evaluation(canonical_json_bytes(report))
    return report


def load_fertility_report(store: DataStore | Path, report_id: str) -> FertilityReport:
    """Read a stored report and verify it against its content address."""

    data_store = store if isinstance(store, DataStore) else DataStore(store)
    try:
        payload = data_store.evaluation_path(report_id).read_bytes()
    except FileNotFoundError as exc:
        raise DataStoreError(f"evaluation report not found: {report_id}") from exc
    if sha256_bytes(payload) != report_id:
        raise DataStoreError("evaluation report does not match its content address")
    try:
        report = FertilityReport.model_validate_json(payload)
    except ValidationError as exc:
        raise DataStoreError("evaluation report is invalid") from exc
    if canonical_json_bytes(report) != payload:
        raise DataStoreError("evaluation report is not in canonical form")
    return report
