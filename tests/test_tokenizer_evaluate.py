"""Held-out fertility reports: hand-checkable, grouped, deterministic, immutable."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from conftest import (
    TOKENIZER_FIXTURE_VOCAB,
    TOKENIZER_SPLIT_SALT,
    TOKENIZER_TRAIN_TEXTS,
    group_for_split,
    tokenizer_registration,
)

from epor.data.models import BuildRequest, DatasetBuild, SplitRatios
from epor.data.service import DataEngine
from epor.data.store import DataStore, DataStoreError, canonical_json_bytes, sha256_bytes
from epor.tokenizer import (
    CorpusError,
    FertilityReport,
    evaluate_tokenizer,
    load_fertility_report,
    load_tokenizer,
    train_tokenizer,
)
from epor.tokenizer.evaluate import MEASUREMENT_SCOPE

# Validation documents with known sizes. Each source carries registration hints,
# so the build assigns the domain and language labels the report groups by.
HELD_OUT: dict[str, tuple[str, str, str]] = {
    # source id: (text, domain hint, language hint)
    "held-prose": ("the archive records each change", "general", "en"),
    "held-code": ("def tokenize(value):\n    return value", "code", "en"),
    "held-german": ("Grüße aus Köln über café", "general", "de"),
}


@dataclass(frozen=True)
class Fixture:
    data: Path
    build: DatasetBuild
    tokenizer_id: str


def _make(root: Path) -> Fixture:
    engine = DataEngine(root / "data")
    inputs = root / "inputs"
    inputs.mkdir(parents=True)
    train_paths = []
    for name, text in TOKENIZER_TRAIN_TEXTS.items():
        path = inputs / name
        path.write_text(text, encoding="utf-8")
        train_paths.append(path)
    engine.ingest(tokenizer_registration("train-corpus", group_for_split("train")), train_paths)
    for source_id, (text, domain, language) in HELD_OUT.items():
        path = inputs / f"{source_id}.txt"
        path.write_text(text, encoding="utf-8")
        registration = tokenizer_registration(source_id, group_for_split("validation"))
        registration = registration.model_copy(
            update={"domain_hint": domain, "language_hint": language}
        )
        engine.ingest(registration, [path])
    build, _ = engine.build(
        BuildRequest(
            dataset_id="fertility-fixture",
            split_ratios=SplitRatios(train=50, validation=50, test=0),
            split_salt=TOKENIZER_SPLIT_SALT,
            intended_uses=["tokenizer tests"],
            prohibited_uses=["production deployment"],
        )
    )
    trained = train_tokenizer(
        root / "data", build.dataset_id, build.build_id, vocab_size=TOKENIZER_FIXTURE_VOCAB
    )
    return Fixture(root / "data", build, trained.tokenizer_id)


@pytest.fixture(scope="module")
def shared(tmp_path_factory: pytest.TempPathFactory) -> Fixture:
    return _make(tmp_path_factory.mktemp("fertility"))


@pytest.fixture
def fixture(shared: Fixture, tmp_path: Path) -> Fixture:
    shutil.copytree(shared.data, tmp_path / "data")
    return Fixture(tmp_path / "data", shared.build, shared.tokenizer_id)


def _evaluate(fixture: Fixture, **kwargs: Any) -> FertilityReport:
    return evaluate_tokenizer(
        fixture.data,
        fixture.tokenizer_id,
        fixture.build.dataset_id,
        fixture.build.build_id,
        **kwargs,
    )


def _expected(texts: list[str], fixture: Fixture) -> dict[str, object]:
    tokenizer = load_tokenizer(fixture.data, fixture.tokenizer_id)
    ids = [token for text in texts for token in tokenizer.encode(text)]
    size = sum(len(text.encode("utf-8")) for text in texts)
    words = sum(len(text.split()) for text in texts)
    fallback = sum(1 for token in ids if 256 <= token < 512)
    return {
        "documents": len(texts),
        "bytes": size,
        "characters": sum(len(text) for text in texts),
        "words": words,
        "tokens": len(ids),
        "byte_fallback_tokens": fallback,
        "tokens_per_byte": round(len(ids) / size, 6),
        "tokens_per_word": round(len(ids) / words, 6),
        "bytes_per_token": round(size / len(ids), 6),
        "byte_fallback_rate": round(fallback / len(ids), 6),
    }


def test_held_out_fixture_is_in_the_validation_split(fixture: Fixture) -> None:
    held = {item.source_id: item for item in fixture.build.documents if item.split == "validation"}
    assert set(held) == set(HELD_OUT)
    for source_id, (_, domain, language) in HELD_OUT.items():
        assert (held[source_id].domain, held[source_id].language) == (domain, language)


def test_default_split_is_validation_with_hand_checkable_numbers(fixture: Fixture) -> None:
    report = _evaluate(fixture)
    assert report.split == "validation" and not report.in_sample
    assert report.split_sha256 == fixture.build.artifacts["splits/validation.jsonl"]
    assert report.round_trip_failures == 0
    assert report.scope == MEASUREMENT_SCOPE

    overall = report.overall
    # Counted by hand from HELD_OUT: the German text has 24 characters, five of
    # them ("ü", "ß", "ö", "ü", "é") two UTF-8 bytes each, so 29 bytes.
    assert overall.documents == 3
    assert overall.bytes == 31 + 37 + 29
    assert overall.characters == 31 + 37 + 24
    assert overall.words == 5 + 4 + 5
    texts = [text for text, _, _ in HELD_OUT.values()]
    assert overall.model_dump() == _expected(texts, fixture)
    assert overall.tokens_per_byte is not None and 0 < overall.tokens_per_byte <= 1


def test_metrics_are_grouped_by_domain_and_language(fixture: Fixture) -> None:
    report = _evaluate(fixture)
    by_domain = {
        "code": [HELD_OUT["held-code"][0]],
        "general": [HELD_OUT["held-prose"][0], HELD_OUT["held-german"][0]],
    }
    by_language = {
        "de": [HELD_OUT["held-german"][0]],
        "en": [HELD_OUT["held-prose"][0], HELD_OUT["held-code"][0]],
    }
    assert list(report.by_domain) == sorted(by_domain)
    assert list(report.by_language) == sorted(by_language)
    for label, texts in by_domain.items():
        assert report.by_domain[label].model_dump() == _expected(texts, fixture)
    for label, texts in by_language.items():
        assert report.by_language[label].model_dump() == _expected(texts, fixture)


def test_report_is_deterministic_immutable_and_verifiable(fixture: Fixture) -> None:
    first = _evaluate(fixture)
    again = _evaluate(fixture)
    assert first == again
    path = DataStore(fixture.data).evaluation_path(first.report_id)
    payload = path.read_bytes()
    assert payload == canonical_json_bytes(first)
    assert sha256_bytes(payload) == first.report_id
    assert load_fertility_report(fixture.data, first.report_id) == first

    path.write_bytes(payload.replace(b'"round_trip_failures":0', b'"round_trip_failures":1'))
    with pytest.raises(DataStoreError, match="content address"):
        load_fertility_report(fixture.data, first.report_id)


def test_train_split_is_explicit_and_flagged_in_sample(fixture: Fixture) -> None:
    report = _evaluate(fixture, split="train")
    assert report.split == "train" and report.in_sample
    assert report.overall.documents == len(TOKENIZER_TRAIN_TEXTS)
    assert report.round_trip_failures == 0
    assert report.report_id != _evaluate(fixture).report_id


def test_empty_split_is_refused(fixture: Fixture) -> None:
    with pytest.raises(CorpusError, match="no test documents"):
        _evaluate(fixture, split="test")


def test_tombstoned_documents_are_excluded(fixture: Fixture) -> None:
    tombstone = DataEngine(fixture.data).remove(
        target_kind="source_id",
        target="held-code",
        reason="Fixture removal request.",
        requested_by="tests",
    )
    report = _evaluate(fixture)
    assert report.excluded_document_count == 1
    assert report.applied_tombstone_ids == [tombstone.id]
    assert report.overall.documents == 2
    assert "code" not in report.by_domain


def test_cancellation_stops_without_a_report(fixture: Fixture) -> None:
    class Cancelled(RuntimeError):
        pass

    def check() -> None:
        raise Cancelled

    with pytest.raises(Cancelled):
        _evaluate(fixture, check_cancelled=check)
    assert not any((fixture.data / "evaluations").iterdir())
