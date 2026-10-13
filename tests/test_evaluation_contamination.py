"""N-gram contamination checks against real fixture builds and synthetic streams."""

from __future__ import annotations

import tracemalloc
from collections.abc import Iterator
from typing import Any

import pytest
from conftest import VALIDATION_MARKER, TokenShardFixture, make_token_shards
from evaluation_support import TEST_HARNESS, StubScoringModel

from epor.evaluation import (
    ContaminationItem,
    ContaminationSettings,
    EvaluationCancelledError,
    SuiteDefinition,
    TrainingCorpus,
    TrainingData,
    run_contamination,
    run_suite,
    scan_contamination,
)
from epor.evaluation.contamination import ContaminationError, NgramIndex, ngram_hash, text_units
from epor.evaluation.report import report_bytes

PLANTED_QUESTION = (
    "Which ancient lighthouse keeper recorded the tides of the northern harbour every "
    "morning for forty years before the great storm arrived?"
)
PLANTED_CHOICES = ["Alda Venn", "Borin Hale", "Cato Mere", "Dara Quill"]
NEAR_QUESTION = (
    "Which ancient BEACON keeper recorded the tides of the northern harbour every "
    "morning  for forty years before the great storm arrived?!"
)
UNRELATED_QUESTION = (
    "Seven purple giraffes negotiated a careful treaty with the patient moon about the "
    "price of imaginary cheese wheels"
)
UNRELATED_CHOICES = ["velvet", "orbit", "marrow", "quartz"]
FILLER = (
    "Archivists in the valley catalogue river stones by colour and weight, and they "
    "publish the ledger each spring for the village council to review. "
)
MARKER_LINE = " ".join([VALIDATION_MARKER] * 12) + " held out evaluation text only"


def _mc(item_id: str, question: str, choices: list[str]) -> dict[str, Any]:
    return {"id": item_id, "question": question, "choices": choices, "answer": 0}


def contamination_suite(**settings: Any) -> SuiteDefinition:
    return SuiteDefinition.model_validate(
        {
            "id": "contamination-test",
            "version": 1,
            "scope": "harness-fixture",
            "title": "Contamination test",
            "tasks": [
                {
                    "id": "items",
                    "type": "multiple-choice",
                    "items": [
                        _mc("planted", PLANTED_QUESTION, PLANTED_CHOICES),
                        _mc("near", NEAR_QUESTION, PLANTED_CHOICES),
                        _mc("unrelated", UNRELATED_QUESTION, UNRELATED_CHOICES),
                        _mc("marker", MARKER_LINE, ["alpha beta", "gamma delta"]),
                        _mc("short", "Too short?", ["yes", "no"]),
                    ],
                },
                {"id": "scan", "type": "contamination", "contamination": settings or None},
            ],
        }
    )


@pytest.fixture(scope="module")
def clean_build(tmp_path_factory: pytest.TempPathFactory) -> TokenShardFixture:
    planted = PLANTED_QUESTION + "\n" + "\n".join(PLANTED_CHOICES)
    extra = (FILLER * 4) + planted + "\n" + (FILLER * 4)
    return make_token_shards(tmp_path_factory.mktemp("clean"), extra_train_text=extra)


@pytest.fixture(scope="module")
def leaky_build(tmp_path_factory: pytest.TempPathFactory) -> TokenShardFixture:
    # Train text sharing a long run of the validation document's marker words.
    extra = (FILLER * 6) + " ".join([VALIDATION_MARKER] * 30) + "\n" + (FILLER * 6)
    return make_token_shards(tmp_path_factory.mktemp("leaky"), extra_train_text=extra)


def _data(build: TokenShardFixture) -> TrainingData:
    return TrainingData(data_root=build.data_root, tokenized_id=build.tokenized_id)


def _details(report: Any, task: str = "scan") -> dict[str, dict[str, Any]]:
    return {item.item_id: item.details for item in report.task(task).items}


def test_planted_overlaps_are_flagged_and_unrelated_items_are_not(
    clean_build: TokenShardFixture,
) -> None:
    report = run_contamination(contamination_suite(), _data(clean_build), harness=TEST_HARNESS)
    details = _details(report)
    assert details["items/planted"]["flagged"]
    assert details["items/planted"]["text"]["train"]["coverage"] == 1.0
    assert details["items/near"]["text"]["flagged"]
    assert 0.5 <= details["items/near"]["text"]["train"]["coverage"] < 1.0
    assert not details["items/unrelated"]["flagged"]
    assert details["items/unrelated"]["text"]["train"]["coverage"] == 0.0
    assert not details["items/short"]["text"]["assessed"]
    assert details["items/short"]["text"]["train"]["coverage"] is None
    assert [s.task_id for s in report.skipped_tasks] == ["items"]
    assert report.model.kind == "data-only"
    assert report.model.tokenizer_id == clean_build.tokenizer_id
    task = report.task("scan")
    assert task.metric("assessed_items").value == 5.0  # "short" is token-assessed only
    assert task.metric("flagged_items").value is not None
    assert task.settings["lower_bound"] is True
    build = [s for s in report.data if s.kind == "tokenized-build"]
    assert build[0].source_id == clean_build.tokenized_id
    fixture = [s for s in report.data if s.kind == "project-fixture"]
    assert fixture[0].task_ids == ["items", "scan"]


def test_token_and_text_levels_agree_on_the_fixture(clean_build: TokenShardFixture) -> None:
    report = run_contamination(contamination_suite(), _data(clean_build), harness=TEST_HARNESS)
    details = _details(report)
    for item_id in ("items/planted", "items/unrelated"):
        assert details[item_id]["text"]["flagged"] == details[item_id]["token"]["flagged"]
    assert details["items/planted"]["token"]["train"]["coverage"] >= 0.9
    assert details["items/unrelated"]["token"]["train"]["coverage"] < 0.5
    match = details["items/planted"]["token"]["train"]["matches"][0]
    assert len(match["document_id"]) == 64 and match["unit_offset"] >= 0


def test_held_out_splits_can_be_scanned(clean_build: TokenShardFixture) -> None:
    suite = contamination_suite(splits=["train", "validation", "test"], levels=["text"])
    report = run_contamination(suite, _data(clean_build), harness=TEST_HARNESS)
    marker = _details(report)["items/marker"]["text"]
    # Every marker word is covered; the two invented choices are not.
    assert marker["validation"]["coverage"] == pytest.approx(17 / 21)
    assert marker["train"]["coverage"] == 0.0
    assert not marker["flagged"]  # flagged means present in train
    task = report.task("scan")
    assert task.metric("text.validation.any_match_rate").value == 0.25
    assert task.metric("text.test.mean_coverage").value == 0.0


def test_reports_never_contain_item_or_training_text(clean_build: TokenShardFixture) -> None:
    report = run_contamination(contamination_suite(), _data(clean_build), harness=TEST_HARNESS)
    payload = report_bytes(report).decode("utf-8").lower()
    for fragment in ("lighthouse", "giraffes", "borin hale", VALIDATION_MARKER, "archivists"):
        assert fragment not in payload
    planted = _details(report)["items/planted"]
    assert len(planted["item_sha256"]) == 64


def test_report_ids_are_deterministic(clean_build: TokenShardFixture) -> None:
    first = run_contamination(contamination_suite(), _data(clean_build), harness=TEST_HARNESS)
    second = run_contamination(contamination_suite(), _data(clean_build), harness=TEST_HARNESS)
    assert first.report_id == second.report_id
    other = run_contamination(
        contamination_suite(threshold=0.95), _data(clean_build), harness=TEST_HARNESS
    )
    assert other.report_id != first.report_id


def test_run_suite_uses_explicit_training_data(clean_build: TokenShardFixture) -> None:
    suite = contamination_suite(levels=["text"])
    model = StubScoringModel(default_logprob=-1.0)
    skipped = run_suite(suite, model, harness=TEST_HARNESS)
    assert [s.task_id for s in skipped.skipped_tasks] == ["scan"]
    report = run_suite(suite, model, harness=TEST_HARNESS, training_data=_data(clean_build))
    assert report.task("scan").task_type == "contamination"
    assert report.task("items").task_type == "multiple-choice"


def test_split_leakage_detects_held_out_text_in_train(
    clean_build: TokenShardFixture, leaky_build: TokenShardFixture
) -> None:
    suite = contamination_suite(mode="split-leakage")
    leaky = run_contamination(suite, _data(leaky_build), harness=TEST_HARNESS)
    clean = run_contamination(suite, _data(clean_build), harness=TEST_HARNESS)
    assert leaky.task("scan").format == "split-leakage-v1"
    assert leaky.task("scan").metric("flagged_items").value == 1.0
    assert clean.task("scan").metric("flagged_items").value == 0.0
    flagged = [item for item in leaky.task("scan").items if item.score == 1.0]
    assert flagged[0].item_id.startswith("validation:")


def test_cancellation_stops_a_scan(clean_build: TokenShardFixture) -> None:
    with pytest.raises(EvaluationCancelledError):
        run_contamination(
            contamination_suite(),
            _data(clean_build),
            harness=TEST_HARNESS,
            should_cancel=lambda: True,
        )


def test_settings_validation() -> None:
    with pytest.raises(ValueError, match="include train"):
        ContaminationSettings(splits=["validation"])
    with pytest.raises(ValueError, match="unknown item tasks"):
        contamination_suite(targets=["missing"])
    with pytest.raises(ValueError, match="no items"):
        SuiteDefinition.model_validate(
            {
                "id": "bad",
                "version": 1,
                "scope": "harness-fixture",
                "title": "bad",
                "tasks": [{"id": "scan", "type": "contamination", "items": []}],
            }
        )


def test_hashes_are_stable_and_text_units_normalize() -> None:
    assert text_units("Hello,  WORLD! ﬁne") == ["hello", "world", "fine"]
    assert ngram_hash(["a", "b"]) == ngram_hash(["a", "b"])
    assert ngram_hash(["a", "b"]) != ngram_hash(["ab"])
    assert ngram_hash([1, 2, 3]) != ngram_hash(["1", "2", "3"])
    assert ngram_hash([1, 2, 3]) == ngram_hash((1, 2, 3))
    index = NgramIndex(max_ngrams=2)
    with pytest.raises(ContaminationError):
        index.add(0, ["a", "b", "c", "d"], 2)


def test_streaming_scan_has_bounded_memory() -> None:
    words = [f"w{i}" for i in range(500)]

    def documents(split: str) -> Iterator[tuple[str, list[str]]]:
        if split != "train":
            return
        for number in range(4_000):
            # 200 fresh words per document; one document plants the item.
            if number == 2_500:
                yield f"doc-{number}", text_units(PLANTED_QUESTION)
            else:
                yield f"doc-{number}", [words[(number + k) % 500] for k in range(200)]

    corpus = TrainingCorpus(text=documents, token=None, encode=None)
    items = [
        ContaminationItem("planted", "t", PLANTED_QUESTION),
        ContaminationItem("unrelated", "t", UNRELATED_QUESTION),
    ]
    tracemalloc.start()
    result = scan_contamination(
        "scan",
        items,
        corpus,
        scope="harness-fixture",
        settings=ContaminationSettings(levels=["text"]),
    )
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    # 800,000 streamed words would need tens of MB if they were held at once.
    assert peak < 4 * 1024 * 1024
    assert result.settings["documents_scanned"] == {"text.train": 4_000}
    assert [item.score for item in result.items] == [1.0, 0.0]


def test_token_level_needs_a_tokenizer() -> None:
    corpus = TrainingCorpus(text=None, token=lambda split: iter(()), encode=None)
    with pytest.raises(ContaminationError):
        scan_contamination(
            "scan",
            [ContaminationItem("x", "t", PLANTED_QUESTION)],
            corpus,
            scope="harness-fixture",
            settings=ContaminationSettings(levels=["token"]),
        )
