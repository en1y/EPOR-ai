"""Sealed evaluation reports: determinism, storage, tamper detection, and comparison."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from evaluation_support import TEST_HARNESS, small_suite, small_suite_model

from epor.data.store import DataStore, DataStoreError
from epor.evaluation import (
    EvaluationReport,
    ReportError,
    compare_reports,
    load_report,
    paired_bootstrap,
    run_suite,
    store_report,
    verify_report,
)
from epor.evaluation.report import parse_report, report_bytes, report_digest


def _report(*, second_correct: bool = True) -> EvaluationReport:
    model = small_suite_model(second_correct=second_correct)
    return run_suite(small_suite(), model, harness=TEST_HARNESS)


def test_identical_inputs_give_identical_report_ids() -> None:
    first = _report()
    second = _report()
    assert first.report_id == second.report_id == report_digest(first)
    assert report_bytes(first) == report_bytes(second)
    assert first.scope == "harness-fixture"
    assert "Not a benchmark" in first.claim_boundary
    assert all(metric.scope == "harness-fixture" for t in first.tasks for metric in t.metrics)
    # Anything that changes the inputs changes the ID.
    assert _report(second_correct=False).report_id != first.report_id
    other_harness = TEST_HARNESS.model_copy(update={"code_revision": "other"})
    changed = run_suite(small_suite(), small_suite_model(), harness=other_harness)
    assert changed.report_id != first.report_id


def test_report_records_lineage_seeds_and_suite_identity() -> None:
    report = _report()
    suite = small_suite()
    assert report.suite_id == "test-suite"
    assert report.suite_sha256 == suite.sha256
    assert report.seeds == {"evaluation": 0}
    assert [source.kind for source in report.data] == ["project-fixture", "project-fixture"]
    assert {source.source_id for source in report.data} == {"test-suite/em", "test-suite/mc"}
    assert report.model.tokenizer_id == "stub-tokenizer"
    assert report.harness.code_revision == "test-revision"


def test_store_and_load_round_trip_and_immutability(tmp_path: Path) -> None:
    store = DataStore(tmp_path / "data")
    report = _report()
    path = store_report(store, report)
    assert path == tmp_path / "data" / "evaluation-reports" / f"{report.report_id}.json"
    assert path.stat().st_mode & 0o777 == 0o600
    assert store.evaluation_report_ids() == [report.report_id]
    assert store_report(store, report) == path  # identical bytes: no-op
    assert load_report(store, report.report_id) == report
    # The fertility-report layer is untouched.
    assert store.evaluation_ids() == []


def test_tampering_is_detected(tmp_path: Path) -> None:
    store = DataStore(tmp_path / "data")
    report = _report()
    path = store_report(store, report)
    document = json.loads(path.read_bytes())
    document["tasks"][0]["metrics"][0]["value"] = 0.123
    path.chmod(0o600)
    path.write_bytes(json.dumps(document, sort_keys=True, separators=(",", ":")).encode())
    with pytest.raises(ReportError, match="does not match its content"):
        load_report(store, report.report_id)

    # A re-sealed edit stored under the old name is caught by the name check.
    resealed = report.model_copy(update={"suite_version": 2})
    resealed = resealed.model_copy(update={"report_id": report_digest(resealed)})
    path.write_bytes(report_bytes(resealed))
    with pytest.raises(ReportError, match="wrong ID"):
        load_report(store, report.report_id)

    # Non-canonical bytes (pretty-printed) are refused even with a valid digest.
    pretty = json.dumps(json.loads(report_bytes(report)), indent=2).encode()
    with pytest.raises(ReportError, match="not canonical"):
        parse_report(pretty)

    forged = report.model_copy(update={"report_id": "f" * 64})
    with pytest.raises(ReportError):
        verify_report(forged)
    with pytest.raises(ReportError):
        store_report(store, forged)
    with pytest.raises(ReportError, match="not found"):
        load_report(store, "a" * 64)
    with pytest.raises(ReportError):
        load_report(store, "../escape")


def test_store_refuses_different_bytes_at_an_existing_address(tmp_path: Path) -> None:
    store = DataStore(tmp_path / "data")
    report = _report()
    store_report(store, report)
    with pytest.raises(DataStoreError, match="different bytes"):
        store.write_evaluation_report(report.report_id, b"{}")


def test_paired_bootstrap_is_deterministic_and_hand_checkable() -> None:
    baseline = [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0]
    candidate = [1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]
    first = paired_bootstrap(baseline, candidate, resamples=500, seed=7, label="t")
    second = paired_bootstrap(baseline, candidate, resamples=500, seed=7, label="t")
    assert first == second
    mean, lower, upper = first
    assert mean == pytest.approx(3 / 8)
    assert 0.0 <= lower <= mean <= upper <= 1.0
    spread = [((index * 37) % 101) / 101 for index in range(20)]
    zeros = [0.0] * 20
    assert paired_bootstrap(zeros, spread, resamples=200, seed=1) != paired_bootstrap(
        zeros, spread, resamples=200, seed=2
    )
    # Constant differences have a degenerate interval at the difference.
    assert paired_bootstrap([0.0, 0.5], [1.0, 1.5], resamples=50) == (1.0, 1.0, 1.0)
    with pytest.raises(ValueError):
        paired_bootstrap([1.0], [1.0, 2.0])


def test_compare_reports_deltas_and_bootstrap() -> None:
    baseline = _report(second_correct=False)
    candidate = _report(second_correct=True)
    comparison = compare_reports(baseline, candidate, resamples=200, seed=3)
    assert comparison == compare_reports(baseline, candidate, resamples=200, seed=3)
    assert comparison.baseline_report_id == baseline.report_id
    assert comparison.warnings == []
    deltas = {(d.task_id, d.name): d for d in comparison.metric_deltas}
    assert deltas[("mc", "accuracy_norm")].baseline == pytest.approx(0.5)
    assert deltas[("mc", "accuracy_norm")].candidate == pytest.approx(1.0)
    assert deltas[("mc", "accuracy_norm")].delta == pytest.approx(0.5)
    assert deltas[("em", "exact_match")].delta == pytest.approx(0.0)
    mc = next(b for b in comparison.bootstraps if b.task_id == "mc")
    assert (mc.paired_items, mc.unmatched_items) == (2, 0)
    assert mc.mean_delta == pytest.approx(0.5)
    assert mc.ci_lower is not None and mc.ci_upper is not None
    assert 0.0 <= mc.ci_lower <= mc.ci_upper <= 1.0


def test_compare_reports_warns_on_incomparable_inputs() -> None:
    baseline = _report()
    other_suite = small_suite(id="other-suite")
    candidate = run_suite(other_suite, small_suite_model(), harness=TEST_HARNESS)
    warnings = compare_reports(baseline, candidate).warnings
    assert any("different suites" in warning for warning in warnings)
    assert any("evaluation data differs" in warning for warning in warnings)
