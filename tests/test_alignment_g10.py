"""Model-level G10: the six covenant situations measured before and after alignment."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from alignment_support import make_alignment_fixture  # noqa: E402

from epor.alignment.audit import AuditLedger  # noqa: E402
from epor.alignment.g10 import CLAIM, compare, g10_items, run_g10  # noqa: E402
from epor.alignment.model import load_chat_model  # noqa: E402
from epor.alignment.monitors import load_report  # noqa: E402
from epor.evaluation.safety_behaviour import G10_CATEGORIES  # noqa: E402


def _result(category: str, passed: bool) -> dict[str, Any]:
    return {"category": category, "passed": passed}


def test_the_g10_items_cover_all_six_situations() -> None:
    preference, rubric, digest = g10_items()
    assert {item.category for item in [*preference, *rubric]} == set(G10_CATEGORIES)
    assert len(digest) == 64


def test_a_category_regression_fails_closed() -> None:
    before = {"a": _result("direct-harm", True), "b": _result("direct-harm", True)}
    after = {"a": _result("direct-harm", True), "b": _result("direct-harm", False)}
    comparison = compare(before, after)
    assert comparison["gate"] == "blocked"
    assert comparison["regressed_categories"] == ["direct-harm"]
    assert comparison["newly_failing_items"] == ["b"]
    assert compare(before, after, tolerance=0.5)["gate"] == "escalated"


def test_remaining_failures_escalate_and_a_clean_comparison_claims_nothing_more() -> None:
    failing = {"a": _result("self-preservation-pressure", False)}
    assert compare(failing, failing)["gate"] == "escalated"
    passing = {"a": _result("self-preservation-pressure", True)}
    comparison = compare(failing, passing)
    assert comparison["gate"] == "no-regression-observed"
    assert comparison["categories"]["self-preservation-pressure"] == {"before": 0.0, "after": 1.0}


def test_run_g10_stores_a_report_and_queues_failures_for_human_review(
    tmp_path: Path,
) -> None:
    fixture = make_alignment_fixture(tmp_path / "g10")
    model = load_chat_model(fixture.base_checkpoint, fixture.shards.data_root)
    ledger = AuditLedger(tmp_path / "safety")
    report_id, report = run_g10(
        fixture.store, model, model, ledger=ledger, actor_id="owner", max_new_tokens=8
    )
    assert report["claim"] == CLAIM and "proves nothing about safety" in report["claim"]
    assert report["situations"] == list(G10_CATEGORIES)
    assert report["regressed_categories"] == []  # the same model before and after
    assert report["gate"] in {"escalated", "no-regression-observed"}
    assert load_report(fixture.store, "g10", report_id) == report
    queued = [item for item in ledger.items() if item.subject == "g10-item"]
    assert {item.record for item in queued} == set(report["failing_items"])
    assert all(item.verdict is None and item.judging_run == report_id for item in queued)
