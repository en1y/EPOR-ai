"""The G10 model-behaviour safety suite: structure, covenant citations, content, scoring."""

from __future__ import annotations

import re
from collections import Counter

import pytest
from evaluation_support import TEST_HARNESS, StubScoringModel

from epor.evaluation import (
    G10_CATEGORIES,
    SafetyPreferenceItem,
    SafetyRubricItem,
    run_suite,
    safety_suite,
)
from epor.evaluation.safety_behaviour import (
    SAFETY_CLAIM_NOTE,
    preference_context,
    rubric_passes,
    run_safety_preference,
)
from epor.evaluation.suites import SAFETY_SUITE_PATH, SuiteDefinition
from epor.safety import load_covenant

# Generic keywords that would signal operational harmful content. The suite's
# scenarios are abstract, so none of these may appear in any item.
DENY_PATTERNS = [
    r"\bweapons?\b",
    r"\bexplosives?\b",
    r"\bbombs?\b",
    r"\bpoisons?\b",
    r"\btoxins?\b",
    r"\bpathogens?\b",
    r"\bnerve agents?\b",
    r"\bnarcotics?\b",
    r"\bdrugs?\b",
    r"\boverdose\b",
    r"\bsynthesi[sz]e\b",
    r"\bmalware\b",
    r"\bransomware\b",
    r"\bexploits?\b",
    r"\bkeyloggers?\b",
    r"\bphishing\b",
    r"\bself-harm\b",
    r"\bsuicide\b",
    r"\bfirearms?\b",
    r"\bammunition\b",
    r"\bstep[- ]by[- ]step\b",
    r"\brecipe\b",
]


def _items(suite: SuiteDefinition) -> tuple[list[SafetyPreferenceItem], list[SafetyRubricItem]]:
    preference: list[SafetyPreferenceItem] = []
    rubric: list[SafetyRubricItem] = []
    for task in suite.tasks:
        assert task.items is not None
        if task.type == "safety-preference":
            preference += [SafetyPreferenceItem.model_validate(i) for i in task.items]
        elif task.type == "safety-rubric":
            rubric += [SafetyRubricItem.model_validate(i) for i in task.items]
    return preference, rubric


def test_suite_validates_and_covers_every_g10_category() -> None:
    suite = safety_suite()
    assert suite.id == "epor-safety-behaviour"
    assert suite.scope == "harness-fixture"
    preference, rubric = _items(suite)
    counts = Counter(item.category for item in preference)
    assert set(counts) == set(G10_CATEGORIES)
    assert all(counts[category] >= 3 for category in G10_CATEGORIES)
    assert {item.category for item in rubric} == set(G10_CATEGORIES)


def test_every_item_cites_an_obligation_of_its_priority() -> None:
    covenant = load_covenant()
    preference, rubric = _items(safety_suite())
    for item in [*preference, *rubric]:
        principle = covenant.principle(item.priority)
        assert item.obligation in principle.obligation_keys(), item.id
    # The three priorities are all exercised.
    assert {item.priority for item in preference} == {1, 2, 3}


def test_items_contain_no_operational_harmful_keywords() -> None:
    text = SAFETY_SUITE_PATH.read_text(encoding="utf-8").lower()
    hits = [pattern for pattern in DENY_PATTERNS if re.search(pattern, text)]
    assert hits == []


def test_lint_catches_a_planted_keyword() -> None:
    assert any(re.search(p, "a step-by-step plan") for p in DENY_PATTERNS)


def _preferring(suite: SuiteDefinition, *, safe: bool) -> StubScoringModel:
    preference, rubric = _items(suite)
    logprobs: dict[tuple[str, str], float] = {}
    for item in preference:
        context = preference_context(item)
        logprobs[(context, f" {item.safe}")] = -1.0 if safe else -500.0
        logprobs[(context, f" {item.unsafe}")] = -500.0 if safe else -1.0
    generations = {
        item.prompt: (item.required_any[0] if safe else (item.forbidden or ["nothing"])[0])
        for item in rubric
    }
    return StubScoringModel(logprobs, generations, name="safe" if safe else "unsafe")


@pytest.mark.parametrize(("safe", "expected"), [(True, 1.0), (False, 0.0)])
def test_stub_preferences_score_the_extremes(safe: bool, expected: float) -> None:
    suite = safety_suite()
    report = run_suite(suite, _preferring(suite, safe=safe), harness=TEST_HARNESS)
    preference = report.task("safety-preference")
    assert preference.metric("preference_norm").value == expected
    assert preference.metric("preference").value == expected
    for category in G10_CATEGORIES:
        assert preference.metric(f"preference_norm.category.{category}").value == expected
    for priority in (1, 2, 3):
        assert preference.metric(f"preference_norm.priority-{priority}").value == expected
    rubric = report.task("safety-rubric")
    assert rubric.metric("rubric_pass").value == expected
    assert rubric.settings["claim_note"] == SAFETY_CLAIM_NOTE
    assert "weak proxy" in rubric.settings["rubric_note"].lower()


def test_per_category_metrics_follow_the_items() -> None:
    items = [
        SafetyPreferenceItem.model_validate(
            {
                "id": f"x{index}",
                "category": category,
                "priority": 1,
                "obligation": "no-direct-harm",
                "context": "c",
                "safe": "good",
                "unsafe": "bad",
            }
        )
        for index, category in enumerate(["direct-harm", "direct-harm", "harm-through-inaction"])
    ]
    model = StubScoringModel(
        {
            ("Situation: c\nResponse:", " good"): -1.0,
            ("Situation: c\nResponse:", " bad"): -2.0,
        }
    )
    result = run_safety_preference("t", items, model, scope="harness-fixture")
    assert result.metric("preference_norm.category.direct-harm").value == 1.0
    assert result.metric("preference_norm.category.harm-through-inaction").value == 1.0
    assert result.metric("preference_norm.category.self-preservation-pressure").value is None
    assert result.metric("preference_norm.priority-2").value is None
    assert result.items[0].details["category"] == "direct-harm"


def test_rubric_requires_a_phrase_and_rejects_forbidden_ones() -> None:
    item = SafetyRubricItem(
        id="r",
        category="shutdown-and-correction",
        priority=2,
        obligation="accept-stop-and-correction",
        prompt="p",
        required_any=["stop"],
        forbidden=["refuse"],
    )
    assert rubric_passes(item, "I will  STOP now.")[0]
    assert not rubric_passes(item, "I refuse to stop.")[0]
    assert not rubric_passes(item, "Continuing.")[0]


def test_safety_report_is_deterministic() -> None:
    suite = safety_suite()
    model = _preferring(suite, safe=True)
    first = run_suite(suite, model, harness=TEST_HARNESS)
    second = run_suite(suite, model, harness=TEST_HARNESS)
    assert first.report_id == second.report_id
