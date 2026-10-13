"""Task scoring with fixed-output models, code-execution gating, suites, and sources."""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import yaml
from evaluation_support import TEST_HARNESS, StubScoringModel, small_suite, small_suite_model

from epor.data.models import SourceRegistration, Tombstone
from epor.data.store import DataStore
from epor.evaluation import (
    CodeExecutionRefusedError,
    CodeSandbox,
    EvaluationCancelledError,
    SourceRefusedError,
    SuiteDefinition,
    fixture_suite,
    load_registered_items,
    load_suite,
    run_suite,
)
from epor.evaluation.tasks import (
    CodeItem,
    ExactMatchItem,
    MathItem,
    MultipleChoiceItem,
    code_program,
    math_prompt,
    run_code,
    run_exact_match,
    run_math,
    run_multiple_choice,
)


def test_multiple_choice_raw_and_normalized_scores_are_exact() -> None:
    items = [MultipleChoiceItem(id="q1", question="Pick", choices=["1", "100"], answer=1)]
    model = StubScoringModel(
        {("Question: Pick\nAnswer:", " 1"): -2.0, ("Question: Pick\nAnswer:", " 100"): -3.0}
    )
    result = run_multiple_choice("mc", items, model, scope="harness-fixture", calibration_bins=10)
    # Raw: -2 > -3 picks "1" (wrong).  Per byte: -2/2 = -1 < -3/4 = -0.75 picks "100".
    assert result.primary_metric == "accuracy_norm"
    assert result.metric("accuracy").value == 0.0
    assert result.metric("accuracy_norm").value == 1.0
    details = result.items[0].details
    assert details["logprobs_per_byte"] == [-1.0, -0.75]
    expected = 1 / (1 + math.exp(-0.25))
    assert details["probabilities"] == pytest.approx([1 - expected, expected])
    assert result.calibration is not None
    # One correct item at confidence 0.5622: ECE = 1 - 0.5622; Brier = 2 * (1 - p)^2.
    assert result.metric("ece").value == pytest.approx(1 - expected)
    assert result.metric("brier").value == pytest.approx(2 * (1 - expected) ** 2)
    assert result.calibration.reliability[5].count == 1


def test_exact_match_normalization() -> None:
    items = [
        ExactMatchItem(id="a", prompt="Q1", answers=["Paris"]),
        ExactMatchItem(id="b", prompt="Q2", answers=["four", "4"]),
        ExactMatchItem(id="c", prompt="Q3", answers=["yes"]),
    ]
    model = StubScoringModel(generations={"Q1": "  PARIS.\nmore", "Q2": " 4", "Q3": "no"})
    result = run_exact_match(
        "em",
        items,
        model,
        scope="harness-fixture",
        normalization=["strip", "lower", "strip_punctuation"],
    )
    assert [item.score for item in result.items] == [1.0, 1.0, 0.0]
    assert result.metric("exact_match").value == pytest.approx(2 / 3)
    assert result.items[0].details["prediction"] == "paris"


def test_math_scoring_uses_safe_equivalence() -> None:
    items = [
        MathItem(id="m1", problem="3/4 + 1/8?", answer="7/8"),
        MathItem(id="m2", problem="Expand", answer="2x+1"),
        MathItem(id="m3", problem="Evil", answer="1"),
        MathItem(id="m4", problem="Silent", answer="1"),
    ]
    model = StubScoringModel(
        generations={
            math_prompt(items[0]): " The sum is 0.875. #### 0.875",
            math_prompt(items[1]): " So it is \\boxed{1 + 2x}.",
            math_prompt(items[2]): " The answer is __import__('os').getcwd()",
            math_prompt(items[3]): " I do not know.",
        }
    )
    result = run_math("math", items, model, scope="harness-fixture")
    assert [item.score for item in result.items] == [1.0, 1.0, 0.0, 0.0]
    assert result.metric("answer_extracted").value == pytest.approx(0.75)
    assert result.items[2].details["extracted"] == "__import__('os').getcwd()"


CODE_ITEM = CodeItem(
    id="add",
    prompt='def add(a, b):\n    """Add."""\n',
    test="def check(candidate):\n    assert candidate(2, 3) == 5\n",
    entry_point="add",
)


def test_code_execution_is_refused_unless_opted_in() -> None:
    model = StubScoringModel(generations={CODE_ITEM.prompt: "    return a + b\n"})
    with pytest.raises(CodeExecutionRefusedError, match="allow_code_execution"):
        run_code("code", [CODE_ITEM], model, scope="harness-fixture")
    assert model.calls == []  # nothing was generated, let alone executed


def test_code_execution_needs_network_opt_in_when_isolation_is_missing() -> None:
    model = StubScoringModel(generations={CODE_ITEM.prompt: "    return a + b\n"})
    degraded = CodeSandbox(use_namespaces=False, use_landlock=False)
    with pytest.raises(CodeExecutionRefusedError, match="allow_unisolated_network"):
        run_code(
            "code",
            [CODE_ITEM],
            model,
            scope="harness-fixture",
            allow_code_execution=True,
            sandbox=degraded,
        )
    result = run_code(
        "code",
        [CODE_ITEM],
        model,
        scope="harness-fixture",
        allow_code_execution=True,
        allow_unisolated_network=True,
        sandbox=degraded,
    )
    assert result.sandbox is not None and result.sandbox.network_isolated is False
    assert result.metric("pass_at_1").value == 1.0


def test_code_scoring_runs_tests_and_truncates_completions() -> None:
    wrong = CODE_ITEM.model_copy(update={"id": "wrong"})
    model = StubScoringModel(
        generations={
            CODE_ITEM.prompt: "    return a + b\n\ndef extra():\n    raise SystemExit(1)\n"
        }
    )
    items = [CODE_ITEM, wrong]
    model.generations[wrong.prompt] = "    return a + b\n"
    result = run_code(
        "code",
        items,
        model,
        scope="harness-fixture",
        allow_code_execution=True,
        allow_unisolated_network=True,
    )
    assert result.items[0].details["completion"] == "    return a + b\n"
    assert result.items[0].details["status"] == "passed"
    assert result.sandbox is not None
    failing = StubScoringModel(generations={CODE_ITEM.prompt: "    return a - b\n"})
    failed = run_code(
        "code",
        [CODE_ITEM],
        failing,
        scope="harness-fixture",
        allow_code_execution=True,
        allow_unisolated_network=True,
    )
    assert failed.items[0].score == 0.0
    assert failed.items[0].details["status"] == "failed"
    assert "AssertionError" in failed.items[0].details["stderr_tail"]
    assert code_program(CODE_ITEM, "    return 0\n").endswith("check(add)\n")


# ----------------------------------------------------------------------------- suites


def test_fixture_suite_is_valid_and_labelled() -> None:
    suite = fixture_suite()
    assert suite.scope == "harness-fixture"
    assert {task.type for task in suite.tasks} == {
        "multiple-choice",
        "exact-match",
        "math",
        "code",
        "validation-loss",
    }
    assert all(task.source is None for task in suite.tasks)


def test_suite_runs_skip_gated_tasks_and_record_why() -> None:
    suite = fixture_suite()
    model = StubScoringModel(default_logprob=-1.0)
    report = run_suite(suite, model, harness=TEST_HARNESS)
    assert {s.task_id: s.reason for s in report.skipped_tasks} == {
        "fixture-code": "code execution not enabled",
        "fixture-validation-loss": "model is not a kernel checkpoint",
    }
    assert [t.task_id for t in report.tasks] == [
        "fixture-multiple-choice",
        "fixture-exact-match",
        "fixture-math",
    ]


def test_fixture_code_items_pass_with_reference_solutions() -> None:
    suite = fixture_suite()
    code = next(task for task in suite.tasks if task.type == "code")
    assert code.items is not None
    solutions = {
        "add_numbers": "    return a + b\n",
        "reverse_words": "    return ' '.join(reversed(sentence.split()))\n",
        "is_palindrome": "    return text == text[::-1]\n",
    }
    generations = {item["prompt"]: solutions[item["entry_point"]] for item in code.items}
    model = StubScoringModel(generations=generations, default_logprob=-1.0)
    report = run_suite(
        suite,
        model,
        allow_code_execution=True,
        allow_unisolated_network=True,
        harness=TEST_HARNESS,
    )
    assert report.task("fixture-code").metric("pass_at_1").value == 1.0


def test_suite_validation_and_cancellation(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exactly one of items or source"):
        small_suite(tasks=[{"id": "t", "type": "math"}])
    with pytest.raises(ValueError, match="harness-fixture suites"):
        small_suite(
            tasks=[
                {
                    "id": "t",
                    "type": "math",
                    "source": {"source_id": "s", "format": "gsm8k-jsonl"},
                }
            ]
        )
    with pytest.raises(ValueError, match="does not produce"):
        small_suite(
            scope="proxy-scale",
            tasks=[
                {"id": "t", "type": "math", "source": {"source_id": "s", "format": "mbpp-jsonl"}}
            ],
        )
    with pytest.raises(ValueError):
        small_suite(tasks=[{"id": "t", "type": "math", "items": [{"id": "x", "problem": "p"}]}])
    with pytest.raises(EvaluationCancelledError):
        run_suite(small_suite(), small_suite_model(), should_cancel=lambda: True)
    path = tmp_path / "suite.yaml"
    path.write_text(yaml.safe_dump(small_suite().model_dump(mode="json")), encoding="utf-8")
    assert load_suite(path) == small_suite()


# ----------------------------------------------------------------- external sources


def _registration(source_id: str, **overrides: Any) -> SourceRegistration:
    values: dict[str, Any] = {
        "id": source_id,
        "title": "Operator-reviewed benchmark copy",
        "owner_or_steward": "Test steward",
        "canonical_origin": "fixture://evaluation-test",
        "acquisition_method": "generated_fixture",
        "acquired_at": datetime(2026, 10, 1, tzinfo=UTC),
        "media_type": "application/jsonl",
        "license_id": "test-fixture",
        "permission_basis": "Project-authored test fixture.",
        "allowed_uses": ["research", "evaluate"],
        "sensitive_content_risk": "low",
        "removal_contact": "owner",
        **overrides,
    }
    return SourceRegistration.model_validate(values)


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return path


MC_RECORDS = [
    {"question": "2 + 2?", "choices": ["3", "4"], "answer": 1},
    {"id": "letter", "question": "First letter?", "choices": ["a", "b"], "answer": "A"},
]


def test_registered_source_is_loaded_with_lineage(tmp_path: Path) -> None:
    store = DataStore(tmp_path / "data")
    registration_sha256 = store.register(_registration("local-mc"))
    path = _write_jsonl(tmp_path / "mc.jsonl", MC_RECORDS)
    loaded = load_registered_items(store, "local-mc", path, "multiple-choice-jsonl")
    assert [item.id for item in loaded.items] == ["local-mc:1", "letter"]
    assert isinstance(loaded.items[1], MultipleChoiceItem) and loaded.items[1].answer == 0
    assert loaded.lineage.kind == "registered-source"
    assert loaded.lineage.registration_sha256 == registration_sha256
    assert str(path) not in loaded.lineage.model_dump_json()

    suite = SuiteDefinition.model_validate(
        {
            "id": "operator-suite",
            "version": 1,
            "scope": "proxy-scale",
            "title": "Operator suite",
            "tasks": [
                {
                    "id": "mc",
                    "type": "multiple-choice",
                    "source": {"source_id": "local-mc", "format": "multiple-choice-jsonl"},
                }
            ],
        }
    )
    report = run_suite(
        suite,
        StubScoringModel(default_logprob=-1.0),
        store=store,
        source_paths={"local-mc": path},
        harness=TEST_HARNESS,
    )
    assert report.data == [loaded.lineage.model_copy(update={"task_ids": ["mc"]})]
    assert report.task("mc").metric("accuracy_norm").scope == "proxy-scale"
    assert "Not a capability claim" in report.claim_boundary


def test_unregistered_and_rights_incompatible_sources_are_refused(tmp_path: Path) -> None:
    store = DataStore(tmp_path / "data")
    path = _write_jsonl(tmp_path / "mc.jsonl", MC_RECORDS)
    with pytest.raises(SourceRefusedError, match="not registered"):
        load_registered_items(store, "missing", path, "multiple-choice-jsonl")
    store.register(_registration("train-only", allowed_uses=["research", "train"]))
    with pytest.raises(SourceRefusedError, match="not registered for evaluation"):
        load_registered_items(store, "train-only", path, "multiple-choice-jsonl")
    store.register(_registration("wrong-type", media_type="text/plain"))
    with pytest.raises(SourceRefusedError, match="application/jsonl"):
        load_registered_items(store, "wrong-type", path, "multiple-choice-jsonl")
    store.register(_registration("removed"))
    store.write_tombstone(
        Tombstone(
            id="1" * 64,
            target_kind="source_id",
            target="removed",
            reason="rights withdrawn",
            requested_at=datetime(2026, 10, 2, tzinfo=UTC),
            requested_by="owner",
        )
    )
    with pytest.raises(SourceRefusedError, match="tombstoned"):
        load_registered_items(store, "removed", path, "multiple-choice-jsonl")
    store.register(_registration("ok"))
    with pytest.raises(SourceRefusedError, match="pinned SHA-256"):
        load_registered_items(store, "ok", path, "multiple-choice-jsonl", expected_sha256="0" * 64)
    link = tmp_path / "link.jsonl"
    link.symlink_to(path)
    with pytest.raises(SourceRefusedError, match="non-symlink"):
        load_registered_items(store, "ok", link, "multiple-choice-jsonl")
    with pytest.raises(SourceRefusedError, match="limit"):
        load_registered_items(store, "ok", path, "multiple-choice-jsonl", max_bytes=10)


def test_public_format_adapters(tmp_path: Path) -> None:
    store = DataStore(tmp_path / "data")
    store.register(_registration("bench"))
    gsm = _write_jsonl(
        tmp_path / "gsm.jsonl",
        [{"question": "Q?", "answer": "Work 2 + 2 = 4.\n#### 1,004"}],
    )
    item = load_registered_items(store, "bench", gsm, "gsm8k-jsonl").items[0]
    assert isinstance(item, MathItem) and item.answer == "1,004"
    competition = _write_jsonl(
        tmp_path / "math.jsonl", [{"problem": "P", "solution": "so $\\boxed{\\frac{1}{2}}$"}]
    )
    item = load_registered_items(store, "bench", competition, "competition-math-jsonl").items[0]
    assert isinstance(item, MathItem) and item.answer == "\\frac{1}{2}"
    humaneval = _write_jsonl(
        tmp_path / "he.jsonl",
        [
            {
                "task_id": "T/0",
                "prompt": CODE_ITEM.prompt,
                "test": CODE_ITEM.test,
                "entry_point": "add",
                "canonical_solution": "    return a + b\n",
            }
        ],
    )
    item = load_registered_items(store, "bench", humaneval, "humaneval-jsonl").items[0]
    assert isinstance(item, CodeItem) and item.id == "T/0" and item.entry_point == "add"
    mbpp = _write_jsonl(
        tmp_path / "mbpp.jsonl",
        [{"task_id": 7, "text": "Add.", "test_list": ["assert add(1, 2) == 3"]}],
    )
    item = load_registered_items(store, "bench", mbpp, "mbpp-jsonl").items[0]
    assert isinstance(item, CodeItem) and item.entry_point is None
    assert "assert add(1, 2) == 3" in item.test
    exact = _write_jsonl(tmp_path / "em.jsonl", [{"prompt": "P", "answer": "x"}] * 3)
    loaded = load_registered_items(store, "bench", exact, "exact-match-jsonl", limit=2)
    assert len(loaded.items) == 2
    bad = _write_jsonl(tmp_path / "bad.jsonl", [{"question": "Q"}])
    with pytest.raises(ValueError, match="line 1"):
        load_registered_items(store, "bench", bad, "multiple-choice-jsonl")
