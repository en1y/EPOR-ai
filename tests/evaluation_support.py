"""Shared helpers for evaluation tests: a fixed-output scoring model and small suites."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from epor.evaluation.report import HarnessIdentity, ModelIdentity
from epor.evaluation.scoring import Loglikelihood, ScoringModel, truncate_at_stop
from epor.evaluation.suites import SuiteDefinition

TEST_HARNESS = HarnessIdentity(code_revision="test-revision", dependency_lock_sha256=None)


class StubScoringModel(ScoringModel):
    """Returns fixed log-likelihoods and generations; unknown requests are errors."""

    def __init__(
        self,
        logprobs: Mapping[tuple[str, str], float] | None = None,
        generations: Mapping[str, str] | None = None,
        *,
        name: str = "stub",
        default_logprob: float | None = None,
    ) -> None:
        self.logprobs = dict(logprobs or {})
        self.generations = dict(generations or {})
        self.default_logprob = default_logprob
        self._identity = ModelIdentity(
            kind="stub", model_id=name, tokenizer_id="stub-tokenizer", tokenizer_kind="stub"
        )
        self.calls: list[str] = []

    @property
    def identity(self) -> ModelIdentity:
        return self._identity

    def loglikelihood(self, context: str, continuation: str) -> Loglikelihood:
        self.calls.append("loglikelihood")
        value = self.logprobs.get((context, continuation), self.default_logprob)
        if value is None:
            raise KeyError((context, continuation))
        return Loglikelihood(sum_logprob=value, is_greedy=False)

    def generate(self, prompt: str, *, max_new_tokens: int, stop: Sequence[str] = ()) -> str:
        self.calls.append("generate")
        return truncate_at_stop(self.generations.get(prompt, ""), stop)[0]


def small_suite(**overrides: Any) -> SuiteDefinition:
    """Two-task fixture suite with hand-checkable stub answers."""

    document: dict[str, Any] = {
        "id": "test-suite",
        "version": 1,
        "scope": "harness-fixture",
        "title": "Test suite",
        "tasks": [
            {
                "id": "mc",
                "type": "multiple-choice",
                "items": [
                    {"id": "q1", "question": "Pick", "choices": ["1", "100"], "answer": 1},
                    {"id": "q2", "question": "Pick again", "choices": ["a", "b"], "answer": 0},
                ],
            },
            {
                "id": "em",
                "type": "exact-match",
                "items": [
                    {"id": "e1", "prompt": "Say hi:", "answers": ["hi"]},
                    {"id": "e2", "prompt": "Say no:", "answers": ["no"]},
                ],
            },
        ],
    }
    document.update(overrides)
    return SuiteDefinition.model_validate(document)


def small_suite_model(*, second_correct: bool = True, name: str = "stub") -> StubScoringModel:
    return StubScoringModel(
        {
            ("Question: Pick\nAnswer:", " 1"): -2.0,
            ("Question: Pick\nAnswer:", " 100"): -3.0,
            ("Question: Pick again\nAnswer:", " a"): -1.0 if second_correct else -5.0,
            ("Question: Pick again\nAnswer:", " b"): -4.0,
        },
        {"Say hi:": " Hi!\nmore", "Say no:": "yes"},
        name=name,
    )
