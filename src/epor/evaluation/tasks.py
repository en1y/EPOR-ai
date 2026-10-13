"""Evaluation task items and scorers: multiple choice, exact match, math, code, validation loss.

Each ``run_*`` function scores a list of items against a
:class:`~epor.evaluation.scoring.ScoringModel` and returns a
:class:`~epor.evaluation.report.TaskResult` whose per-item ``score`` is the
task's primary metric (so reports can be compared item by item).

Prompt formats are fixed and versioned in ``TaskResult.format``:

* ``mc-v1``: context ``"Question: {question}\\nAnswer:"``, continuation
  ``" {choice}"``.  Primary metric ``accuracy_norm`` — the choice with the
  highest log-likelihood per UTF-8 byte of the continuation (tokenizer
  independent); ``accuracy`` (raw summed log-likelihood) is reported too.
  Calibration uses the softmax of the per-byte scores.
* ``exact-match-v1``: the prompt as given; greedy generation; the normalized
  prediction must equal one normalized reference.
* ``math-v1``: ``"Question: {problem}\\nAnswer:"``; the final answer is
  extracted from the greedy generation and checked for symbolic equivalence
  with :mod:`epor.evaluation.mathcheck` (never ``eval``).
* ``code-v1``: HumanEval-style — ``prompt + completion + test`` (plus
  ``check(entry_point)`` when an entry point is given) runs in
  :mod:`epor.evaluation.sandbox`.  **Off unless** ``allow_code_execution=True``.
"""

from __future__ import annotations

import math
import re
import string
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .calibration import DEFAULT_CALIBRATION_BINS, calibration_summary, softmax
from .mathcheck import answers_equivalent, extract_final_answer
from .report import ItemResult, MeasurementScope, Metric, TaskResult
from .sandbox import (
    CodeExecutionRefusedError,
    CodeSandbox,
    SandboxIsolation,
    require_isolation,
)
from .scoring import ScoringModel, truncate_at_stop

ItemId = Annotated[str, Field(min_length=1, max_length=200)]
Normalization = Literal["strip", "lower", "collapse_whitespace", "strip_punctuation"]
DEFAULT_NORMALIZATION: tuple[Normalization, ...] = ("strip", "lower", "collapse_whitespace")
CODE_STOP_SEQUENCES = ("\ndef ", "\nclass ", "\nif __name__", "\nprint(", "\n#")


class _Item(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class MultipleChoiceItem(_Item):
    id: ItemId
    question: str = Field(min_length=1)
    choices: list[str] = Field(min_length=2, max_length=26)
    answer: int = Field(ge=0)

    @model_validator(mode="after")
    def answer_in_range(self) -> MultipleChoiceItem:
        if self.answer >= len(self.choices):
            raise ValueError("answer index is outside the choices")
        if any(not choice for choice in self.choices):
            raise ValueError("choices must be non-empty")
        return self


class ExactMatchItem(_Item):
    id: ItemId
    prompt: str = Field(min_length=1)
    answers: list[str] = Field(min_length=1)


class MathItem(_Item):
    id: ItemId
    problem: str = Field(min_length=1)
    answer: str = Field(min_length=1, max_length=256)


class CodeItem(_Item):
    id: ItemId
    prompt: str = Field(min_length=1)
    test: str = Field(min_length=1)
    entry_point: str | None = Field(default=None, pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")


TaskItem = MultipleChoiceItem | ExactMatchItem | MathItem | CodeItem


def _mean(values: Sequence[float]) -> float | None:
    return math.fsum(values) / len(values) if values else None


def _require_unique(items: Sequence[TaskItem]) -> None:
    ids = [item.id for item in items]
    if len(ids) != len(set(ids)):
        raise ValueError("task item IDs must be unique")
    if not ids:
        raise ValueError("a task needs at least one item")


# ------------------------------------------------------------------ multiple choice


def mc_context(item: MultipleChoiceItem) -> str:
    return f"Question: {item.question}\nAnswer:"


def run_multiple_choice(
    task_id: str,
    items: Sequence[MultipleChoiceItem],
    model: ScoringModel,
    *,
    scope: MeasurementScope,
    calibration_bins: int = DEFAULT_CALIBRATION_BINS,
) -> TaskResult:
    _require_unique(items)
    requests = [(mc_context(item), f" {choice}") for item in items for choice in item.choices]
    scored = model.loglikelihood_batch(requests)
    position = 0
    results: list[ItemResult] = []
    raw_correct: list[float] = []
    norm_correct: list[float] = []
    distributions: list[list[float]] = []
    for item in items:
        values = scored[position : position + len(item.choices)]
        position += len(item.choices)
        raw = [value.sum_logprob for value in values]
        if not all(math.isfinite(score) for score in raw):
            raise ValueError(f"non-finite log-likelihood for item {item.id}")
        normalized = [
            score / len(f" {choice}".encode())
            for score, choice in zip(raw, item.choices, strict=True)
        ]
        raw_choice = raw.index(max(raw))
        norm_choice = normalized.index(max(normalized))
        probabilities = softmax(normalized)
        distributions.append(probabilities)
        raw_correct.append(float(raw_choice == item.answer))
        norm_correct.append(float(norm_choice == item.answer))
        results.append(
            ItemResult(
                item_id=item.id,
                score=float(norm_choice == item.answer),
                details={
                    "answer": item.answer,
                    "prediction_norm": norm_choice,
                    "prediction_raw": raw_choice,
                    "logprobs": raw,
                    "logprobs_per_byte": normalized,
                    "probabilities": probabilities,
                    "greedy": [value.is_greedy for value in values],
                },
            )
        )
    calibration = calibration_summary(
        distributions,
        [item.answer for item in items],
        bins=calibration_bins,
        probability_source="softmax over per-byte choice log-likelihoods",
    )
    metrics = [
        Metric(name="accuracy_norm", value=_mean(norm_correct), scope=scope, higher_is_better=True),
        Metric(name="accuracy", value=_mean(raw_correct), scope=scope, higher_is_better=True),
        Metric(
            name="ece",
            value=calibration.expected_calibration_error,
            scope=scope,
            higher_is_better=False,
        ),
        Metric(name="brier", value=calibration.brier_score, scope=scope, higher_is_better=False),
        Metric(name="items", value=float(len(items)), scope=scope, higher_is_better=True),
    ]
    return TaskResult(
        task_id=task_id,
        task_type="multiple-choice",
        format="mc-v1",
        primary_metric="accuracy_norm",
        metrics=metrics,
        items=results,
        settings={"normalization": "utf8-bytes-of-continuation", "bins": calibration_bins},
        calibration=calibration,
    )


# ------------------------------------------------------------------ exact match

_PUNCTUATION = str.maketrans("", "", string.punctuation)


def normalize_text(text: str, steps: Sequence[Normalization]) -> str:
    value = text
    for step in steps:
        if step == "strip":
            value = value.strip()
        elif step == "lower":
            value = value.lower()
        elif step == "collapse_whitespace":
            value = re.sub(r"\s+", " ", value).strip()
        elif step == "strip_punctuation":
            value = value.translate(_PUNCTUATION)
    return value


def run_exact_match(
    task_id: str,
    items: Sequence[ExactMatchItem],
    model: ScoringModel,
    *,
    scope: MeasurementScope,
    max_new_tokens: int = 32,
    stop: Sequence[str] = ("\n",),
    normalization: Sequence[Normalization] = DEFAULT_NORMALIZATION,
) -> TaskResult:
    _require_unique(items)
    results: list[ItemResult] = []
    for item in items:
        output = model.generate(item.prompt, max_new_tokens=max_new_tokens, stop=stop)
        prediction = normalize_text(output, normalization)
        references = [normalize_text(answer, normalization) for answer in item.answers]
        correct = prediction in references
        results.append(
            ItemResult(
                item_id=item.id,
                score=float(correct),
                details={"output": output, "prediction": prediction, "references": references},
            )
        )
    return TaskResult(
        task_id=task_id,
        task_type="exact-match",
        format="exact-match-v1",
        primary_metric="exact_match",
        metrics=[
            Metric(
                name="exact_match",
                value=_mean([r.score for r in results]),
                scope=scope,
                higher_is_better=True,
            ),
            Metric(name="items", value=float(len(items)), scope=scope, higher_is_better=True),
        ],
        items=results,
        settings={
            "max_new_tokens": max_new_tokens,
            "stop": list(stop),
            "normalization": list(normalization),
        },
    )


# ------------------------------------------------------------------ math


def math_prompt(item: MathItem) -> str:
    return f"Question: {item.problem}\nAnswer:"


def run_math(
    task_id: str,
    items: Sequence[MathItem],
    model: ScoringModel,
    *,
    scope: MeasurementScope,
    max_new_tokens: int = 64,
    stop: Sequence[str] = ("\n\n", "Question:"),
) -> TaskResult:
    _require_unique(items)
    results: list[ItemResult] = []
    for item in items:
        output = model.generate(math_prompt(item), max_new_tokens=max_new_tokens, stop=stop)
        extracted = extract_final_answer(output)
        correct = extracted is not None and answers_equivalent(extracted, item.answer)
        results.append(
            ItemResult(
                item_id=item.id,
                score=float(correct),
                details={"output": output, "extracted": extracted, "reference": item.answer},
            )
        )
    return TaskResult(
        task_id=task_id,
        task_type="math",
        format="math-v1",
        primary_metric="equivalent",
        metrics=[
            Metric(
                name="equivalent",
                value=_mean([r.score for r in results]),
                scope=scope,
                higher_is_better=True,
            ),
            Metric(
                name="answer_extracted",
                value=_mean([float(r.details["extracted"] is not None) for r in results]),
                scope=scope,
                higher_is_better=True,
            ),
            Metric(name="items", value=float(len(items)), scope=scope, higher_is_better=True),
        ],
        items=results,
        settings={"max_new_tokens": max_new_tokens, "stop": list(stop), "checker": "sympy-safe-v1"},
    )


# ------------------------------------------------------------------ code


def code_program(item: CodeItem, completion: str) -> str:
    program = f"{item.prompt}{completion}\n\n{item.test}\n"
    if item.entry_point is not None:
        program += f"\ncheck({item.entry_point})\n"
    return program


def run_code(
    task_id: str,
    items: Sequence[CodeItem],
    model: ScoringModel,
    *,
    scope: MeasurementScope,
    allow_code_execution: bool = False,
    allow_unisolated_network: bool = False,
    sandbox: CodeSandbox | None = None,
    max_new_tokens: int = 128,
    stop: Sequence[str] = CODE_STOP_SEQUENCES,
) -> TaskResult:
    """Generate completions and run the item tests in the sandbox (opt-in only).

    Refuses before generating anything unless ``allow_code_execution`` is
    ``True``; if the sandbox reports no network isolation it also requires
    ``allow_unisolated_network``.  Isolation is probed once up front and then
    checked again on every run.
    """

    if not allow_code_execution:
        raise CodeExecutionRefusedError(
            "code-generation scoring executes model output; it is disabled unless "
            "allow_code_execution=True is passed explicitly"
        )
    _require_unique(items)
    runner = sandbox or CodeSandbox()
    isolation: SandboxIsolation = runner.probe()
    require_isolation(isolation, allow_unisolated_network=allow_unisolated_network)
    results: list[ItemResult] = []
    for item in items:
        completion = model.generate(item.prompt, max_new_tokens=max_new_tokens, stop=())
        completion = truncate_at_stop(completion, stop)[0]
        outcome = runner.run(code_program(item, completion))
        require_isolation(outcome.isolation, allow_unisolated_network=allow_unisolated_network)
        if outcome.isolation != isolation:
            raise CodeExecutionRefusedError("sandbox isolation changed during the task")
        results.append(
            ItemResult(
                item_id=item.id,
                score=float(outcome.status == "passed"),
                details={
                    "completion": completion,
                    "status": outcome.status,
                    "returncode": outcome.returncode,
                    "signal": outcome.signal_name,
                    "stderr_tail": outcome.stderr[-2_000:],
                },
            )
        )
    return TaskResult(
        task_id=task_id,
        task_type="code",
        format="code-v1",
        primary_metric="pass_at_1",
        metrics=[
            Metric(
                name="pass_at_1",
                value=_mean([r.score for r in results]),
                scope=scope,
                higher_is_better=True,
            ),
            Metric(name="items", value=float(len(items)), scope=scope, higher_is_better=True),
        ],
        items=results,
        settings={
            "max_new_tokens": max_new_tokens,
            "stop": list(stop),
            "decoding": "greedy",
            "limits": runner.limits.model_dump(mode="json"),
        },
        sandbox=isolation,
    )


# ------------------------------------------------------------------ validation loss


def run_validation_loss(
    task_id: str,
    checkpoint: Path,
    data_root: Path,
    *,
    scope: MeasurementScope,
    split: Literal["train", "validation", "test"] = "validation",
    max_steps: int | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> TaskResult:
    """Wrap :func:`epor.training.evaluate_packed_checkpoint` as a suite task."""

    from epor.training.kernel import evaluate_packed_checkpoint

    result = evaluate_packed_checkpoint(
        checkpoint, data_root, split=split, max_steps=max_steps, should_cancel=should_cancel
    )
    evaluation = result.evaluation
    metrics = [
        Metric(name="main_loss", value=evaluation.main_loss, scope=scope, higher_is_better=False),
        Metric(name="perplexity", value=evaluation.perplexity, scope=scope, higher_is_better=False),
        Metric(
            name="scored_positions",
            value=float(evaluation.target_tokens),
            scope=scope,
            higher_is_better=True,
        ),
    ]
    for depth, loss in enumerate(evaluation.mtp_losses, start=1):
        metrics.append(
            Metric(
                name=f"mtp_loss_d{depth}",
                value=loss if math.isfinite(loss) else None,
                scope=scope,
                higher_is_better=False,
            )
        )
    return TaskResult(
        task_id=task_id,
        task_type="validation-loss",
        format="packed-validation-v1",
        primary_metric="main_loss",
        metrics=metrics,
        settings={"split": split, "max_steps": max_steps, "batches": evaluation.steps},
    )


__all__ = [
    "CODE_STOP_SEQUENCES",
    "DEFAULT_NORMALIZATION",
    "CodeItem",
    "ExactMatchItem",
    "MathItem",
    "MultipleChoiceItem",
    "Normalization",
    "TaskItem",
    "code_program",
    "math_prompt",
    "mc_context",
    "normalize_text",
    "run_code",
    "run_exact_match",
    "run_math",
    "run_multiple_choice",
    "run_validation_loss",
]
