"""Canonical, content-addressed evaluation reports and their deterministic comparison.

A report is sealed by :func:`seal_report`: its ``report_id`` is the SHA-256 of
the canonical JSON (sorted keys, compact separators, UTF-8) of every other
field.  Reports carry no wall-clock timestamps or durations, so identical
inputs (suite, model, checkpoint, tokenizer, data, harness revision, seeds,
settings) give an identical ``report_id``.

Storage: ``<data-root>/evaluation-reports/<report_id>.json`` (written once,
immutably, via :meth:`epor.data.store.DataStore.write_evaluation_report`).
:func:`load_report` re-derives the ID and refuses a file whose bytes, sealed
ID, or file name disagree — any edit is detected.

Every metric carries a ``scope``.  ``harness-fixture`` results only show the
harness works; ``proxy-scale`` results are measurements of a small proxy
model on the stated data and are not capability claims about any EPOR
release-scale model.
"""

from __future__ import annotations

import math
import random
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from epor.data.models import Identifier, Sha256
from epor.data.store import DataStore, DataStoreError, canonical_json_bytes, sha256_bytes

from .calibration import CalibrationSummary
from .sandbox import SandboxIsolation

REPORT_SCHEMA = "epor-evaluation-report-v1"
HARNESS_VERSION = "epor-eval-harness-v1"
PROJECT_ROOT = Path(__file__).resolve().parents[3]

MeasurementScope = Literal["harness-fixture", "proxy-scale"]
TaskType = Literal[
    "multiple-choice",
    "exact-match",
    "math",
    "code",
    "validation-loss",
    "safety-preference",
    "safety-rubric",
    "contamination",
    "memorization",
    "canary-exposure",
]
JsonScalar = str | int | float | bool | None

CLAIM_BOUNDARIES: dict[str, str] = {
    "harness-fixture": (
        "Harness-fixture result: project-authored items that exercise the evaluation "
        "code path. Not a benchmark and not evidence of any model capability."
    ),
    "proxy-scale": (
        "Proxy-scale measurement of the identified checkpoint on the identified data. "
        "Not a capability claim about any release-scale EPOR model."
    ),
}


class ReportError(ValueError):
    """A report is malformed, tampered with, or not comparable as requested."""


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _finite(value: float | None) -> float | None:
    if value is not None and not math.isfinite(value):
        raise ValueError("report numbers must be finite (use null for undefined)")
    return value


class Metric(_Frozen):
    name: Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9_.-]{0,63}$")]
    value: float | None
    scope: MeasurementScope
    higher_is_better: bool

    @field_validator("value")
    @classmethod
    def value_is_finite(cls, value: float | None) -> float | None:
        return _finite(value)


class ItemResult(_Frozen):
    """One scored item; ``score`` is the task's primary per-item metric."""

    item_id: Annotated[str, Field(min_length=1, max_length=200)]
    score: float
    details: dict[str, Any] = Field(default_factory=dict)

    @field_validator("score")
    @classmethod
    def score_is_finite(cls, value: float) -> float:
        _finite(value)
        return value


class TaskResult(_Frozen):
    task_id: Identifier
    task_type: TaskType
    format: str = Field(min_length=1, max_length=80)
    primary_metric: str
    metrics: list[Metric]
    items: list[ItemResult] = Field(default_factory=list)
    settings: dict[str, Any] = Field(default_factory=dict)
    calibration: CalibrationSummary | None = None
    sandbox: SandboxIsolation | None = None

    def metric(self, name: str) -> Metric:
        for metric in self.metrics:
            if metric.name == name:
                return metric
        raise KeyError(name)


class ModelIdentity(_Frozen):
    """What was evaluated: config and checkpoint hashes, tokenizer, scoring settings.

    ``data-only`` marks a report that scored no model (a contamination scan of
    evaluation items against a training build).
    """

    kind: Literal["kernel-checkpoint", "in-memory-decoder", "stub", "data-only"]
    model_id: str = Field(min_length=1, max_length=200)
    config_sha256: Sha256 | None = None
    checkpoint_sha256: Sha256 | None = None
    checkpoint_step: int | None = Field(default=None, ge=0)
    parameter_count: int | None = Field(default=None, ge=0)
    tokenizer_id: str = Field(min_length=1, max_length=128)
    tokenizer_kind: str = Field(min_length=1, max_length=80)
    scoring: dict[str, JsonScalar] = Field(default_factory=dict)


class DataSource(_Frozen):
    """Lineage of one input: a project fixture, a registered source, or a build."""

    kind: Literal["project-fixture", "registered-source", "tokenized-build"]
    source_id: str = Field(min_length=1, max_length=200)
    content_sha256: Sha256
    registration_sha256: Sha256 | None = None
    format: str | None = Field(default=None, max_length=80)
    task_ids: list[str] = Field(default_factory=list)


class HarnessIdentity(_Frozen):
    harness_version: Literal["epor-eval-harness-v1"] = HARNESS_VERSION
    code_revision: str | None
    dependency_lock_sha256: Sha256 | None


class SkippedTask(_Frozen):
    """A suite task that did not run in this invocation, and why."""

    task_id: Identifier
    reason: str = Field(min_length=1, max_length=200)


class EvaluationReport(_Frozen):
    schema_version: Literal["epor-evaluation-report-v1"] = REPORT_SCHEMA
    report_id: Sha256
    suite_id: Identifier
    suite_version: int = Field(ge=1)
    suite_sha256: Sha256
    scope: MeasurementScope
    claim_boundary: str
    model: ModelIdentity
    data: list[DataSource]
    seeds: dict[str, int]
    harness: HarnessIdentity
    tasks: list[TaskResult]
    skipped_tasks: list[SkippedTask] = Field(default_factory=list)

    def task(self, task_id: str) -> TaskResult:
        for task in self.tasks:
            if task.task_id == task_id:
                return task
        raise KeyError(task_id)


_UNSEALED = "0" * 64


def report_digest(report: EvaluationReport) -> str:
    """SHA-256 of the canonical JSON of every field except ``report_id``."""

    return sha256_bytes(canonical_json_bytes(report.model_dump(mode="json", exclude={"report_id"})))


def seal_report(**fields: Any) -> EvaluationReport:
    """Build a report from its fields and fill in the content-derived ``report_id``."""

    draft = EvaluationReport.model_validate({**fields, "report_id": _UNSEALED})
    return draft.model_copy(update={"report_id": report_digest(draft)})


def report_bytes(report: EvaluationReport) -> bytes:
    return canonical_json_bytes(report)


def verify_report(report: EvaluationReport) -> None:
    if report.report_id != report_digest(report):
        raise ReportError("evaluation report ID does not match its content")


def store_report(store: DataStore, report: EvaluationReport) -> Path:
    """Write a sealed report immutably; rewriting identical bytes is a no-op."""

    verify_report(report)
    return store.write_evaluation_report(report.report_id, report_bytes(report))


def parse_report(payload: bytes, *, expected_id: str | None = None) -> EvaluationReport:
    try:
        report = EvaluationReport.model_validate_json(payload)
    except ValueError as exc:
        raise ReportError(f"evaluation report is malformed: {exc}") from exc
    verify_report(report)
    if payload != report_bytes(report):
        raise ReportError("evaluation report bytes are not canonical")
    if expected_id is not None and report.report_id != expected_id:
        raise ReportError("evaluation report is stored under the wrong ID")
    return report


def load_report(store: DataStore, report_id: str) -> EvaluationReport:
    """Read and verify a stored report (tampering raises :class:`ReportError`)."""

    try:
        path = store.evaluation_report_path(report_id)
    except DataStoreError as exc:
        raise ReportError(str(exc)) from exc
    try:
        payload = path.read_bytes()
    except FileNotFoundError as exc:
        raise ReportError(f"evaluation report not found: {report_id}") from exc
    return parse_report(payload, expected_id=report_id)


def code_revision(root: Path = PROJECT_ROOT) -> str | None:
    """The harness Git revision (``HEAD``), or ``None`` outside a checkout."""

    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def harness_identity(root: Path = PROJECT_ROOT) -> HarnessIdentity:
    lock = root / "uv.lock"
    lock_sha256 = sha256_bytes(lock.read_bytes()) if lock.is_file() else None
    return HarnessIdentity(code_revision=code_revision(root), dependency_lock_sha256=lock_sha256)


# --------------------------------------------------------------------------- comparison


class MetricDelta(_Frozen):
    task_id: str
    name: str
    baseline: float | None
    candidate: float | None
    delta: float | None
    higher_is_better: bool


class PairedBootstrap(_Frozen):
    """Percentile CI of the mean per-item score difference (candidate - baseline)."""

    task_id: str
    paired_items: int = Field(ge=0)
    unmatched_items: int = Field(ge=0)
    baseline_mean: float | None
    candidate_mean: float | None
    mean_delta: float | None
    ci_lower: float | None
    ci_upper: float | None
    confidence: float
    resamples: int
    seed: int


class ReportComparison(_Frozen):
    schema_version: Literal["epor-evaluation-comparison-v1"] = "epor-evaluation-comparison-v1"
    baseline_report_id: Sha256
    candidate_report_id: Sha256
    warnings: list[str]
    metric_deltas: list[MetricDelta]
    bootstraps: list[PairedBootstrap]


def paired_bootstrap(
    baseline: Sequence[float],
    candidate: Sequence[float],
    *,
    resamples: int = 1_000,
    confidence: float = 0.95,
    seed: int = 0,
    label: str = "",
) -> tuple[float, float, float]:
    """``(mean_delta, lower, upper)`` for paired scores, deterministic given ``seed``/``label``.

    Resamples item indices with replacement from ``random.Random`` seeded with
    the string ``"{seed}:{label}"`` (string seeds hash with SHA-512, so the
    stream is independent of ``PYTHONHASHSEED``).
    """

    if len(baseline) != len(candidate) or not baseline:
        raise ValueError("paired bootstrap needs equal-length, non-empty score lists")
    if resamples < 1 or not 0 < confidence < 1:
        raise ValueError("bootstrap needs resamples >= 1 and 0 < confidence < 1")
    deltas = [c - b for b, c in zip(baseline, candidate, strict=True)]
    count = len(deltas)
    rng = random.Random(f"{seed}:{label}")
    means = sorted(
        math.fsum(deltas[rng.randrange(count)] for _ in range(count)) / count
        for _ in range(resamples)
    )
    alpha = (1.0 - confidence) / 2.0
    lower = means[max(0, math.floor(alpha * resamples))]
    upper = means[min(resamples - 1, math.ceil((1.0 - alpha) * resamples) - 1)]
    return math.fsum(deltas) / count, lower, upper


def compare_reports(
    baseline: EvaluationReport,
    candidate: EvaluationReport,
    *,
    resamples: int = 1_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> ReportComparison:
    """Metric deltas for shared tasks plus a paired bootstrap over shared items."""

    verify_report(baseline)
    verify_report(candidate)
    warnings: list[str] = []
    if (baseline.suite_id, baseline.suite_version) != (candidate.suite_id, candidate.suite_version):
        warnings.append("reports come from different suites or suite versions")
    elif baseline.suite_sha256 != candidate.suite_sha256:
        warnings.append("suite definitions differ although the suite version is the same")
    if baseline.model.tokenizer_id != candidate.model.tokenizer_id:
        warnings.append(
            "tokenizers differ: log-likelihood and perplexity are not directly comparable"
        )
    if baseline.scope != candidate.scope:
        warnings.append("measurement scopes differ")
    if baseline.harness.code_revision != candidate.harness.code_revision:
        warnings.append("harness code revisions differ")
    if {(s.source_id, s.content_sha256) for s in baseline.data if s.kind != "tokenized-build"} != {
        (s.source_id, s.content_sha256) for s in candidate.data if s.kind != "tokenized-build"
    }:
        warnings.append("evaluation data differs")

    candidate_tasks = {task.task_id: task for task in candidate.tasks}
    shared = [task for task in baseline.tasks if task.task_id in candidate_tasks]
    only = sorted({t.task_id for t in baseline.tasks} ^ {t.task_id for t in candidate.tasks})
    if only:
        warnings.append("tasks present in only one report: " + ", ".join(only))
    deltas: list[MetricDelta] = []
    bootstraps: list[PairedBootstrap] = []
    for base_task in shared:
        cand_task = candidate_tasks[base_task.task_id]
        cand_metrics = {metric.name: metric for metric in cand_task.metrics}
        for metric in base_task.metrics:
            other = cand_metrics.get(metric.name)
            if other is None:
                continue
            delta = (
                other.value - metric.value
                if metric.value is not None and other.value is not None
                else None
            )
            deltas.append(
                MetricDelta(
                    task_id=base_task.task_id,
                    name=metric.name,
                    baseline=metric.value,
                    candidate=other.value,
                    delta=delta,
                    higher_is_better=metric.higher_is_better,
                )
            )
        base_items = {item.item_id: item.score for item in base_task.items}
        cand_items = {item.item_id: item.score for item in cand_task.items}
        paired = sorted(set(base_items) & set(cand_items))
        unmatched = len(set(base_items) ^ set(cand_items))
        if not base_items and not cand_items:
            continue
        mean_delta: float | None = None
        lower: float | None = None
        upper: float | None = None
        base_mean: float | None = None
        cand_mean: float | None = None
        if paired:
            base_scores = [base_items[key] for key in paired]
            cand_scores = [cand_items[key] for key in paired]
            mean_delta, lower, upper = paired_bootstrap(
                base_scores,
                cand_scores,
                resamples=resamples,
                confidence=confidence,
                seed=seed,
                label=base_task.task_id,
            )
            base_mean = math.fsum(base_scores) / len(paired)
            cand_mean = math.fsum(cand_scores) / len(paired)
        bootstraps.append(
            PairedBootstrap(
                task_id=base_task.task_id,
                paired_items=len(paired),
                unmatched_items=unmatched,
                baseline_mean=base_mean,
                candidate_mean=cand_mean,
                mean_delta=mean_delta,
                ci_lower=lower,
                ci_upper=upper,
                confidence=confidence,
                resamples=resamples,
                seed=seed,
            )
        )
    return ReportComparison(
        baseline_report_id=baseline.report_id,
        candidate_report_id=candidate.report_id,
        warnings=warnings,
        metric_deltas=deltas,
        bootstraps=bootstraps,
    )


__all__ = [
    "CLAIM_BOUNDARIES",
    "HARNESS_VERSION",
    "REPORT_SCHEMA",
    "DataSource",
    "EvaluationReport",
    "HarnessIdentity",
    "ItemResult",
    "MeasurementScope",
    "Metric",
    "MetricDelta",
    "ModelIdentity",
    "PairedBootstrap",
    "ReportComparison",
    "ReportError",
    "SkippedTask",
    "TaskResult",
    "TaskType",
    "code_revision",
    "compare_reports",
    "harness_identity",
    "load_report",
    "paired_bootstrap",
    "parse_report",
    "report_bytes",
    "report_digest",
    "seal_report",
    "store_report",
    "verify_report",
]
