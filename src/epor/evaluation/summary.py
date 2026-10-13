"""Torch-free summaries and verified listings of stored evaluation reports.

Used by the CLI, control-job results, and the read-only console endpoints.
Every listed report is re-verified against its content address
(:func:`~epor.evaluation.report.load_report`); one that fails is named in
``integrity_errors`` instead of being listed.

The console view drops per-item ``details`` (generated text, prompts, and
benchmark item text) and keeps only item counts: the console shows metrics,
calibration bins, and settings, and the full report stays on disk for review.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from epor.data.store import DataStore, DataStoreError

from .report import EvaluationReport, ReportError, load_report

MAX_LISTED_REPORTS = 200
MAX_REPORT_BYTES = 64 * 1024 * 1024


def primary_metrics(report: EvaluationReport) -> list[dict[str, Any]]:
    """Each task's primary metric with its scope (``value`` is ``None`` when undefined)."""

    rows: list[dict[str, Any]] = []
    for task in report.tasks:
        try:
            metric = task.metric(task.primary_metric)
        except KeyError:
            value, scope = None, report.scope
        else:
            value, scope = metric.value, metric.scope
        rows.append(
            {
                "task_id": task.task_id,
                "task_type": task.task_type,
                "metric": task.primary_metric,
                "value": value,
                "scope": scope,
            }
        )
    return rows


def report_summary(report: EvaluationReport) -> dict[str, Any]:
    """A small fixed summary; its keys are safe for control-job results."""

    return {
        "report_id": report.report_id,
        "suite_id": report.suite_id,
        "suite_version": report.suite_version,
        "scope": report.scope,
        "claim_boundary": report.claim_boundary,
        "model_kind": report.model.kind,
        "model_id": report.model.model_id,
        "checkpoint_step": report.model.checkpoint_step,
        "parameter_count": report.model.parameter_count,
        "task_count": len(report.tasks),
        "skipped_task_count": len(report.skipped_tasks),
        "primary_metrics": primary_metrics(report),
        "code_executed": any(task.sandbox is not None for task in report.tasks),
    }


def report_view(report: EvaluationReport) -> dict[str, Any]:
    """The verified report without per-item details, plus each task's item count."""

    payload = report.model_dump(mode="json")
    for task in payload["tasks"]:
        task["item_count"] = len(task.pop("items"))
    payload["summary"] = report_summary(report)
    return payload


def _load_bounded(store: DataStore, report_id: str) -> EvaluationReport:
    try:
        path = store.evaluation_report_path(report_id)
    except DataStoreError as exc:
        raise ReportError(str(exc)) from exc
    if path.is_file() and path.stat().st_size > MAX_REPORT_BYTES:
        raise ReportError("evaluation report exceeds the listing size limit")
    return load_report(store, report_id)


def load_verified_report(data_root: str | Path, report_id: str) -> EvaluationReport:
    """One stored report, verified and size-bounded (raises :class:`ReportError`)."""

    return _load_bounded(DataStore(Path(data_root)), report_id)


def list_reports(data_root: str | Path) -> dict[str, Any]:
    """Summaries of every stored report that verifies, and what did not."""

    store = DataStore(Path(data_root))
    identifiers = store.evaluation_report_ids()
    reports: list[dict[str, Any]] = []
    errors: list[str] = []
    for report_id in identifiers[:MAX_LISTED_REPORTS]:
        try:
            reports.append(report_summary(_load_bounded(store, report_id)))
        except (ReportError, OSError) as exc:
            errors.append(f"{report_id}: {exc}")
    if len(identifiers) > MAX_LISTED_REPORTS:
        errors.append(
            f"{len(identifiers) - MAX_LISTED_REPORTS} further report(s) not listed "
            f"(limit {MAX_LISTED_REPORTS})"
        )
    return {"reports": reports, "integrity_errors": errors}


__all__ = [
    "MAX_LISTED_REPORTS",
    "MAX_REPORT_BYTES",
    "list_reports",
    "load_verified_report",
    "primary_metrics",
    "report_summary",
    "report_view",
]
