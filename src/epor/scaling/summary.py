"""Torch-free summaries and verified listings of stored scaling reports.

Used by the CLI, control-job results, and the read-only console endpoints.
Every stored measurement is re-verified against both of its addresses (the
report ID over the content and the file name over the full bytes) by
:func:`~epor.scaling.report.load_scaling_reports`; anything that fails is
named in ``integrity_errors`` instead of being listed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from epor.data.store import DataStore

from .ladder import PROJECT_ROOT, derive_rung, load_ladder, rung_status
from .report import ScalingReport, ScalingReportError, load_scaling_reports

MAX_LISTED_REPORTS = 200
MAX_REPORT_BYTES = 64 * 1024 * 1024


def _headline(report: ScalingReport) -> dict[str, Any]:
    results = report.results
    measured = report.measured or {}
    if report.kind == "hardware-profile":
        model = report.inputs.get("model", {})
        return {
            "model_slug": model.get("slug"),
            "config_sha256": model.get("config_sha256"),
            "tokens_per_second": measured.get("tokens_per_second"),
            "mfu": measured.get("mfu"),
            "mfu_status": measured.get("mfu_status", "unknown"),
        }
    if report.kind == "scaling-sweep":
        counts = results.get("counts", {})
        return {
            "points": len(results.get("points", [])),
            "completed_points": counts.get("completed", 0),
            "failed_points": counts.get("failed", 0),
            "projection_only_points": counts.get("projection-only", 0),
        }
    if report.kind == "mixture-ablation":
        return {"mixtures": len(results.get("mixtures", []))}
    if report.kind == "scaling-fit":
        return {
            "fit_status": {
                part: results.get(part, {}).get("status")
                for part in ("loss_vs_parameters", "loss_vs_tokens", "joint")
            }
        }
    target = results.get("target", {})
    return {
        "target_slug": target.get("slug"),
        "local_execution": target.get("local_execution"),
        "unknown_input_count": len(results.get("unknown_inputs", [])),
        "approval_required": bool(results.get("approval", {}).get("required", True)),
    }


def scaling_summary(report: ScalingReport) -> dict[str, Any]:
    """A small fixed summary; its keys are safe for control-job results."""

    return {
        "report_id": report.report_id,
        "content_sha256": report.content_sha256,
        "kind": report.kind,
        "scope": report.scope,
        "claim_boundary": report.claim_boundary,
        "warning_count": len(report.warnings),
        "recorded_at": (report.measured or {}).get("recorded_at"),
        **_headline(report),
    }


def _bounded_reports(store: DataStore, report_id: str) -> list[ScalingReport]:
    directory = store.scaling_report_directory(report_id)
    for path in directory.glob("*.json"):
        if path.stat().st_size > MAX_REPORT_BYTES:
            raise ScalingReportError(f"{path.name} exceeds the listing size limit")
    return load_scaling_reports(store, report_id)


def load_verified_scaling_reports(data_root: str | Path, report_id: str) -> list[ScalingReport]:
    """Every stored measurement of one report ID, verified and size-bounded."""

    from epor.data.store import DataStoreError

    store = DataStore(Path(data_root))
    try:
        return _bounded_reports(store, report_id)
    except DataStoreError as exc:
        raise ScalingReportError(str(exc)) from exc


def list_scaling_reports(data_root: str | Path) -> dict[str, Any]:
    """Summaries of every stored measurement that verifies, and what did not."""

    from epor.data.store import DataStoreError

    store = DataStore(Path(data_root))
    identifiers = store.scaling_report_ids()
    reports: list[dict[str, Any]] = []
    errors: list[str] = []
    for report_id in identifiers[:MAX_LISTED_REPORTS]:
        try:
            reports += [scaling_summary(item) for item in _bounded_reports(store, report_id)]
        except (ScalingReportError, DataStoreError, OSError) as exc:
            errors.append(f"{report_id}: {exc}")
    if len(identifiers) > MAX_LISTED_REPORTS:
        errors.append(
            f"{len(identifiers) - MAX_LISTED_REPORTS} further report(s) not listed "
            f"(limit {MAX_LISTED_REPORTS})"
        )
    return {"reports": reports, "integrity_errors": errors}


def ladder_table(
    vocab_size: int | None = None, project_root: str | Path = PROJECT_ROOT
) -> list[dict[str, Any]]:
    """Each shipped rung's counts and local or projection-only status."""

    return [
        rung_status(derive_rung(config, vocab_size)).model_dump(mode="json")
        for config in load_ladder(project_root)
    ]


__all__ = [
    "MAX_LISTED_REPORTS",
    "MAX_REPORT_BYTES",
    "ladder_table",
    "list_scaling_reports",
    "load_verified_scaling_reports",
    "scaling_summary",
]
