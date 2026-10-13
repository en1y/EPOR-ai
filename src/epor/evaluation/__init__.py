"""EPOR evaluation harness: reports, calibration, safe math checking, code sandbox.

Modules
-------
:mod:`~epor.evaluation.report`
    Sealed, content-addressed :class:`EvaluationReport` (stored under
    ``<data-root>/evaluation-reports/<report_id>.json``), verification, and
    :func:`compare_reports` (metric deltas + seeded paired bootstrap).
:mod:`~epor.evaluation.calibration`
    Expected calibration error, Brier score, and reliability bins.
:mod:`~epor.evaluation.mathcheck`
    ``eval``-free answer parsing and equivalence.
:mod:`~epor.evaluation.sandbox`
    Defense-in-depth subprocess sandbox (not a security boundary).

All results are scoped ``harness-fixture`` or ``proxy-scale``; none is a
capability claim about a release-scale EPOR model.
"""

from .calibration import CalibrationSummary, ReliabilityBin, calibration_summary
from .report import (
    DataSource,
    EvaluationReport,
    ItemResult,
    MeasurementScope,
    Metric,
    ModelIdentity,
    ReportComparison,
    ReportError,
    TaskResult,
    compare_reports,
    load_report,
    paired_bootstrap,
    seal_report,
    store_report,
    verify_report,
)
from .sandbox import (
    CodeExecutionRefusedError,
    CodeSandbox,
    SandboxIsolation,
    SandboxLimits,
    SandboxResult,
)

__all__ = [
    "CalibrationSummary",
    "CodeExecutionRefusedError",
    "CodeSandbox",
    "DataSource",
    "EvaluationReport",
    "ItemResult",
    "MeasurementScope",
    "Metric",
    "ModelIdentity",
    "ReliabilityBin",
    "ReportComparison",
    "ReportError",
    "SandboxIsolation",
    "SandboxLimits",
    "SandboxResult",
    "TaskResult",
    "calibration_summary",
    "compare_reports",
    "load_report",
    "paired_bootstrap",
    "seal_report",
    "store_report",
    "verify_report",
]
