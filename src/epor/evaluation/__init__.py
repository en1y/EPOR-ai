"""EPOR evaluation harness: suites, tasks, calibration, sandboxed code scoring, reports.

Modules
-------
:mod:`~epor.evaluation.report`
    Sealed, content-addressed :class:`EvaluationReport` (stored under
    ``<data-root>/evaluation-reports/<report_id>.json``), verification, and
    :func:`compare_reports` (metric deltas + seeded paired bootstrap).
:mod:`~epor.evaluation.scoring`
    The :class:`ScoringModel` interface (log-likelihood, greedy generation).
:mod:`~epor.evaluation.decoder`
    Adapters over ``EporDecoder``: kernel checkpoints and the debug byte
    tokenizer path.  Imports torch, so it is not re-exported here.
:mod:`~epor.evaluation.tasks`
    Multiple choice (+calibration), exact match, math (safe SymPy), code
    (sandbox, opt-in), validation loss.
:mod:`~epor.evaluation.calibration`
    Expected calibration error, Brier score, and reliability bins.
:mod:`~epor.evaluation.mathcheck`
    ``eval``-free answer parsing and equivalence.
:mod:`~epor.evaluation.sandbox`
    Defense-in-depth subprocess sandbox (not a security boundary).
:mod:`~epor.evaluation.sources`
    Adapters for operator-registered external benchmark files.
:mod:`~epor.evaluation.suites`
    YAML suite definitions and :func:`run_suite`.

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
from .scoring import Loglikelihood, ScoringModel
from .sources import SourceRefusedError, load_registered_items
from .suites import (
    EvaluationCancelledError,
    SuiteDefinition,
    fixture_suite,
    load_suite,
    run_suite,
)

__all__ = [
    "CalibrationSummary",
    "CodeExecutionRefusedError",
    "CodeSandbox",
    "DataSource",
    "EvaluationCancelledError",
    "EvaluationReport",
    "ItemResult",
    "Loglikelihood",
    "MeasurementScope",
    "Metric",
    "ModelIdentity",
    "ReliabilityBin",
    "ReportComparison",
    "ReportError",
    "SandboxIsolation",
    "SandboxLimits",
    "SandboxResult",
    "ScoringModel",
    "SourceRefusedError",
    "SuiteDefinition",
    "TaskResult",
    "calibration_summary",
    "compare_reports",
    "fixture_suite",
    "load_registered_items",
    "load_report",
    "load_suite",
    "paired_bootstrap",
    "run_suite",
    "seal_report",
    "store_report",
    "verify_report",
]
