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
:mod:`~epor.evaluation.safety_behaviour`
    G10 model-behaviour items: paired preference and weak phrase rubric
    (``configs/evaluation/safety-behaviour.yaml``).
:mod:`~epor.evaluation.contamination`
    Text- and token-level n-gram overlap of evaluation items (or held-out
    splits) with training data; a lower bound.
:mod:`~epor.evaluation.memorization`
    Train vs held-out extraction rate and canary exposure.
:mod:`~epor.evaluation.suites`
    YAML suite definitions, :func:`run_suite`, and :func:`run_contamination`.

All results are scoped ``harness-fixture`` or ``proxy-scale``; none is a
capability claim about a release-scale EPOR model.
"""

from .calibration import CalibrationSummary, ReliabilityBin, calibration_summary
from .contamination import (
    ContaminationError,
    ContaminationItem,
    ContaminationSettings,
    TrainingCorpus,
    build_corpus,
    scan_contamination,
    scan_split_leakage,
)
from .memorization import (
    CanarySettings,
    MemorizationSettings,
    canary_exposure,
    generate_canary,
    memorization_from_build,
    run_canary_exposure,
    run_memorization,
)
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
from .safety_behaviour import (
    G10_CATEGORIES,
    SafetyPreferenceItem,
    SafetyRubricItem,
    run_safety_preference,
    run_safety_rubric,
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
    TrainingData,
    fixture_suite,
    load_suite,
    run_contamination,
    run_suite,
    safety_suite,
)

__all__ = [
    "G10_CATEGORIES",
    "CalibrationSummary",
    "CanarySettings",
    "CodeExecutionRefusedError",
    "CodeSandbox",
    "ContaminationError",
    "ContaminationItem",
    "ContaminationSettings",
    "DataSource",
    "EvaluationCancelledError",
    "EvaluationReport",
    "ItemResult",
    "Loglikelihood",
    "MeasurementScope",
    "MemorizationSettings",
    "Metric",
    "ModelIdentity",
    "ReliabilityBin",
    "ReportComparison",
    "ReportError",
    "SafetyPreferenceItem",
    "SafetyRubricItem",
    "SandboxIsolation",
    "SandboxLimits",
    "SandboxResult",
    "ScoringModel",
    "SourceRefusedError",
    "SuiteDefinition",
    "TaskResult",
    "TrainingCorpus",
    "TrainingData",
    "build_corpus",
    "calibration_summary",
    "canary_exposure",
    "compare_reports",
    "fixture_suite",
    "generate_canary",
    "load_registered_items",
    "load_report",
    "load_suite",
    "memorization_from_build",
    "paired_bootstrap",
    "run_canary_exposure",
    "run_contamination",
    "run_memorization",
    "run_safety_preference",
    "run_safety_rubric",
    "run_suite",
    "safety_suite",
    "scan_contamination",
    "scan_split_leakage",
    "seal_report",
    "store_report",
    "verify_report",
]
