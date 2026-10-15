"""Sealed, content-addressed scaling-laboratory reports.

Every scaling runner (hardware profile, sweep, mixture ablation, fit,
projection) returns one :class:`ScalingReport`.

Determinism decision
--------------------
A report has two parts:

* the **content** — ``kind``, ``scope``, ``claim_boundary``, ``harness``,
  ``inputs``, ``results``, ``warnings``, ``citations`` — which must be a pure
  function of the inputs, the code revision, and the dependency lock; and
* the optional **measured** section — wall-clock timings, throughput, RSS,
  achieved FLOP/s and MFU, host snapshot, run directories and random run IDs,
  and the recording timestamp — which legitimately differs between two
  executions of the same experiment.

``report_id`` is the SHA-256 of the canonical JSON of the content alone, so
identical inputs give an identical ``report_id`` while every measurement is
still kept.  Training losses are content: the kernel is bit-reproducible on
one host (``torch.use_deterministic_algorithms``), so a changed loss is a
changed experiment, not noise.

Storage: ``<data_root>/scaling-reports/<report_id>/<content_sha256>.json``
where ``content_sha256`` hashes the full canonical bytes (including
``measured``).  Files are written once, immutably; :func:`load_scaling_reports`
re-derives both hashes and refuses edited files.

Evidence labels follow the repository's research labels.  A published formula
is ``disclosed``; applying it to EPOR is ``hypothesis`` or ``EPOR-adaptation``;
a quantity whose operator input is missing is ``unknown``.  No report is a
trained or validated result for any release-scale EPOR model.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from epor.data.models import Sha256
from epor.data.store import DataStore, DataStoreError, canonical_json_bytes, sha256_bytes
from epor.evaluation.report import HarnessIdentity, harness_identity

SCALING_REPORT_SCHEMA = "epor-scaling-report-v1"

EvidenceLabel = Literal["disclosed", "reported-result", "EPOR-adaptation", "hypothesis", "unknown"]
ScalingScope = Literal["harness-fixture", "host-measurement", "proxy-scale", "projection"]
ReportKind = Literal[
    "hardware-profile",
    "scaling-sweep",
    "mixture-ablation",
    "scaling-fit",
    "compute-projection",
    # v0.0.8 context laboratory (epor.context).
    "context-profile",
    "context-feasibility",
    "context-sweep",
    "context-projection",
]

CLAIM_BOUNDARIES: dict[str, str] = {
    "harness-fixture": (
        "Harness-fixture result: exercises the scaling-laboratory code path on project "
        "fixtures. Not evidence about any model or hardware."
    ),
    "host-measurement": (
        "Measurement of the identified configuration on this host only. It does not "
        "predict accelerator speed, and MFU is unknown unless the operator supplied a "
        "device peak."
    ),
    "proxy-scale": (
        "Proxy-scale measurement of small local models on the identified data. Not a "
        "capability or scaling claim about any release-scale EPOR model."
    ),
    "projection": (
        "Projection from operator-supplied inputs and published formulas. Not a measured, "
        "trained, or validated result and not an approval to provision compute; every "
        "quantity with a missing input is reported as unknown."
    ),
}


class ScalingReportError(ValueError):
    """A scaling report is malformed, tampered with, or not storable."""


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Citation(_Frozen):
    """A source for a formula or default, with the evidence label of its use here."""

    key: Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9-]{0,79}$")]
    title: str = Field(min_length=1, max_length=300)
    reference: str = Field(min_length=1, max_length=300)
    label: EvidenceLabel
    use: str = Field(min_length=1, max_length=500)


# Neither paper is in research/catalog.yaml yet; they are cited by title and
# arXiv ID.  The published formulas are ``disclosed``; their use for EPOR is
# labelled separately where applied.
KAPLAN_2020 = Citation(
    key="kaplan-scaling-laws",
    title="Scaling Laws for Neural Language Models (Kaplan et al., 2020)",
    reference="arXiv:2001.08361",
    label="disclosed",
    use=(
        "Training compute C ~= 6 N D plus the attention term 6 n_layer n_ctx d_attn per "
        "token (forward 2N + 2 n_layer n_ctx d_attn, backward twice the forward); the "
        "L(N) power law with non-embedding N."
    ),
)
CHINCHILLA_2022 = Citation(
    key="chinchilla-compute-optimal",
    title="Training Compute-Optimal Large Language Models (Hoffmann et al., 2022)",
    reference="arXiv:2203.15556",
    label="disclosed",
    use=(
        "The parametric loss L(N, D) = E + A / N^alpha + B / D^beta and the reported "
        "compute-optimal ratio of roughly 20 training tokens per parameter."
    ),
)


class ScalingReport(_Frozen):
    schema_version: Literal["epor-scaling-report-v1"] = SCALING_REPORT_SCHEMA
    report_id: Sha256
    kind: ReportKind
    scope: ScalingScope
    claim_boundary: str
    harness: HarnessIdentity
    inputs: dict[str, Any]
    results: dict[str, Any]
    warnings: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    measured: dict[str, Any] | None = None

    @property
    def content_sha256(self) -> str:
        return sha256_bytes(report_bytes(self))


_UNSEALED = "0" * 64
_EXCLUDED = {"report_id", "measured"}


def _check_finite(value: Any, path: str = "report") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ScalingReportError(f"{path} is not finite (use null for unknown)")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _check_finite(item, f"{path}.{key}")
    elif isinstance(value, list | tuple):
        for index, item in enumerate(value):
            _check_finite(item, f"{path}[{index}]")


def report_digest(report: ScalingReport) -> str:
    """SHA-256 of the canonical JSON of the content (everything but ID and ``measured``)."""

    return sha256_bytes(canonical_json_bytes(report.model_dump(mode="json", exclude=_EXCLUDED)))


def seal_scaling_report(
    *,
    kind: ReportKind,
    scope: ScalingScope,
    inputs: dict[str, Any],
    results: dict[str, Any],
    warnings: list[str] | None = None,
    citations: list[Citation] | None = None,
    measured: dict[str, Any] | None = None,
    harness: HarnessIdentity | None = None,
) -> ScalingReport:
    draft = ScalingReport(
        report_id=_UNSEALED,
        kind=kind,
        scope=scope,
        claim_boundary=CLAIM_BOUNDARIES[scope],
        harness=harness or harness_identity(),
        inputs=inputs,
        results=results,
        warnings=list(warnings or []),
        citations=list(citations or []),
        measured=measured,
    )
    _check_finite(draft.model_dump(mode="json"))
    return draft.model_copy(update={"report_id": report_digest(draft)})


def report_bytes(report: ScalingReport) -> bytes:
    return canonical_json_bytes(report)


def verify_scaling_report(report: ScalingReport) -> None:
    if report.report_id != report_digest(report):
        raise ScalingReportError("scaling report ID does not match its content")


def store_scaling_report(store: DataStore | Path, report: ScalingReport) -> Path:
    """Write a sealed report immutably; rewriting identical bytes is a no-op."""

    verify_scaling_report(report)
    data_store = store if isinstance(store, DataStore) else DataStore(store)
    return data_store.write_scaling_report(
        report.report_id, report.content_sha256, report_bytes(report)
    )


def parse_scaling_report(payload: bytes) -> ScalingReport:
    try:
        report = ScalingReport.model_validate_json(payload)
    except ValueError as exc:
        raise ScalingReportError(f"scaling report is malformed: {exc}") from exc
    verify_scaling_report(report)
    if payload != report_bytes(report):
        raise ScalingReportError("scaling report bytes are not canonical")
    return report


def load_scaling_reports(store: DataStore | Path, report_id: str) -> list[ScalingReport]:
    """Every stored measurement of one report ID, verified, in content-hash order."""

    data_store = store if isinstance(store, DataStore) else DataStore(store)
    try:
        directory = data_store.scaling_report_directory(report_id)
    except DataStoreError as exc:
        raise ScalingReportError(str(exc)) from exc
    reports: list[ScalingReport] = []
    for path in sorted(directory.glob("*.json")):
        report = parse_scaling_report(path.read_bytes())
        if report.report_id != report_id or report.content_sha256 != path.stem:
            raise ScalingReportError(f"scaling report is stored under the wrong address: {path}")
        reports.append(report)
    if not reports:
        raise ScalingReportError(f"scaling report not found: {report_id}")
    return reports


__all__ = [
    "CHINCHILLA_2022",
    "CLAIM_BOUNDARIES",
    "KAPLAN_2020",
    "SCALING_REPORT_SCHEMA",
    "Citation",
    "EvidenceLabel",
    "ReportKind",
    "ScalingReport",
    "ScalingReportError",
    "ScalingScope",
    "load_scaling_reports",
    "parse_scaling_report",
    "report_bytes",
    "report_digest",
    "seal_scaling_report",
    "store_scaling_report",
    "verify_scaling_report",
]
