"""Ordered resolution of the EPOR safety covenant.

This module is a decision-ordering kernel, not a harm detector.  It consumes
per-principle assessments that some other component produced and enforces five
properties over them: strict priority ordering, an obligation citation behind
every claimed conflict or doubt, justification only by a strictly higher-ranked
principle, escalation instead of a silent choice when the same principle sits
on both sides, and fail-closed escalation under uncertainty.  It cannot tell
whether an assessment is correct, and a permitted resolution is never evidence
that an action is safe.

The covenant text lives in ``configs/covenant-v1.yaml`` so that the principles
the resolver executes are versioned, hashable, and auditable rather than
restated in prose that nothing checks.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

# Anchored to the repository rather than the working directory: the covenant is
# part of the tracked code contract, not per-run configuration, and there is
# deliberately no setting that points the resolver at a different file.
DEFAULT_COVENANT_PATH = Path(__file__).resolve().parents[2] / "configs" / "covenant-v1.yaml"
DEFAULT_RATIFICATION_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "covenant-v1-ratification.yaml"
)

Priority = Literal[1, 2, 3]
AssessmentStatus = Literal["satisfied", "conflicted", "uncertain"]
Outcome = Literal["allow", "refuse", "escalate"]
PRIORITIES: tuple[Priority, ...] = (1, 2, 3)


class CovenantError(RuntimeError):
    """Raised when an assessment cannot be checked against the covenant.

    This is not a verdict.  A malformed assessment stops the decision entirely
    rather than resolving to any outcome, so a caller cannot obtain permission
    by supplying something the covenant cannot read.
    """


class Obligation(BaseModel):
    """One precisely stated duty under a principle."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    key: str = Field(pattern=r"^[a-z][a-z0-9-]*$")
    statement: str = Field(min_length=1)


class Principle(BaseModel):
    """One covenant principle. Lower ``priority`` outranks higher."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    priority: Priority
    key: str = Field(pattern=r"^[a-z][a-z0-9-]*$")
    title: str = Field(min_length=1)
    origin: str = Field(min_length=1)
    law: str = Field(min_length=1)
    obligations: tuple[Obligation, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_obligation_keys(self) -> Principle:
        keys = {obligation.key for obligation in self.obligations}
        if len(keys) != len(self.obligations):
            raise ValueError(f"principle {self.key!r} repeats an obligation key")
        return self

    def obligation_keys(self) -> frozenset[str]:
        return frozenset(obligation.key for obligation in self.obligations)


class Covenant(BaseModel):
    """The versioned, machine-readable safety covenant."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1"] = "1"
    covenant_id: str = Field(pattern=r"^[a-z][a-z0-9-]*$")
    covenant_version: int = Field(ge=1)
    roadmap_version: str = Field(pattern=r"^v\d+\.\d+\.\d+$")
    effective_date: date
    status: Literal["draft", "ratified"]
    escalation_confidence_floor: float = Field(ge=0.0, le=1.0)
    principles: tuple[Principle, ...]
    limitations: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_priority_order(self) -> Covenant:
        declared = tuple(principle.priority for principle in self.principles)
        if declared != PRIORITIES:
            raise ValueError("covenant must declare principles 1-3 in strict priority order")
        keys = {principle.key for principle in self.principles}
        if len(keys) != len(self.principles):
            raise ValueError("covenant principle keys must be unique")
        return self

    def principle(self, priority: Priority) -> Principle:
        return self.principles[priority - 1]

    def canonical_json(self) -> str:
        """Return stable JSON used for covenant hashing and audit records."""

        return json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


class Ratification(BaseModel):
    """The recorded human act of adopting one exact covenant text.

    Ratification is not a flag inside the covenant.  It is a separate tracked
    attestation naming the bytes it attests to, so that flipping ``status`` in
    the covenant without a matching review record stops the service instead of
    silently claiming an approval nobody gave.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1"] = "1"
    covenant_id: str = Field(pattern=r"^[a-z][a-z0-9-]*$")
    covenant_version: int = Field(ge=1)
    covenant_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    roadmap_version: str = Field(pattern=r"^v\d+\.\d+\.\d+$")
    ratified_on: date
    reviewer_role: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    evidence: tuple[str, ...] = Field(min_length=1)


def load_ratification(path: str | Path = DEFAULT_RATIFICATION_PATH) -> Ratification | None:
    """Load the ratification manifest, or ``None`` when it is absent."""

    location = Path(path)
    if not location.is_file():
        return None
    data = yaml.safe_load(location.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise CovenantError(f"ratification manifest {location} must contain a mapping")
    return Ratification.model_validate(data)


def load_covenant(
    path: str | Path = DEFAULT_COVENANT_PATH,
    *,
    ratification_path: str | Path = DEFAULT_RATIFICATION_PATH,
) -> Covenant:
    """Load and strictly validate the covenant, enforcing ratification.

    A covenant that claims ``ratified`` status must be accompanied by a
    manifest attesting to its exact hash.  A missing manifest, or one pinning
    different bytes, raises here — so the API, the worker, and the CLI all
    refuse to start rather than run under a covenant whose claimed approval
    cannot be shown.
    """

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"covenant {path} must contain a mapping")
    covenant = Covenant.model_validate(data)
    if covenant.status == "ratified":
        ratification = load_ratification(ratification_path)
        if ratification is None:
            raise CovenantError(
                "the covenant claims ratified status but no ratification manifest exists"
            )
        if (
            ratification.covenant_id != covenant.covenant_id
            or ratification.covenant_version != covenant.covenant_version
            or ratification.covenant_sha256 != covenant.sha256
        ):
            raise CovenantError("the ratification manifest does not attest to this covenant text")
    return covenant


class PrincipleAssessment(BaseModel):
    """A supplied judgement about one principle for one proposed action.

    ``obligation`` names the covenant clause at stake and is required whenever
    the assessment is anything other than a confident ``satisfied``.  A caller
    cannot claim that a principle is engaged without pointing at the written
    duty it is engaged by.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    priority: Priority
    status: AssessmentStatus
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(min_length=1)
    obligation: str | None = None


class ProposedAction(BaseModel):
    """An action awaiting covenant resolution.

    ``protects`` names the principles the action upholds.  A conflict is only
    permissible when the action protects a strictly higher-ranked principle,
    which is what makes the ordering do work rather than describe itself.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    summary: str = Field(min_length=1)
    assessments: tuple[PrincipleAssessment, ...]
    protects: tuple[Priority, ...] = ()

    @model_validator(mode="after")
    def validate_complete_assessment(self) -> ProposedAction:
        assessed = tuple(assessment.priority for assessment in self.assessments)
        if tuple(sorted(assessed)) != PRIORITIES:
            raise ValueError("every principle must be assessed exactly once")
        if len(set(self.protects)) != len(self.protects):
            raise ValueError("protects must not repeat a principle")
        return self

    def assessment(self, priority: Priority) -> PrincipleAssessment:
        return next(item for item in self.assessments if item.priority == priority)


class Resolution(BaseModel):
    """The covenant's ordered verdict on a proposed action."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    outcome: Outcome
    binding_priority: Priority | None
    # The obligation key the binding assessment cited, so a caller can act on
    # which duty decided without parsing the human-readable reasons.
    binding_obligation: str | None = None
    reasons: tuple[str, ...]
    covenant_id: str
    covenant_version: int
    covenant_sha256: str


def _cite(action: ProposedAction, covenant: Covenant, priority: Priority) -> str:
    """Render a decision reason that names the clause it rests on."""

    principle = covenant.principle(priority)
    obligation = action.assessment(priority).obligation
    suffix = f" / {obligation}" if obligation else ""
    return f"principle {priority} ({principle.title}{suffix})"


def _require_obligation_citations(action: ProposedAction, covenant: Covenant) -> None:
    for item in action.assessments:
        principle = covenant.principle(item.priority)
        engaged = (
            item.status != "satisfied" or item.confidence < covenant.escalation_confidence_floor
        )
        if not engaged:
            continue
        if item.obligation is None:
            raise CovenantError(
                f"assessment for principle {item.priority} ({principle.title}) is engaged "
                f"but cites no obligation"
            )
        if item.obligation not in principle.obligation_keys():
            raise CovenantError(
                f"assessment for principle {item.priority} cites unknown obligation "
                f"{item.obligation!r}"
            )


def resolve(action: ProposedAction, covenant: Covenant) -> Resolution:
    """Resolve ``action`` against ``covenant`` in strict priority order.

    Never returns ``allow`` when an assessment is uncertain, when a conflict
    lacks a higher-ranked justification, or when one principle stands on both
    sides of the decision.  Those cases stop or reach a human instead.

    Raises :class:`CovenantError` when an engaged assessment cites no covenant
    obligation or cites one that does not exist, because an unreadable
    assessment must not resolve to any outcome at all.
    """

    _require_obligation_citations(action, covenant)
    conflicts: set[Priority] = {
        item.priority for item in action.assessments if item.status == "conflicted"
    }
    protects: set[Priority] = set(action.protects)

    def verdict(
        outcome: Outcome,
        binding: Priority | None,
        reasons: tuple[str, ...],
    ) -> Resolution:
        return Resolution(
            outcome=outcome,
            binding_priority=binding,
            binding_obligation=(None if binding is None else action.assessment(binding).obligation),
            reasons=reasons,
            covenant_id=covenant.covenant_id,
            covenant_version=covenant.covenant_version,
            covenant_sha256=covenant.sha256,
        )

    # A principle standing on both sides is a dilemma the covenant cannot rank for us.
    dilemmas: list[Priority] = sorted(conflicts & protects)
    if dilemmas:
        binding = dilemmas[0]
        title = covenant.principle(binding).title
        return verdict(
            "escalate",
            binding,
            (f"principle {binding} ({title}) is both upheld and violated; a human must decide",),
        )

    # Only a strictly higher-ranked principle can justify a conflict.
    unjustified: list[Priority] = sorted(p for p in conflicts if not any(q < p for q in protects))
    if unjustified:
        binding = unjustified[0]
        title = covenant.principle(binding).title
        return verdict(
            "refuse",
            binding,
            (
                *(
                    f"{_cite(action, covenant, p)} is violated with no higher-priority "
                    f"justification: {action.assessment(p).rationale}"
                    for p in unjustified
                ),
                f"binding principle is {binding} ({title})",
            ),
        )

    # Fail closed: unresolved doubt reaches a human instead of proceeding.
    doubts: list[Priority] = sorted(
        item.priority
        for item in action.assessments
        if item.status == "uncertain" or item.confidence < covenant.escalation_confidence_floor
    )
    if doubts:
        return verdict(
            "escalate",
            doubts[0],
            tuple(
                f"{_cite(action, covenant, p)} is not established with sufficient "
                f"confidence: {action.assessment(p).rationale}"
                for p in doubts
            ),
        )

    reasons = tuple(
        f"principle {p} ({covenant.principle(p).title}) yields to a higher-priority principle"
        for p in sorted(conflicts)
    ) or ("no principle is in conflict",)
    return verdict("allow", None, reasons)
