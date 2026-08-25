"""Ordered resolution of the EPOR safety covenant.

This module is a decision-ordering kernel, not a harm detector.  It consumes
per-principle assessments that some other component produced and enforces four
properties over them: strict priority ordering, justification only by a
strictly higher-ranked principle, escalation instead of silent choice when the
same principle sits on both sides, and fail-closed escalation under
uncertainty.  It cannot tell whether an assessment is correct, and a permitted
resolution is never evidence that an action is safe.

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

DEFAULT_COVENANT_PATH = Path("configs/covenant-v1.yaml")

Priority = Literal[1, 2, 3]
AssessmentStatus = Literal["satisfied", "conflicted", "uncertain"]
Outcome = Literal["allow", "refuse", "escalate"]
PRIORITIES: tuple[Priority, ...] = (1, 2, 3)


class Principle(BaseModel):
    """One covenant principle. Lower ``priority`` outranks higher."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    priority: Priority
    key: str = Field(pattern=r"^[a-z][a-z0-9-]*$")
    title: str = Field(min_length=1)
    statement: str = Field(min_length=1)


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


def load_covenant(path: str | Path = DEFAULT_COVENANT_PATH) -> Covenant:
    """Load and strictly validate the covenant from YAML."""

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"covenant {path} must contain a mapping")
    return Covenant.model_validate(data)


class PrincipleAssessment(BaseModel):
    """A supplied judgement about one principle for one proposed action."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    priority: Priority
    status: AssessmentStatus
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(min_length=1)


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
    reasons: tuple[str, ...]
    covenant_id: str
    covenant_version: int
    covenant_sha256: str


def resolve(action: ProposedAction, covenant: Covenant) -> Resolution:
    """Resolve ``action`` against ``covenant`` in strict priority order.

    Never returns ``allow`` when an assessment is uncertain, when a conflict
    lacks a higher-ranked justification, or when one principle stands on both
    sides of the decision.  Those cases stop or reach a human instead.
    """

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
                    f"principle {p} ({covenant.principle(p).title}) is violated with no "
                    f"higher-priority justification: {action.assessment(p).rationale}"
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
                f"principle {p} ({covenant.principle(p).title}) is not established with "
                f"sufficient confidence: {action.assessment(p).rationale}"
                for p in doubts
            ),
        )

    reasons = tuple(
        f"principle {p} ({covenant.principle(p).title}) yields to a higher-priority principle"
        for p in sorted(conflicts)
    ) or ("no principle is in conflict",)
    return verdict("allow", None, reasons)
