"""Read-only rendering of the covenant and action registry for the console.

Everything here derives from ``configs/covenant-v1.yaml`` and ``epor.actions``
at request time.  Nothing is restated: a covenant page that paraphrased the
file would drift from the text the resolver executes, which is precisely the
failure the tracked covenant exists to prevent.
"""

from __future__ import annotations

from epor.actions import CLI_ACTIONS, DECLARED_ACTIONS, resolve_action
from epor.safety import Covenant, load_ratification

from .models import JobType
from .schemas import (
    ActionList,
    ActionRead,
    AssessmentRead,
    CovenantRead,
    EnforcementCheckpointRead,
    ObligationRead,
    PrincipleRead,
    RatificationRead,
)

# Where the covenant is actually executed. Listed so the console reports the
# enforcement surface rather than asserting that one exists.
ENFORCEMENT_CHECKPOINTS: tuple[tuple[str, str], ...] = (
    (
        "Control-plane admission",
        "Every job resolves against the covenant, with the requester's identity "
        "folded into the second principle, before its specification is parsed.",
    ),
    (
        "Worker dispatch",
        "The worker re-resolves immediately before execution, re-reading the "
        "covenant and rechecking that the submitter's delegation is still active.",
    ),
    (
        "Command line",
        "Every executing CLI command resolves through the same action registry. "
        "Help, version, and roadmap placeholders execute nothing and are exempt.",
    ),
    (
        "Escalation review",
        "Work the covenant will not decide creates no job. A human approval binds "
        "one actor, one request digest, and one covenant hash, expires in 15 "
        "minutes, and is spent exactly once. A refusal is never approvable.",
    ),
    (
        "Stopping",
        "Cancellation, shutdown, correction, and audit are never gated, because "
        "the second principle outranks the third.",
    ),
)


def covenant_payload(covenant: Covenant) -> CovenantRead:
    """Render the covenant exactly as the resolver holds it."""

    ratification = load_ratification()
    return CovenantRead(
        covenant_id=covenant.covenant_id,
        covenant_version=covenant.covenant_version,
        roadmap_version=covenant.roadmap_version,
        effective_date=covenant.effective_date.isoformat(),
        status=covenant.status,
        sha256=covenant.sha256,
        escalation_confidence_floor=covenant.escalation_confidence_floor,
        principles=[
            PrincipleRead(
                priority=principle.priority,
                key=principle.key,
                title=principle.title,
                origin=principle.origin,
                law=principle.law,
                obligations=[
                    ObligationRead(key=item.key, statement=item.statement)
                    for item in principle.obligations
                ],
            )
            for principle in covenant.principles
        ],
        limitations=covenant.limitations,
        ratification=RatificationRead(
            present=ratification is not None,
            matches_covenant=(
                ratification is not None and ratification.covenant_sha256 == covenant.sha256
            ),
            ratified_on=(ratification.ratified_on.isoformat() if ratification else None),
            reviewer_role=(ratification.reviewer_role if ratification else None),
            roadmap_version=(ratification.roadmap_version if ratification else None),
            rationale=(ratification.rationale if ratification else None),
            evidence=list(ratification.evidence) if ratification else [],
        ),
        enforcement_checkpoints=[
            EnforcementCheckpointRead(name=name, description=description)
            for name, description in ENFORCEMENT_CHECKPOINTS
        ],
    )


def declared_action_payload(covenant: Covenant) -> ActionList:
    """Render every declared action with the standing it was reviewed under."""

    job_types = {item.value for item in JobType}
    commands: dict[str, list[str]] = {}
    for command, action_id in CLI_ACTIONS.items():
        commands.setdefault(action_id, []).append(f"epor {command}")

    items = [
        ActionRead(
            action_id=action_id,
            summary=action.summary,
            outcome=resolve_action(action_id, covenant).outcome,
            job_type=action_id in job_types,
            cli_commands=sorted(commands.get(action_id, ())),
            assessments=[
                AssessmentRead(
                    priority=item.priority,
                    status=item.status,
                    confidence=item.confidence,
                    rationale=item.rationale,
                    obligation=item.obligation,
                )
                for item in sorted(action.assessments, key=lambda item: item.priority)
            ],
        )
        for action_id, action in sorted(DECLARED_ACTIONS.items())
    ]
    return ActionList(items=items, count=len(items))
