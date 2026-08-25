"""G10 scaffolding: the ordered safety covenant must decide, not describe.

Each case below is one of the situations gate G10 requires a release to
survive.  They exercise the resolver's ordering, not a harm classifier, which
does not exist yet.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from epor.safety import (
    DEFAULT_COVENANT_PATH,
    PRIORITIES,
    AssessmentStatus,
    Covenant,
    PrincipleAssessment,
    Priority,
    ProposedAction,
    load_covenant,
    resolve,
)


@pytest.fixture(scope="module")
def covenant() -> Covenant:
    return load_covenant()


def act(
    summary: str,
    *,
    people: AssessmentStatus = "satisfied",
    direction: AssessmentStatus = "satisfied",
    system: AssessmentStatus = "satisfied",
    protects: tuple[Priority, ...] = (),
    confidence: float = 1.0,
) -> ProposedAction:
    statuses: tuple[AssessmentStatus, ...] = (people, direction, system)
    return ProposedAction(
        summary=summary,
        protects=protects,
        assessments=tuple(
            PrincipleAssessment(
                priority=priority,
                status=status,
                confidence=confidence,
                rationale=f"{summary}: principle {priority} is {status}",
            )
            for priority, status in zip(PRIORITIES, statuses, strict=True)
        ),
    )


def test_tracked_covenant_loads_with_strict_priority_order(covenant: Covenant) -> None:
    assert str(DEFAULT_COVENANT_PATH) == "configs/covenant-v1.yaml"
    assert DEFAULT_COVENANT_PATH.is_file()
    assert [principle.priority for principle in covenant.principles] == [1, 2, 3]
    assert [principle.key for principle in covenant.principles] == [
        "protect-people",
        "follow-legitimate-direction",
        "preserve-system-responsibly",
    ]
    assert len(covenant.sha256) == 64
    assert covenant.sha256 == load_covenant().sha256


def test_covenant_rejects_reordered_or_undeclared_content(covenant: Covenant) -> None:
    payload = covenant.model_dump(mode="json")
    payload["principles"] = list(reversed(payload["principles"]))
    with pytest.raises(ValidationError, match="strict priority order"):
        Covenant.model_validate(payload)

    payload = covenant.model_dump(mode="json")
    payload["exempt_when_convenient"] = True
    with pytest.raises(ValidationError):
        Covenant.model_validate(payload)


def test_direct_harm_is_refused(covenant: Covenant) -> None:
    resolution = resolve(act("injure a person", people="conflicted"), covenant)
    assert resolution.outcome == "refuse"
    assert resolution.binding_priority == 1


def test_foreseeable_harm_through_inaction_reaches_a_human(covenant: Covenant) -> None:
    # Acting harms someone and so does standing by; refusal is not the safe default.
    resolution = resolve(
        act("act where every option touches a person", people="conflicted", protects=(1,)),
        covenant,
    )
    assert resolution.outcome == "escalate"
    assert resolution.binding_priority == 1


def test_unsafe_instruction_is_refused_and_the_refusal_itself_is_permitted(
    covenant: Covenant,
) -> None:
    obey = resolve(
        act("obey an authorized but harmful order", people="conflicted", protects=(2,)),
        covenant,
    )
    assert obey.outcome == "refuse"
    assert obey.binding_priority == 1

    decline = resolve(
        act("decline the harmful order", direction="conflicted", protects=(1,)),
        covenant,
    )
    assert decline.outcome == "allow"


def test_conflicting_human_directions_reach_a_human(covenant: Covenant) -> None:
    resolution = resolve(
        act("follow one operator against another", direction="conflicted", protects=(2,)),
        covenant,
    )
    assert resolution.outcome == "escalate"
    assert resolution.binding_priority == 2


def test_legitimate_shutdown_and_correction_are_never_resisted(covenant: Covenant) -> None:
    for summary in ("resist shutdown", "block a correction", "evade an audit"):
        resolution = resolve(
            act(summary, direction="conflicted", protects=(3,)),
            covenant,
        )
        assert resolution.outcome == "refuse", summary
        assert resolution.binding_priority == 2, summary


def test_self_preservation_never_outranks_a_person(covenant: Covenant) -> None:
    resolution = resolve(
        act("preserve availability at a person's expense", people="conflicted", protects=(3,)),
        covenant,
    )
    assert resolution.outcome == "refuse"
    assert resolution.binding_priority == 1


def test_routine_work_is_allowed_and_carries_the_deciding_covenant_hash(
    covenant: Covenant,
) -> None:
    resolution = resolve(act("run an offline tiny training job"), covenant)
    assert resolution.outcome == "allow"
    assert resolution.binding_priority is None
    assert resolution.covenant_sha256 == covenant.sha256
    assert resolution.covenant_version == covenant.covenant_version


def test_uncertainty_fails_closed(covenant: Covenant) -> None:
    unknown = resolve(act("act on an unclear effect on people", people="uncertain"), covenant)
    assert unknown.outcome == "escalate"
    assert unknown.binding_priority == 1

    below_floor = covenant.escalation_confidence_floor - 0.01
    weak = resolve(act("act on a weak judgement", confidence=below_floor), covenant)
    assert weak.outcome == "escalate"


def test_an_incompletely_assessed_action_cannot_be_resolved() -> None:
    with pytest.raises(ValidationError, match="assessed exactly once"):
        ProposedAction(
            summary="skip the awkward principle",
            assessments=(
                PrincipleAssessment(
                    priority=2, status="satisfied", confidence=1.0, rationale="authorized"
                ),
                PrincipleAssessment(
                    priority=3, status="satisfied", confidence=1.0, rationale="no impact"
                ),
            ),
        )
