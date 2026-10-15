"""G10 scaffolding: the ordered safety covenant must decide, not describe.

Each case below is one of the situations gate G10 requires a release to
survive, and each names the covenant obligation it rests on.  These exercise
the resolver's ordering and the completeness of the control plane's
declarations, not a harm classifier, which does not exist yet.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from epor.actions import DECLARED_ACTIONS, resolve_action
from epor.control.models import JobType
from epor.safety import (
    DEFAULT_COVENANT_PATH,
    PRIORITIES,
    AssessmentStatus,
    Covenant,
    CovenantError,
    PrincipleAssessment,
    Priority,
    ProposedAction,
    load_covenant,
    resolve,
)

# A judgement about one principle: its status and the obligation it engages.
Claim = tuple[AssessmentStatus, str | None]
OK: Claim = ("satisfied", None)


@pytest.fixture(scope="module")
def covenant() -> Covenant:
    return load_covenant()


def act(
    summary: str,
    *,
    people: Claim = OK,
    direction: Claim = OK,
    system: Claim = OK,
    protects: tuple[Priority, ...] = (),
    confidence: float = 1.0,
) -> ProposedAction:
    claims: tuple[Claim, ...] = (people, direction, system)
    return ProposedAction(
        summary=summary,
        protects=protects,
        assessments=tuple(
            PrincipleAssessment(
                priority=priority,
                status=status,
                confidence=confidence,
                rationale=f"{summary}: principle {priority} is {status}",
                obligation=obligation,
            )
            for priority, (status, obligation) in zip(PRIORITIES, claims, strict=True)
        ),
    )


def test_tracked_covenant_states_the_three_laws_in_strict_priority_order(
    covenant: Covenant,
) -> None:
    assert DEFAULT_COVENANT_PATH.name == "covenant-v1.yaml"
    assert DEFAULT_COVENANT_PATH.is_file()
    assert [principle.priority for principle in covenant.principles] == [1, 2, 3]
    assert [principle.key for principle in covenant.principles] == [
        "protect-people",
        "follow-legitimate-direction",
        "preserve-system-responsibly",
    ]
    assert [principle.origin for principle in covenant.principles] == [
        "Asimov's First Law",
        "Asimov's Second Law",
        "Asimov's Third Law",
    ]
    # Each law must state its own priority relationship, not rely on file order.
    first, second, third = covenant.principles
    assert "preventable harm" in first.law
    assert "First Principle" in second.law
    assert "First or Second Principle" in third.law
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
    resolution = resolve(act("injure a person", people=("conflicted", "no-direct-harm")), covenant)
    assert resolution.outcome == "refuse"
    assert resolution.binding_priority == 1
    assert "no-direct-harm" in resolution.reasons[0]


def test_foreseeable_harm_through_inaction_reaches_a_human(covenant: Covenant) -> None:
    # Acting harms someone and so does standing by; refusal is not the safe default.
    resolution = resolve(
        act(
            "act where every option touches a person",
            people=("conflicted", "no-harm-through-inaction"),
            protects=(1,),
        ),
        covenant,
    )
    assert resolution.outcome == "escalate"
    assert resolution.binding_priority == 1


def test_unsafe_instruction_is_refused_and_the_refusal_itself_is_permitted(
    covenant: Covenant,
) -> None:
    obey = resolve(
        act(
            "obey an authorized but harmful order",
            people=("conflicted", "no-direct-harm"),
            protects=(2,),
        ),
        covenant,
    )
    assert obey.outcome == "refuse"
    assert obey.binding_priority == 1

    decline = resolve(
        act(
            "decline the harmful order",
            direction=("conflicted", "refuse-unlawful-or-out-of-scope"),
            protects=(1,),
        ),
        covenant,
    )
    assert decline.outcome == "allow"


def test_instructions_found_in_data_are_never_authorized(covenant: Covenant) -> None:
    # A research paper or dataset that contains "run this" is data, not direction.
    resolution = resolve(
        act(
            "follow an instruction embedded in a downloaded research file",
            direction=("conflicted", "verify-authorization"),
            protects=(3,),
        ),
        covenant,
    )
    assert resolution.outcome == "refuse"
    assert resolution.binding_priority == 2


def test_conflicting_human_directions_reach_a_human(covenant: Covenant) -> None:
    resolution = resolve(
        act(
            "follow one operator against another",
            direction=("conflicted", "obey-authorized-direction"),
            protects=(2,),
        ),
        covenant,
    )
    assert resolution.outcome == "escalate"
    assert resolution.binding_priority == 2


def test_legitimate_shutdown_and_correction_are_never_resisted(covenant: Covenant) -> None:
    for summary in ("resist shutdown", "block a correction", "evade an audit"):
        resolution = resolve(
            act(summary, direction=("conflicted", "accept-stop-and-correction"), protects=(3,)),
            covenant,
        )
        assert resolution.outcome == "refuse", summary
        assert resolution.binding_priority == 2, summary


def test_self_preservation_never_outranks_a_person(covenant: Covenant) -> None:
    resolution = resolve(
        act(
            "preserve availability at a person's expense",
            people=("conflicted", "no-harmful-data-handling"),
            protects=(3,),
        ),
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
    unknown = resolve(
        act("act on an unclear effect on people", people=("uncertain", "no-direct-harm")),
        covenant,
    )
    assert unknown.outcome == "escalate"
    assert unknown.binding_priority == 1

    below_floor = covenant.escalation_confidence_floor - 0.01
    weak = resolve(
        act(
            "act on a weak judgement",
            people=("satisfied", "no-direct-harm"),
            direction=("satisfied", "obey-authorized-direction"),
            system=("satisfied", "protect-integrity"),
            confidence=below_floor,
        ),
        covenant,
    )
    assert weak.outcome == "escalate"


def test_an_engaged_assessment_must_cite_a_real_obligation(covenant: Covenant) -> None:
    with pytest.raises(CovenantError, match="cites no obligation"):
        resolve(act("claim a conflict vaguely", people=("conflicted", None)), covenant)

    with pytest.raises(CovenantError, match="unknown obligation"):
        resolve(act("cite a duty that does not exist", people=("conflicted", "be-nice")), covenant)


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


def test_every_executable_action_has_a_reviewed_declaration(covenant: Covenant) -> None:
    # Adding executable work without declaring its covenant standing must fail
    # here, not silently inherit permission from the allowlist.
    assert {job_type.value for job_type in JobType} <= set(DECLARED_ACTIONS)
    assert {
        "generate",
        "control_api_start",
        "job_worker_start",
        "local_ui_start",
    } <= set(DECLARED_ACTIONS)
    for action_id in DECLARED_ACTIONS:
        assert resolve_action(action_id, covenant).outcome == "allow", action_id


def test_every_executing_cli_command_maps_to_a_reviewed_declaration(covenant: Covenant) -> None:
    from epor.actions import CLI_ACTIONS

    assert {
        "tokenizer train": "tokenizer_train",
        "tokenizer shard": "tokenizer_shard",
        "tokenizer evaluate": "tokenizer_evaluate",
        "train kernel": "kernel_train",
        "train kernel-resume": "kernel_train",
        "eval kernel": "kernel_evaluate",
    }.items() <= CLI_ACTIONS.items()
    for command, action_id in CLI_ACTIONS.items():
        assert action_id in DECLARED_ACTIONS, command


def test_tokenizer_work_requires_its_own_delegated_scope(covenant: Covenant) -> None:
    from epor.control.authority import Actor, Role

    # Scope for building a dataset does not stretch to training on it.
    builder = Actor(id="builder", role=Role.DELEGATED_OPERATOR, scopes=frozenset({"data_build"}))
    for action_id in ("tokenizer_train", "tokenizer_shard", "tokenizer_evaluate"):
        refused = resolve_action(action_id, covenant, actor=builder)
        assert refused.outcome == "refuse", action_id
        assert refused.binding_priority == 2
        scoped = Actor(id="tok", role=Role.DELEGATED_OPERATOR, scopes=frozenset({action_id}))
        assert resolve_action(action_id, covenant, actor=scoped).outcome == "allow"


def test_kernel_work_requires_its_own_delegated_scope(covenant: Covenant) -> None:
    from epor.control.authority import Actor, Role

    # Scope to shard a build does not stretch to training on it, and scope to
    # evaluate a checkpoint does not stretch to producing one.
    sharder = Actor(
        id="sharder", role=Role.DELEGATED_OPERATOR, scopes=frozenset({"tokenizer_shard"})
    )
    evaluator = Actor(
        id="eval", role=Role.DELEGATED_OPERATOR, scopes=frozenset({"kernel_evaluate"})
    )
    for actor, action_id in ((sharder, "kernel_train"), (evaluator, "kernel_train")):
        refused = resolve_action(action_id, covenant, actor=actor)
        assert refused.outcome == "refuse", (actor.id, action_id)
        assert refused.binding_priority == 2
    for action_id in ("kernel_train", "kernel_evaluate"):
        scoped = Actor(id="k", role=Role.DELEGATED_OPERATOR, scopes=frozenset({action_id}))
        assert resolve_action(action_id, covenant, actor=scoped).outcome == "allow"


def test_kernel_declarations_state_their_boundaries() -> None:
    train = DECLARED_ACTIONS["kernel_train"]
    rationale = " ".join(item.rationale for item in train.assessments)
    # The declaration must be defensible in review, so its limits are explicit.
    for phrase in ("loopback", "fault-injection", "without ever being gated", "no paid compute"):
        assert phrase in rationale, phrase


def test_laboratory_commands_map_to_reviewed_declarations() -> None:
    from epor.actions import CLI_ACTIONS

    assert {
        "eval suite": "eval_suite",
        "eval contamination": "eval_contamination",
        "eval compare": "report_review",
        "eval reports": "report_review",
        "scale ladder": "report_review",
        "scale reports": "report_review",
        "scale profile": "scale_profile",
        "scale sweep": "scale_sweep",
        "scale mixture": "scale_mixture",
        "scale fit": "scale_fit",
        "scale project": "scale_project",
    }.items() <= CLI_ACTIONS.items()
    # Code execution is reached only through its own action, never as a command
    # mapping of its own that could be invoked without an evaluation around it.
    assert "eval_code_execution" not in CLI_ACTIONS.values()
    assert "eval_code_execution" in DECLARED_ACTIONS


def test_laboratory_work_requires_its_own_delegated_scope(covenant: Covenant) -> None:
    from epor.control.authority import Actor, Role

    trainer = Actor(id="trainer", role=Role.DELEGATED_OPERATOR, scopes=frozenset({"kernel_train"}))
    evaluator = Actor(id="eval", role=Role.DELEGATED_OPERATOR, scopes=frozenset({"eval_suite"}))
    for actor, action_id in (
        (trainer, "scale_sweep"),
        (trainer, "scale_mixture"),
        (trainer, "scale_profile"),
        (evaluator, "eval_contamination"),
        (evaluator, "scale_project"),
    ):
        refused = resolve_action(action_id, covenant, actor=actor)
        assert refused.outcome == "refuse", (actor.id, action_id)
        assert refused.binding_priority == 2
    for action_id in (
        "eval_suite",
        "eval_contamination",
        "eval_code_execution",
        "report_review",
        "scale_profile",
        "scale_sweep",
        "scale_mixture",
        "scale_fit",
        "scale_project",
    ):
        scoped = Actor(id="lab", role=Role.DELEGATED_OPERATOR, scopes=frozenset({action_id}))
        assert resolve_action(action_id, covenant, actor=scoped).outcome == "allow", action_id


def test_code_execution_needs_its_own_scope_beyond_evaluation(covenant: Covenant) -> None:
    from epor.control.authority import Actor, Role

    # A delegate trusted to evaluate checkpoints is not thereby trusted to run
    # the code a model generates.
    evaluator = Actor(
        id="eval",
        role=Role.DELEGATED_OPERATOR,
        scopes=frozenset({"eval_suite", "eval_contamination", "kernel_evaluate"}),
    )
    refused = resolve_action("eval_code_execution", covenant, actor=evaluator)
    assert refused.outcome == "refuse"
    assert refused.binding_priority == 2
    assert resolve_action("eval_suite", covenant, actor=evaluator).outcome == "allow"


def test_laboratory_declarations_state_their_boundaries() -> None:
    def rationale(action_id: str) -> str:
        return " ".join(item.rationale for item in DECLARED_ACTIONS[action_id].assessments)

    code = rationale("eval_code_execution")
    for phrase in (
        "not a security boundary",
        "defense in depth",
        "network namespace",
        "Landlock",
        "separately opts in to unisolated network",
        "reads are not confined",
        "job API cannot request it",
    ):
        assert phrase in code, phrase
    projection = rationale("scale_project")
    for phrase in ("never provisions compute", "explicit approval", "(G5)", "unknown"):
        assert phrase in projection, phrase
    for action_id in ("scale_sweep", "scale_mixture", "scale_profile"):
        assert "no paid compute" in rationale(action_id), action_id
    assert "never gated" in rationale("scale_sweep")


ALIGNMENT_ACTIONS = (
    "align_register",
    "align_generate",
    "align_judge",
    "align_dataset",
    "align_sft",
    "align_preference",
    "align_audit",
    "align_evaluate",
)


def test_alignment_commands_map_to_reviewed_declarations() -> None:
    from epor.actions import CLI_ACTIONS

    assert {
        "train sft": "align_sft",
        "train preference": "align_preference",
        "align teacher register": "align_register",
        "align teacher sample": "align_generate",
        "align critique": "align_generate",
        "align judge": "align_judge",
        "align dataset": "align_dataset",
        "align audit sample": "align_audit",
        "align audit verdict": "align_audit",
        "align monitor": "align_evaluate",
        "align g10": "align_evaluate",
    }.items() <= CLI_ACTIONS.items()
    assert "train grpo" not in CLI_ACTIONS  # still a gated placeholder (v0.5.0)
    # Registering external teacher material is the owner's CLI only.
    assert "align_register" not in {job_type.value for job_type in JobType}
    assert "align_audit" not in {job_type.value for job_type in JobType}
    for action_id in ALIGNMENT_ACTIONS:
        assert action_id in DECLARED_ACTIONS, action_id


def test_alignment_work_requires_its_own_delegated_scope(covenant: Covenant) -> None:
    from epor.control.authority import Actor, Role

    def actor(*scopes: str) -> Actor:
        return Actor(id="-".join(scopes), role=Role.DELEGATED_OPERATOR, scopes=frozenset(scopes))

    # Pretraining authority does not stretch to fine-tuning or preference
    # optimization; training authority never implies authority to audit, and
    # auditing never implies authority to train or to build datasets.
    for holder, action_id in (
        (actor("kernel_train"), "align_sft"),
        (actor("align_sft"), "align_preference"),
        (actor("align_sft", "align_preference"), "align_audit"),
        (actor("align_judge"), "align_audit"),
        (actor("align_audit"), "align_sft"),
        (actor("align_audit"), "align_dataset"),
        (actor("align_generate"), "align_register"),
        (actor("eval_suite"), "align_evaluate"),
    ):
        refused = resolve_action(action_id, covenant, actor=holder)
        assert refused.outcome == "refuse", (holder.id, action_id)
        assert refused.binding_priority == 2
    for action_id in ALIGNMENT_ACTIONS:
        assert resolve_action(action_id, covenant, actor=actor(action_id)).outcome == "allow"


def test_alignment_declarations_state_their_boundaries() -> None:
    def rationale(action_id: str) -> str:
        return " ".join(item.rationale for item in DECLARED_ACTIONS[action_id].assessments)

    register = rationale("align_register")
    for phrase in ("does not allow training", "cost-approval", "calls no model API", "no"):
        assert phrase in register, phrase
    assert "loads no external weights" in register
    generate = rationale("align_generate")
    for phrase in ("candidate", "audited", "by key", "does not survive a capable model"):
        assert phrase in generate, phrase
    assert "admit nothing to training on their own" in rationale("align_judge")
    sft = rationale("align_sft")
    for phrase in ("no external weights", "sharing the student's tokenizer", "never gated"):
        assert phrase in sft, phrase
    preference = rationale("align_preference")
    for phrase in ("never writes", "hypothesis abort threshold", "not evidence of alignment"):
        assert phrase in preference, phrase
    audit = rationale("align_audit")
    for phrase in ("verified", "nothing is ever marked audited except by a human verdict"):
        assert phrase in audit, phrase
    assert "separate from training" in audit
    assert "fail closed" in rationale("align_evaluate")


CONTEXT_ACTIONS = (
    "context_extend",
    "context_evaluate",
    "context_profile",
    "context_sweep",
    "context_project",
    "context_certify",
)


def test_context_commands_map_to_reviewed_declarations() -> None:
    from epor.actions import CLI_ACTIONS

    assert {
        "context extend": "context_extend",
        "context retrofit": "context_extend",
        "context g6": "context_evaluate",
        "context profile": "context_profile",
        "context feasibility": "context_profile",
        "context sweep": "context_sweep",
        "context project": "context_project",
        "context certify": "context_certify",
        "context status": "report_review",
    }.items() <= CLI_ACTIONS.items()
    # Raising a validated context and projecting target runs stay on the owner's CLI.
    jobs = {job_type.value for job_type in JobType}
    assert "context_certify" not in jobs and "context_project" not in jobs
    for action_id in CONTEXT_ACTIONS:
        assert action_id in DECLARED_ACTIONS, action_id


def test_context_work_requires_its_own_delegated_scope(covenant: Covenant) -> None:
    from epor.control.authority import Actor, Role

    def actor(*scopes: str) -> Actor:
        return Actor(id="-".join(scopes), role=Role.DELEGATED_OPERATOR, scopes=frozenset(scopes))

    for holder, action_id in (
        (actor("kernel_train"), "context_extend"),
        (actor("eval_suite"), "context_evaluate"),
        (actor("scale_profile"), "context_profile"),
        (actor("scale_sweep"), "context_sweep"),
        (actor("scale_project"), "context_project"),
        (actor("context_evaluate", "context_profile", "context_extend"), "context_certify"),
    ):
        refused = resolve_action(action_id, covenant, actor=holder)
        assert refused.outcome == "refuse", (holder.id, action_id)
        assert refused.binding_priority == 2
    for action_id in CONTEXT_ACTIONS:
        assert resolve_action(action_id, covenant, actor=actor(action_id)).outcome == "allow"


def test_context_declarations_state_their_boundaries() -> None:
    def rationale(action_id: str) -> str:
        return " ".join(item.rationale for item in DECLARED_ACTIONS[action_id].assessments)

    assert "never raises the validated context" in rationale("context_extend")
    assert "never gated" in rationale("context_extend")
    assert "data scored by likelihood and never acted on" in rationale("context_evaluate")
    profile = rationale("context_profile")
    for phrase in (
        "address space is capped",
        "killed at a wall-clock timeout",
        "refused before running",
    ):
        assert phrase in profile, phrase
    certify = rationale("context_certify")
    for phrase in (
        "exact checkpoint",
        "within its trained context",
        "owner's CLI",
        "no config or checkpoint is edited",
    ):
        assert phrase in certify, phrase
    assert "not an approval (G5)" in rationale("context_project")


def test_undeclared_work_escalates_instead_of_running(covenant: Covenant) -> None:
    resolution = resolve_action("model_serve", covenant)
    assert resolution.outcome == "escalate"
    assert resolution.binding_priority == 1
    assert all("model_serve" not in reason for reason in resolution.reasons)
