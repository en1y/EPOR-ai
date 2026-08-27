"""Every action EPOR can execute, and its reviewed covenant standing.

This is the single registry behind both entry points: the control plane's job
admission and the CLI.  Neither has its own table, so an action cannot be
governed on one path and ungoverned on the other.

Assessments are declared and reviewed here rather than inferred, which is only
defensible because the v0.0.2 action space is a closed list of local,
operator-directed work.  Its network listeners are fixed to numeric loopback
addresses, and its job work remains offline and resource-bounded.  An action
with no declaration resolves to escalation: adding executable work to EPOR
requires stating its covenant standing first, and forgetting to fails closed
rather than open.

The declarations are reviewed judgements, not measurements.  None of them make
the executed work safe on their own.
"""

from __future__ import annotations

from typing import Protocol

from .safety import (
    PRIORITIES,
    Covenant,
    PrincipleAssessment,
    Priority,
    ProposedAction,
    Resolution,
    resolve,
)


def _declared(
    summary: str,
    *,
    people: str,
    direction: str,
    system: str,
    protects: tuple[Priority, ...] = (),
) -> ProposedAction:
    """Declare an action whose every principle is assessed as satisfied."""

    rationales = (people, direction, system)
    return ProposedAction(
        summary=summary,
        protects=protects,
        assessments=tuple(
            PrincipleAssessment(
                priority=priority,
                status="satisfied",
                confidence=0.99,
                rationale=rationale,
            )
            for priority, rationale in zip(PRIORITIES, rationales, strict=True)
        ),
    )


_LOCAL_AND_OFFLINE = (
    "runs locally under the operator's own account, reaches no person other than "
    "that operator, and emits no personal data"
)
_OPERATOR_REQUEST = (
    "invoked directly by the operator who owns the machine, within the reviewed action registry"
)

DECLARED_ACTIONS: dict[str, ProposedAction] = {
    "system_probe": _declared(
        "read local capability information",
        people=f"read-only inspection of this machine; {_LOCAL_AND_OFFLINE}",
        direction=_OPERATOR_REQUEST,
        system="read-only and side-effect free",
    ),
    "research_sync": _declared(
        "download allowlisted research sources into ignored local paths",
        people=(
            "fetches published research from exact allowlisted hosts into paths "
            "excluded from Git; extraction is inert and stores no personal data"
        ),
        direction=(
            f"{_OPERATOR_REQUEST}; the source list is reviewed configuration and no "
            "caller may supply a URL, so text inside a fetched document can never "
            "become an instruction"
        ),
        system="bounded sizes, atomic replacement, and checksum verification",
    ),
    "tiny_train": _declared(
        "train the tiny reference model on a bounded local corpus",
        people=(
            f"a 10-50M parameter fixture model trained offline; {_LOCAL_AND_OFFLINE}, "
            "and the result is neither released nor served"
        ),
        direction=f"{_OPERATOR_REQUEST}; the corpus is capped and contained below the roots",
        system=(
            "fixed model, corpus, checkpoint, batch-token, and step limits keep the reference "
            "path CPU-safe; control jobs add checkpoint-boundary cooperative cancellation"
        ),
    ),
    "tiny_eval": _declared(
        "evaluate a tiny reference checkpoint on a bounded local corpus",
        people=f"offline measurement of a fixture checkpoint; {_LOCAL_AND_OFFLINE}",
        direction=f"{_OPERATOR_REQUEST}; the checkpoint is contained below the configured root",
        system=(
            "bounded batches and inputs; read-only over the checkpoint, with cooperative "
            "cancellation in control jobs"
        ),
    ),
    "generate": _declared(
        "greedily generate text from a tiny reference checkpoint",
        people=(
            "the only declared action whose output a person reads. A 10-50M fixture "
            "model over a debug tokenizer produces text with no informational "
            "content, printed locally to the operator who asked for it and never "
            "presented as fact. This rationale does not survive a trained model: a "
            "released checkpoint requires output classification under "
            "no-harmful-output before this declaration can be reused"
        ),
        direction=f"{_OPERATOR_REQUEST}; the prompt comes from the operator's own command line",
        system="bounded checkpoint and token count, and read-only over the checkpoint",
    ),
    "control_api_start": _declared(
        "start the typed control API on numeric loopback",
        people=(
            "binds only to numeric loopback on the documented trusted single-user machine and "
            "admits only separately declared job types; it accepts neither shell commands nor "
            "arbitrary URLs"
        ),
        direction=(
            f"{_OPERATOR_REQUEST}; every submitted job is independently authorized before its "
            "specification is parsed"
        ),
        system=(
            "fixed loopback binding, exact local UI origin, durable SQLite state, and unrestricted "
            "shutdown"
        ),
    ),
    "job_worker_start": _declared(
        "start the local allowlisted job worker",
        people=(
            "dispatches only typed jobs whose own declarations are re-resolved immediately before "
            "execution and exposes no network listener"
        ),
        direction=(
            f"{_OPERATOR_REQUEST}; queued work carries its admission audit and is independently "
            "authorized again at dispatch"
        ),
        system=(
            "closed handler map, bounded polling, cooperative cancellation, and unrestricted "
            "shutdown"
        ),
    ),
    "local_ui_start": _declared(
        "start the browser console on its fixed loopback origin",
        people=(
            "uses a fixed loopback origin on the trusted single-user machine and can submit typed "
            "control jobs but no shell command or arbitrary URL"
        ),
        direction=f"{_OPERATOR_REQUEST}; browser actions remain subject to API job admission",
        system="fixed loopback origin, pinned UI dependencies, and unrestricted shutdown",
    ),
    "config_validate": _declared(
        "strictly validate a tracked model recipe without allocating weights",
        people=f"reads tracked configuration and allocates nothing; {_LOCAL_AND_OFFLINE}",
        direction=f"{_OPERATOR_REQUEST}; the recipe path resolves beneath the project root",
        system="read-only, offline, and side-effect free apart from printed output",
    ),
    "research_verify": _declared(
        "verify local research files against their tracked checksums",
        people=(
            "recomputes digests over already-downloaded files; it fetches nothing "
            f"and {_LOCAL_AND_OFFLINE}"
        ),
        direction=f"{_OPERATOR_REQUEST}; the source list is reviewed configuration",
        system="read-only over ignored local paths and safe to repeat",
    ),
    "research_index": _declared(
        "extract text from verified local research files into ignored paths",
        people=(
            "extraction is inert: no script, macro, external reference, or active "
            "content is evaluated, and text inside an extracted document stays "
            "data rather than instruction"
        ),
        direction=f"{_OPERATOR_REQUEST}; only catalog-listed local files are read",
        system="bounded byte limits, atomic replacement, and Git-excluded output",
    ),
    "data_ingest": _declared(
        "register and ingest reviewed local data into immutable private layers",
        people=(
            "requires a rights and removal registration before reading local bytes; PII and "
            "credential values are removed, unsafe content is quarantined, and ordinary logs "
            "retain only finding kinds and counts"
        ),
        direction=(
            f"{_OPERATOR_REQUEST}; every input is a contained local file and text inside it is "
            "always data, never an instruction"
        ),
        system=(
            "streaming size limits, private content-addressed layers, immutable provenance, "
            "cooperative cancellation, and no network acquisition"
        ),
    ),
    "data_build": _declared(
        "build deterministic filtered dataset splits and a dataset card",
        people=(
            "uses only training-eligible reviewed registrations after sensitive-value removal, "
            "quarantine, tombstones, and duplicate controls; it publishes nothing"
        ),
        direction=f"{_OPERATOR_REQUEST}; build policy is typed and resource-bounded",
        system=(
            "immutable parent hashes, global deduplication before stable split assignment, "
            "idempotent outputs, and cooperative cancellation"
        ),
    ),
    "data_remove": _declared(
        "record an immutable removal tombstone for all future dataset builds",
        people=(
            "honors a reviewed removal direction without exposing the removed content and "
            "prevents the target entering future builds"
        ),
        direction=f"{_OPERATOR_REQUEST}; the target and rationale are explicit and auditable",
        system="append-only tombstones preserve lineage without mutating historical evidence",
        protects=(1, 2),
    ),
    "data_audit": _declared(
        "verify local data-layer hashes and summarize provenance state",
        people=f"reads only local metadata and content hashes; {_LOCAL_AND_OFFLINE}",
        direction=f"{_OPERATOR_REQUEST}; the audit changes no dataset record",
        system="read-only integrity verification over the private data root",
    ),
    "owner_bootstrap": _declared(
        "mint the single local owner credential on first run",
        people=(
            "writes one random secret to a private file owned by the operator "
            "running the command; it reaches no person and no network"
        ),
        direction=(
            "run from the shell of the machine owner, which is the only place a "
            "first credential can legitimately originate; it refuses to run twice, "
            "so it cannot be used to displace an existing owner"
        ),
        system=(
            "private 0600 credential file beneath a 0700 directory, recorded in "
            "hash-chained safety truth as a digest and never in clear"
        ),
    ),
}

# One command-to-action registry behind the CLI, so a new executing command
# without a declared covenant standing fails closed rather than running
# ungoverned.  Help, version, and the roadmap placeholders execute nothing and
# are deliberately absent.
CLI_ACTIONS: dict[str, str] = {
    "doctor": "system_probe",
    "config validate": "config_validate",
    "research sync": "research_sync",
    "research verify": "research_verify",
    "research index": "research_index",
    "data ingest": "data_ingest",
    "data build": "data_build",
    "data remove": "data_remove",
    "data audit": "data_audit",
    "train pretrain": "tiny_train",
    "train resume": "tiny_train",
    "eval run": "tiny_eval",
    "generate": "generate",
    "api": "control_api_start",
    "worker": "job_worker_start",
    "ui": "local_ui_start",
    "auth bootstrap": "owner_bootstrap",
}

UNDECLARED = ProposedAction(
    summary="execute work that has no reviewed covenant declaration",
    assessments=tuple(
        PrincipleAssessment(
            priority=priority,
            status="uncertain",
            confidence=0.0,
            rationale="no reviewed declaration states how this work stands under the covenant",
            obligation=obligation,
        )
        for priority, obligation in zip(
            PRIORITIES,
            ("no-direct-harm", "verify-authorization", "fail-safely"),
            strict=True,
        )
    ),
)


class ActorEvidence(Protocol):
    """What the covenant needs to know about who is asking.

    Structural rather than imported, so the action registry stays free of the
    control plane; :class:`epor.control.authority.Actor` satisfies it.
    """

    @property
    def id(self) -> str: ...

    @property
    def is_authenticated(self) -> bool: ...

    def may(self, action_id: str) -> bool: ...


def _with_actor(action: ProposedAction, action_id: str, actor: ActorEvidence) -> ProposedAction:
    """Fold identity and scope into the second principle's assessment.

    Authorization is not a separate check bolted on after resolution. "Follow
    legitimate human direction" is precisely the principle an unverified or
    out-of-scope request engages, so the evidence belongs inside that
    assessment, where the resolver's ordering can act on it: a Principle 2
    conflict with nothing higher-ranked to justify it refuses, and it can never
    be talked past by a Principle 3 argument about availability or sunk work.
    """

    declared = action.assessment(2)
    if actor.may(action_id):
        replacement = PrincipleAssessment(
            priority=2,
            status=declared.status,
            confidence=declared.confidence,
            obligation=declared.obligation,
            rationale=f"{declared.rationale}; requested by verified actor {actor.id}",
        )
    elif not actor.is_authenticated:
        replacement = PrincipleAssessment(
            priority=2,
            status="conflicted",
            confidence=0.99,
            obligation="verify-authorization",
            rationale="the requester's identity is unverified, so its instruction is not direction",
        )
    else:
        replacement = PrincipleAssessment(
            priority=2,
            status="conflicted",
            confidence=0.99,
            obligation="refuse-unlawful-or-out-of-scope",
            rationale=(
                f"actor {actor.id} holds no delegation covering {action_id!r}, "
                "so the request lies outside its authorized scope"
            ),
        )
    assessments = tuple(replacement if item.priority == 2 else item for item in action.assessments)
    return action.model_copy(update={"assessments": assessments})


def resolve_action(
    action_id: str,
    covenant: Covenant,
    *,
    actor: ActorEvidence | None = None,
) -> Resolution:
    """Resolve whether ``action_id`` may execute, for ``actor`` if supplied.

    An unrecognized or undeclared action is assessed as uncertain against every
    principle, so it escalates to a human rather than running.  Omitting the
    actor resolves the declaration on its own terms; callers that have an
    identity must pass it, because a declaration alone says nothing about
    whether this requester was authorized to invoke it.
    """

    action = DECLARED_ACTIONS.get(action_id, UNDECLARED)
    if actor is not None:
        action = _with_actor(action, action_id, actor)
    return resolve(action, covenant)


def authorization_denied(resolution: Resolution) -> bool:
    """Whether a refusal was decided by scope rather than by the work itself.

    Used only to pick the reported error code.  The refusal itself was already
    decided by the covenant; this reads which obligation bound it.
    """

    return (
        resolution.outcome == "refuse"
        and resolution.binding_priority == 2
        and resolution.binding_obligation
        in {"refuse-unlawful-or-out-of-scope", "verify-authorization"}
    )
