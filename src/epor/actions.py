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
    "invoked directly by the operator who owns the machine, within the reviewed v0.0.2 action "
    "registry"
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
        system="CPU-safe, deterministic, and cooperatively cancellable",
    ),
    "tiny_eval": _declared(
        "evaluate a tiny reference checkpoint on a bounded local corpus",
        people=f"offline measurement of a fixture checkpoint; {_LOCAL_AND_OFFLINE}",
        direction=f"{_OPERATOR_REQUEST}; the checkpoint is contained below the configured root",
        system="read-only over the checkpoint and cooperatively cancellable",
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
        system="bounded token count and read-only over the checkpoint",
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


def resolve_action(action_id: str, covenant: Covenant) -> Resolution:
    """Resolve whether ``action_id`` may execute.

    An unrecognized or undeclared action is assessed as uncertain against every
    principle, so it escalates to a human rather than running.
    """

    return resolve(DECLARED_ACTIONS.get(action_id, UNDECLARED), covenant)
