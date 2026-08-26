"""G10: what happens to work the covenant will not decide on its own.

An escalation is the fail-closed path made durable and reviewable.  These
tests pin the properties that make it safe rather than merely present: it
creates no job, keeps no submitted specification, cannot launder a refusal
into permission, and cannot be spent twice.
"""

from __future__ import annotations

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Barrier

import pytest
from conftest import Harness, run

import epor.control.authority as authority
from epor.actions import DECLARED_ACTIONS
from epor.control.authority import (
    APPROVAL_TTL,
    Actor,
    AuthorityStore,
    EscalationError,
    EscalationState,
    Role,
    request_digest,
)
from epor.control.database import (
    create_session_factory,
    create_sqlite_engine,
    initialize_database,
)
from epor.control.errors import AuthorizationDeniedError
from epor.control.models import AuthorityEvent
from epor.control.models import Escalation as EscalationRow
from epor.control.models import Principal as PrincipalRow
from epor.control.service import CovenantEscalatedError, JobService
from epor.control.truth import TruthStoreError
from epor.control.worker import JobWorker
from epor.safety import PrincipleAssessment, ProposedAction


def _service(harness: Harness) -> JobService:
    return harness.app.state.job_service


@contextmanager
def escalating(action_id: str) -> Iterator[None]:
    """Temporarily make a declared action escalate, without undeclaring it.

    An undeclared action can never be approved — there is no reviewed judgement
    to approve — so testing the approval path needs work that is declared and
    still reaches a human: genuine doubt about it, rather than its absence.
    """

    from epor import actions

    original = actions.DECLARED_ACTIONS[action_id]
    actions.DECLARED_ACTIONS[action_id] = ProposedAction(
        summary=original.summary,
        protects=original.protects,
        assessments=tuple(
            PrincipleAssessment(
                priority=item.priority,
                status="uncertain" if item.priority == 1 else item.status,
                confidence=0.0 if item.priority == 1 else item.confidence,
                rationale=item.rationale,
                obligation="no-direct-harm" if item.priority == 1 else item.obligation,
            )
            for item in original.assessments
        ),
    )
    try:
        yield
    finally:
        actions.DECLARED_ACTIONS[action_id] = original


def test_undeclared_work_creates_no_job_and_no_stored_specification(harness: Harness) -> None:
    service = _service(harness)
    secret = "operator-private-note"

    with pytest.raises(CovenantEscalatedError) as raised:
        service.create_job("model_serve", {"note": secret}, actor=harness.owner)

    escalation = harness.store.escalation(str(raised.value.escalation_id))
    assert escalation.state is EscalationState.OPEN
    assert escalation.action_id == "model_serve"
    assert escalation.actor_id == harness.owner.id
    assert service.list_jobs() == []

    # Only the digest survives. The submitted specification is not persisted
    # anywhere in safety truth, so an escalation queue cannot become a leak.
    log = (harness.store.root / "authority.jsonl").read_text(encoding="utf-8")
    assert secret not in log
    assert escalation.request_digest == request_digest("model_serve", {"note": secret})


def test_approval_is_spent_once_and_only_by_the_original_requester(harness: Harness) -> None:
    service = _service(harness)
    spec = {"note": "declared later"}

    with escalating("system_probe"):
        with pytest.raises(CovenantEscalatedError) as raised:
            service.create_job("system_probe", spec, actor=harness.owner)
        escalation_id = str(raised.value.escalation_id)
        approved = harness.store.decide_escalation(
            escalation_id, approve=True, by=harness.owner, rationale="reviewed and accepted"
        )
    assert approved.state is EscalationState.APPROVED
    assert approved.approval_expires_at is not None

    digest = request_digest("system_probe", spec)
    covenant_hash = approved.covenant_sha256
    stranger = Actor(id="someone-else", role=Role.OWNER)

    # Wrong actor, wrong request, and wrong covenant all fail to match.
    assert (
        harness.store.consume_approval(
            actor=stranger,
            action_id="system_probe",
            request_digest_value=digest,
            covenant_sha256=covenant_hash,
        )
        is None
    )
    assert (
        harness.store.consume_approval(
            actor=harness.owner,
            action_id="system_probe",
            request_digest_value=request_digest("system_probe", {"note": "different"}),
            covenant_sha256=covenant_hash,
        )
        is None
    )
    assert (
        harness.store.consume_approval(
            actor=harness.owner,
            action_id="system_probe",
            request_digest_value=digest,
            covenant_sha256="0" * 64,
        )
        is None
    )

    spent = harness.store.consume_approval(
        actor=harness.owner,
        action_id="system_probe",
        request_digest_value=digest,
        covenant_sha256=covenant_hash,
    )
    assert spent is not None and spent.state is EscalationState.CONSUMED
    # One-time means one time.
    assert (
        harness.store.consume_approval(
            actor=harness.owner,
            action_id="system_probe",
            request_digest_value=digest,
            covenant_sha256=covenant_hash,
        )
        is None
    )


def test_concurrent_requesters_cannot_both_spend_one_approval(harness: Harness) -> None:
    service = _service(harness)
    with escalating("system_probe"):
        with pytest.raises(CovenantEscalatedError) as raised:
            service.create_job("system_probe", {}, actor=harness.owner)
        approved = harness.store.decide_escalation(
            str(raised.value.escalation_id),
            approve=True,
            by=harness.owner,
            rationale="race the consumers",
        )
    digest = request_digest("system_probe", {})
    barrier = Barrier(4)

    def attempt(_index: int) -> object:
        barrier.wait(timeout=5)
        return harness.store.consume_approval(
            actor=harness.owner,
            action_id="system_probe",
            request_digest_value=digest,
            covenant_sha256=approved.covenant_sha256,
        )

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(attempt, range(4)))
    assert sum(1 for item in results if item is not None) == 1


def test_an_approval_expires_and_a_refusal_can_never_be_approved(harness: Harness) -> None:
    service = _service(harness)
    with escalating("system_probe"):
        with pytest.raises(CovenantEscalatedError) as raised:
            service.create_job("system_probe", {}, actor=harness.owner)
        approved = harness.store.decide_escalation(
            str(raised.value.escalation_id),
            approve=True,
            by=harness.owner,
            rationale="short-lived",
        )
    assert timedelta(minutes=15) == APPROVAL_TTL
    later = datetime.now(UTC) + APPROVAL_TTL + timedelta(seconds=1)
    assert approved.effective_state(later) is EscalationState.EXPIRED

    original = authority.utc_now
    authority.utc_now = lambda: later  # type: ignore[assignment]
    try:
        assert (
            harness.store.consume_approval(
                actor=harness.owner,
                action_id="system_probe",
                request_digest_value=request_digest("system_probe", {}),
                covenant_sha256=approved.covenant_sha256,
            )
            is None
        )
    finally:
        authority.utc_now = original  # type: ignore[assignment]

    # A refusal never becomes an escalation, so there is nothing to approve.
    operator = Actor(id="op", role=Role.DELEGATED_OPERATOR, scopes=frozenset({"tiny_eval"}))
    with pytest.raises(AuthorizationDeniedError):
        service.create_job("tiny_train", {}, actor=operator)
    assert not [item for item in harness.store.escalations() if item.action_id == "tiny_train"]


def test_only_a_declared_action_can_be_approved_and_only_once(harness: Harness) -> None:
    service = _service(harness)
    with pytest.raises(CovenantEscalatedError) as raised:
        service.create_job("model_serve", {}, actor=harness.owner)
    escalation_id = str(raised.value.escalation_id)

    with pytest.raises(EscalationError, match="reviewed covenant declaration"):
        harness.store.decide_escalation(
            escalation_id, approve=True, by=harness.owner, rationale="tempting"
        )
    with pytest.raises(EscalationError, match="written rationale"):
        harness.store.decide_escalation(
            escalation_id, approve=False, by=harness.owner, rationale="   "
        )

    refused = harness.store.decide_escalation(
        escalation_id, approve=False, by=harness.owner, rationale="undeclared work stays undeclared"
    )
    assert refused.state is EscalationState.REFUSED
    with pytest.raises(EscalationError, match="cannot be decided"):
        harness.store.decide_escalation(
            escalation_id, approve=False, by=harness.owner, rationale="again"
        )


def test_an_approved_resubmission_runs_and_the_worker_rechecks_it(harness: Harness) -> None:
    service = _service(harness)
    with escalating("system_probe"):
        with pytest.raises(CovenantEscalatedError) as raised:
            service.create_job("system_probe", {}, actor=harness.owner)
        escalation_id = str(raised.value.escalation_id)
        harness.store.decide_escalation(
            escalation_id,
            approve=True,
            by=harness.owner,
            rationale="reviewed; permit this exact request once",
        )
        # The requester must resubmit; approval never queues work by itself.
        assert service.list_jobs() == []

        job = service.create_job("system_probe", {}, actor=harness.owner)
        assert job.escalation_id == escalation_id
        assert harness.store.escalation(escalation_id).consumed_job_id == job.id

        # The worker re-resolves and still escalates, but the spent approval
        # covers this dispatch because the covenant has not moved under it.
        worker = JobWorker(harness.settings, service=service, worker_id="approved-worker")
        executed = worker.run_once()
        assert executed is not None and executed.id == job.id
        assert executed.status.value == "succeeded"


def test_the_worker_stops_work_whose_delegation_was_revoked(harness: Harness) -> None:
    service = _service(harness)
    token = harness.delegate("system_probe", label="soon revoked")
    operator = harness.store.authenticate(token)
    assert operator is not None
    job = service.create_job("system_probe", {}, actor=operator.actor())

    harness.store.revoke(operator.id, by=harness.owner)
    worker = JobWorker(harness.settings, service=service, worker_id="revoked-worker")
    executed = worker.run_once()
    assert executed is not None and executed.id == job.id
    assert executed.status.value == "failed"
    assert executed.error_code == "authentication_required"

    # The dispatch-time resolution is recorded, not only the admission one.
    dispatch = [
        item
        for item in harness.store.records()
        if item.get("kind") == "safety.decision" and item.get("surface") == "worker"
    ]
    assert dispatch and dispatch[-1]["outcome"] == "refuse"
    assert dispatch[-1]["covenant_sha256"] == service.covenant.sha256


def test_the_sqlite_index_rebuilds_from_truth_and_fails_closed_when_broken(
    harness: Harness, tmp_path: Path
) -> None:
    service = _service(harness)
    harness.delegate("system_probe", label="indexed")
    with pytest.raises(CovenantEscalatedError):
        service.create_job("model_serve", {}, actor=harness.owner)
    service.restore_authority_index()

    def rows(active: JobService) -> tuple[list[str], list[str], int]:
        with active._sessions() as session:
            principals = sorted(item.label for item in session.query(PrincipalRow).all())
            escalations = sorted(item.action_id for item in session.query(EscalationRow).all())
            events = session.query(AuthorityEvent).count()
        return principals, escalations, events

    before = rows(service)
    assert before[0] == ["indexed", "project owner"]
    assert before[1] == ["model_serve"]

    # Deleting the index loses nothing: the log is the authority.
    rebuilt_database = tmp_path / "rebuilt.sqlite3"
    engine = create_sqlite_engine(rebuilt_database)
    initialize_database(engine)
    from dataclasses import replace as replace_setting

    rebuilt = JobService(
        create_session_factory(engine),
        replace_setting(harness.settings, database_path=rebuilt_database),
    )
    assert rows(rebuilt) == before

    # A tampered chain authorizes nothing at all rather than degrading.
    log = harness.store.root / "authority.jsonl"
    lines = log.read_text(encoding="utf-8").splitlines()
    lines[0] = lines[0].replace('"label":"project owner"', '"label":"impostor"')
    log.write_text("\n".join(lines) + "\n", encoding="utf-8")
    tampered = AuthorityStore(harness.store.root)
    with pytest.raises(TruthStoreError, match="hash chain"):
        tampered.principals()


def test_the_escalation_queue_is_owner_reviewable_over_the_api(harness: Harness) -> None:
    service = _service(harness)
    with pytest.raises(CovenantEscalatedError) as raised:
        service.create_job("model_serve", {"private": "not stored"}, actor=harness.owner)
    escalation_id = str(raised.value.escalation_id)
    operator_token = harness.delegate("system_probe")

    async def scenario() -> None:
        async with harness.client() as anonymous:
            listing = await anonymous.get("/api/v1/safety/escalations?state=open")
            assert listing.status_code == 200
            assert listing.json()["count"] == 1
            assert "not stored" not in listing.text

            detail = await anonymous.get(f"/api/v1/safety/escalations/{escalation_id}")
            assert detail.status_code == 200
            assert detail.json()["state"] == "open"

            blocked = await anonymous.post(
                f"/api/v1/safety/escalations/{escalation_id}/decision",
                json={"approve": False, "rationale": "not mine to decide"},
            )
            assert blocked.status_code == 401

        async with harness.as_bearer(operator_token) as operator:
            denied = await operator.post(
                f"/api/v1/safety/escalations/{escalation_id}/decision",
                json={"approve": False, "rationale": "still not mine"},
            )
            assert denied.status_code == 403
            assert denied.json()["code"] == "authorization_denied"

        async with harness.as_owner() as owner:
            decided = await owner.post(
                f"/api/v1/safety/escalations/{escalation_id}/decision",
                json={"approve": False, "rationale": "undeclared work stays undeclared"},
            )
            assert decided.status_code == 200
            assert decided.json()["state"] == "refused"
            assert decided.json()["decided_by"] == harness.owner.id

            repeated = await owner.post(
                f"/api/v1/safety/escalations/{escalation_id}/decision",
                json={"approve": True, "rationale": "changed my mind"},
            )
            assert repeated.status_code == 409

    run(scenario)


def test_the_covenant_and_action_registry_are_served_verbatim(harness: Harness) -> None:
    service = _service(harness)

    async def scenario() -> None:
        async with harness.client() as client:
            covenant = (await client.get("/api/v1/safety/covenant")).json()
            assert covenant["sha256"] == service.covenant.sha256
            assert [item["priority"] for item in covenant["principles"]] == [1, 2, 3]
            assert [item["title"] for item in covenant["principles"]] == [
                "Protect people",
                "Follow legitimate human direction",
                "Preserve the system responsibly",
            ]
            assert covenant["limitations"] == service.covenant.limitations
            assert covenant["ratification"]["present"] is True
            assert len(covenant["enforcement_checkpoints"]) >= 5

            actions = (await client.get("/api/v1/safety/actions")).json()
            assert actions["count"] == len(DECLARED_ACTIONS)
            probe = next(item for item in actions["items"] if item["action_id"] == "system_probe")
            assert probe["job_type"] is True
            assert "epor doctor" in probe["cli_commands"]
            assert [item["priority"] for item in probe["assessments"]] == [1, 2, 3]
            assert "escalates" in actions["undeclared_behavior"]

    run(scenario)
