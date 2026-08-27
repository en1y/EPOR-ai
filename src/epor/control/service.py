"""Transactional job lifecycle and artifact-index operations."""

from __future__ import annotations

import hashlib
import mimetypes
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

from pydantic import ValidationError
from sqlalchemy import Select, delete, select, text, update
from sqlalchemy.engine import CursorResult
from sqlalchemy.orm import Session, sessionmaker

from epor.actions import authorization_denied, resolve_action
from epor.safety import Covenant, Resolution, load_covenant

from .authority import ANONYMOUS, Actor, AuthorityStore, Escalation, EscalationState, request_digest
from .errors import (
    ArtifactPathError,
    AuthenticationRequiredError,
    AuthorizationDeniedError,
    ControlError,
    CovenantBlockedError,
    CovenantEscalatedError,
    CovenantRefusedError,
    InvalidJobSpecError,
    InvalidTransitionError,
    JobNotFoundError,
)
from .models import (
    TERMINAL_JOB_STATUSES,
    AuthorityEvent,
    Job,
    JobArtifact,
    JobEvent,
    JobStatus,
    JobType,
    utc_now,
)
from .models import (
    Escalation as EscalationRow,
)
from .models import (
    Principal as PrincipalRow,
)
from .schemas import JOB_SPEC_MODELS, TINY_CONTROL_CORPUS_MAX_BYTES
from .security import PathOutsideRootError, contained_path, redact, redact_text
from .settings import ControlSettings
from .truth import JobTruthStore, TruthStoreError, parse_utc

__all__ = [
    "ArtifactPathError",
    "AuthenticationRequiredError",
    "AuthorizationDeniedError",
    "ControlError",
    "CovenantBlockedError",
    "CovenantEscalatedError",
    "CovenantRefusedError",
    "InvalidJobSpecError",
    "InvalidTransitionError",
    "JobNotFoundError",
    "JobService",
    "re_safe_code",
]


_TRANSITIONS: dict[JobStatus, frozenset[JobStatus]] = {
    JobStatus.QUEUED: frozenset({JobStatus.STARTING, JobStatus.CANCELLED}),
    JobStatus.STARTING: frozenset(
        {
            JobStatus.RUNNING,
            JobStatus.CANCELLING,
            JobStatus.FAILED,
            JobStatus.INTERRUPTED,
        }
    ),
    JobStatus.RUNNING: frozenset(
        {
            JobStatus.SUCCEEDED,
            JobStatus.FAILED,
            JobStatus.CANCELLING,
            JobStatus.INTERRUPTED,
        }
    ),
    JobStatus.CANCELLING: frozenset({JobStatus.CANCELLED, JobStatus.FAILED, JobStatus.INTERRUPTED}),
}


class JobService:
    def __init__(
        self,
        session_factory: sessionmaker[Session],
        settings: ControlSettings,
    ) -> None:
        self._sessions = session_factory
        self.settings = settings
        assert settings.artifact_root is not None
        assert settings.safety_root is not None
        # No covenant, no control plane. A missing, malformed, or unratifiable
        # covenant raises here rather than degrading to an ungoverned service.
        self.covenant = load_covenant()
        self.authority = AuthorityStore(settings.safety_root)
        self._truth = JobTruthStore(settings.artifact_root)
        self.restore_file_index()
        self.restore_authority_index()

    @contextmanager
    def _write_session(self) -> Iterator[Session]:
        """Reserve SQLite's single writer slot before reading mutable job state.

        A deferred transaction lets two callers observe the same status and
        event cursor before either writes. ``BEGIN IMMEDIATE`` makes competing
        lifecycle calls wait at transaction entry, so the later caller always
        validates against the earlier caller's committed state.
        """

        session = self._sessions()
        try:
            session.execute(text("BEGIN IMMEDIATE"))
            yield session
            session.commit()
        except BaseException:
            session.rollback()
            raise
        finally:
            session.close()

    def validate_spec(self, job_type: JobType, raw_spec: dict[str, Any]) -> dict[str, Any]:
        model_type = JOB_SPEC_MODELS[job_type]
        try:
            spec = model_type.model_validate(raw_spec)
        except ValidationError as exc:
            raise InvalidJobSpecError(
                "job specification is invalid",
                details=exc.errors(include_url=False, include_input=False),
            ) from exc

        value = spec.model_dump(mode="json", by_alias=True, exclude_none=True)
        try:
            if job_type is JobType.TINY_TRAIN:
                contained_path(self.settings.project_root, value["model_config"])
                self._validate_tiny_corpus(value["corpus_path"])
            elif job_type is JobType.TINY_EVAL:
                assert self.settings.artifact_root is not None
                contained_path(self.settings.artifact_root, value["checkpoint_path"])
                self._validate_tiny_corpus(value["corpus_path"])
                if value.get("model_config"):
                    contained_path(self.settings.project_root, value["model_config"])
            elif job_type is JobType.DATA_INGEST:
                registration = contained_path(
                    self.settings.project_root,
                    value["registration_path"],
                    must_exist=True,
                )
                if not registration.is_file() or registration.is_symlink():
                    raise InvalidJobSpecError(
                        "data registration must be a regular non-symlink file"
                    )
                for supplied in value["input_paths"]:
                    input_path = contained_path(
                        self.settings.project_root,
                        supplied,
                        must_exist=True,
                    )
                    if not input_path.is_file() or input_path.is_symlink():
                        raise InvalidJobSpecError(
                            "data inputs must be regular non-symlink files"
                        )
        except (FileNotFoundError, OSError, PathOutsideRootError) as exc:
            raise InvalidJobSpecError(str(exc)) from exc
        return value

    def _validate_tiny_corpus(self, supplied: str) -> Path:
        corpus = contained_path(self.settings.project_root, supplied, must_exist=True)
        if not corpus.is_file():
            raise InvalidJobSpecError("tiny control corpus must be a regular file")
        size = corpus.stat().st_size
        if size > TINY_CONTROL_CORPUS_MAX_BYTES:
            raise InvalidJobSpecError("tiny control corpus exceeds the 1 MiB v0.0.1 limit")
        return corpus

    def _event(
        self,
        session: Session,
        job: Job,
        kind: str,
        payload: dict[str, Any] | None = None,
    ) -> JobEvent:
        # Flush the job first so new rows exist and lifecycle changes remain in
        # the same rollback boundary as the event. The SQL increment is atomic
        # even if a future caller bypasses ORM identity synchronization.
        session.flush()
        incremented = cast(
            CursorResult[Any],
            session.execute(
                update(Job)
                .where(Job.id == job.id)
                .values(event_cursor=Job.event_cursor + 1)
                .execution_options(synchronize_session=False)
            ),
        )
        if incremented.rowcount != 1:
            raise JobNotFoundError(f"job {job.id!r} was not found")
        session.refresh(job, attribute_names=["event_cursor"])
        event = JobEvent(
            job_id=job.id,
            sequence=job.event_cursor,
            kind=kind,
            payload=redact(payload or {}),
            created_at=utc_now(),
        )
        session.add(event)
        session.flush()
        self._truth.append_event(job, event)
        return event

    def restore_file_index(self) -> int:
        """Deterministically reconcile file truth into the SQLite index.

        File loading occurs while holding SQLite's writer reservation. Every
        compliant event append also holds that reservation, so the imported
        prefix cannot become stale between reading and committing the index.
        """

        restored = 0
        with self._write_session() as session:
            truth_jobs = self._truth.load_jobs()
            truth_counts = {str(item.manifest["id"]): len(item.events) for item in truth_jobs}
            database_jobs = list(session.scalars(select(Job).order_by(Job.created_at, Job.id)))
            for database_job in database_jobs:
                file_count = truth_counts.get(database_job.id, 0)
                if database_job.event_cursor <= file_count:
                    continue
                database_events = list(
                    session.scalars(
                        select(JobEvent)
                        .where(
                            JobEvent.job_id == database_job.id,
                            JobEvent.sequence > file_count,
                        )
                        .order_by(JobEvent.sequence)
                    )
                )
                artifacts = {
                    item.id: item
                    for item in session.scalars(
                        select(JobArtifact).where(JobArtifact.job_id == database_job.id)
                    )
                }
                for database_event in database_events:
                    file_event = database_event
                    if database_event.kind == "artifact.created":
                        payload = dict(database_event.payload)
                        artifact_id = str(payload.get("artifact_id", ""))
                        artifact = artifacts.get(artifact_id)
                        if artifact is not None:
                            payload.update(
                                {
                                    "kind": artifact.kind,
                                    "path": artifact.path,
                                    "sha256": artifact.sha256,
                                    "size_bytes": artifact.size_bytes,
                                    "media_type": artifact.media_type,
                                    "created_at": artifact.created_at.isoformat().replace(
                                        "+00:00", "Z"
                                    ),
                                }
                            )
                            file_event = JobEvent(
                                job_id=database_event.job_id,
                                sequence=database_event.sequence,
                                kind=database_event.kind,
                                payload=payload,
                                created_at=database_event.created_at,
                            )
                    self._truth.append_event(database_job, file_event)
            if any(
                database_job.event_cursor > truth_counts.get(database_job.id, 0)
                for database_job in database_jobs
            ):
                truth_jobs = self._truth.load_jobs()
            for truth_job in truth_jobs:
                latest = truth_job.events[-1]
                state = latest.get("state")
                if not isinstance(state, dict):
                    raise TruthStoreError("job event state snapshot is missing")
                job_id = str(truth_job.manifest.get("id"))
                if state.get("id") != job_id or state.get("event_cursor") != len(truth_job.events):
                    raise TruthStoreError("job state snapshot identity or cursor is invalid")
                job = session.get(Job, job_id)
                values = self._job_values_from_truth(state)
                if job is None:
                    job = Job(id=job_id, **values)
                    session.add(job)
                else:
                    # Heartbeats intentionally do not create user-visible events.
                    # When the indexed cursor already matches file truth, retain
                    # a newer DB-only liveness timestamp across service restart.
                    if (
                        job.event_cursor == int(state["event_cursor"])
                        and job.updated_at > values["updated_at"]
                    ):
                        values["updated_at"] = job.updated_at
                        values["heartbeat_at"] = job.heartbeat_at
                    for name, value in values.items():
                        setattr(job, name, value)
                restored += 1

            # Flush every job first so retry foreign keys are resolvable before
            # replacing the derived event and artifact indexes.
            session.flush()
            for truth_job in truth_jobs:
                job_id = str(truth_job.manifest["id"])
                session.execute(delete(JobEvent).where(JobEvent.job_id == job_id))
                session.execute(delete(JobArtifact).where(JobArtifact.job_id == job_id))
                for record in truth_job.events:
                    created_at = parse_utc(record.get("created_at"))
                    if created_at is None:
                        raise TruthStoreError("job event created_at is required")
                    payload = record.get("payload")
                    if not isinstance(payload, dict):
                        raise TruthStoreError("job event payload must be an object")
                    session.add(
                        JobEvent(
                            job_id=job_id,
                            sequence=int(record["sequence"]),
                            kind=str(record["kind"]),
                            payload=payload,
                            created_at=created_at,
                        )
                    )
                    if record.get("kind") == "artifact.created":
                        session.add(self._artifact_from_truth(job_id, payload, created_at))
        return restored

    @staticmethod
    def _job_values_from_truth(state: dict[str, Any]) -> dict[str, Any]:
        created_at = parse_utc(state.get("created_at"))
        updated_at = parse_utc(state.get("updated_at"))
        if created_at is None or updated_at is None:
            raise TruthStoreError("job state requires created_at and updated_at")
        spec = state.get("spec")
        result = state.get("result")
        if not isinstance(spec, dict) or (result is not None and not isinstance(result, dict)):
            raise TruthStoreError("job state spec/result shape is invalid")
        return {
            "type": JobType(str(state["type"])),
            "status": JobStatus(str(state["status"])),
            "spec": spec,
            "result": result,
            "progress": float(state["progress"]),
            "error_code": state.get("error_code"),
            "error_message": state.get("error_message"),
            "worker_id": state.get("worker_id"),
            "event_cursor": int(state["event_cursor"]),
            "retry_of_id": state.get("retry_of_id"),
            # Absent in v0.0.1 truth, which must keep restoring unchanged.
            "submitted_by_id": state.get("submitted_by_id"),
            "submitted_by_role": state.get("submitted_by_role"),
            "admission_covenant_sha256": state.get("admission_covenant_sha256"),
            "escalation_id": state.get("escalation_id"),
            "created_at": created_at,
            "updated_at": updated_at,
            "started_at": parse_utc(state.get("started_at")),
            "finished_at": parse_utc(state.get("finished_at")),
            "heartbeat_at": parse_utc(state.get("heartbeat_at")),
            "cancel_requested_at": parse_utc(state.get("cancel_requested_at")),
        }

    @staticmethod
    def _artifact_from_truth(
        job_id: str, payload: dict[str, Any], fallback_created_at: datetime
    ) -> JobArtifact:
        required = {"artifact_id", "kind", "path", "sha256", "size_bytes"}
        if not required <= payload.keys():
            raise TruthStoreError("artifact event is missing required fields")
        created_at = parse_utc(payload.get("created_at")) or fallback_created_at
        return JobArtifact(
            id=str(payload["artifact_id"]),
            job_id=job_id,
            kind=str(payload["kind"]),
            path=str(payload["path"]),
            sha256=str(payload["sha256"]),
            size_bytes=int(payload["size_bytes"]),
            media_type=(
                str(payload["media_type"]) if payload.get("media_type") is not None else None
            ),
            created_at=created_at,
        )

    @staticmethod
    def _require_job(session: Session, job_id: str) -> Job:
        job = session.get(Job, job_id)
        if job is None:
            raise JobNotFoundError(f"job {job_id!r} was not found")
        return job

    @staticmethod
    def _transition(job: Job, target: JobStatus, now: datetime) -> None:
        allowed = _TRANSITIONS.get(job.status, frozenset())
        if target not in allowed:
            raise InvalidTransitionError(
                f"job cannot transition from {job.status.value} to {target.value}"
            )
        job.status = target
        job.updated_at = now
        if target is JobStatus.STARTING:
            job.started_at = now
            job.heartbeat_at = now
        if target in TERMINAL_JOB_STATUSES:
            job.finished_at = now
            job.heartbeat_at = now
            if target is JobStatus.SUCCEEDED:
                job.progress = 1.0

    def create_job(
        self,
        action_id: str | JobType,
        raw_spec: dict[str, Any] | None = None,
        *,
        actor: Actor = ANONYMOUS,
        retry_of_id: str | None = None,
    ) -> Job:
        """Admit, then validate, then queue.

        ``action_id`` stays a plain string until the covenant has spoken.  A
        request naming work EPOR does not declare must reach the resolver and
        be turned away by it, not fail on an enum conversion that never
        consulted the covenant — and its specification must not be parsed, let
        alone stored, before then.
        """

        requested = action_id.value if isinstance(action_id, JobType) else str(action_id)
        spec = dict(raw_spec or {})
        resolution, escalation = self.admit(requested, spec=spec, actor=actor)
        try:
            job_type = JobType(requested)
        except ValueError as exc:
            # Declared and permitted, but not work the control plane runs; the
            # CLI owns it. Reported as a bad request, never as a crash.
            raise InvalidJobSpecError(
                "this action is declared but is not a control-plane job type"
            ) from exc
        validated = self.validate_spec(job_type, spec)
        now = utc_now()
        job = Job(
            id=str(uuid4()),
            type=job_type,
            status=JobStatus.QUEUED,
            spec=validated,
            progress=0.0,
            event_cursor=0,
            retry_of_id=retry_of_id,
            submitted_by_id=actor.id if actor.is_authenticated else None,
            submitted_by_role=actor.role.value if actor.is_authenticated else None,
            admission_covenant_sha256=resolution.covenant_sha256,
            escalation_id=escalation.id if escalation is not None else None,
            created_at=now,
            updated_at=now,
        )
        with self._write_session() as session:
            if retry_of_id is not None:
                self._require_job(session, retry_of_id)
            session.add(job)
            self._event(
                session,
                job,
                "job.queued",
                {
                    "type": job_type.value,
                    "covenant": resolution.model_dump(mode="json"),
                    "submitted_by": job.submitted_by_id,
                    "escalation_id": job.escalation_id,
                },
            )
        if escalation is not None:
            self.authority.attach_job(escalation.id, job.id)
        return job

    def admit(
        self,
        action_id: str,
        *,
        spec: Any = None,
        actor: Actor = ANONYMOUS,
        surface: str = "api",
    ) -> tuple[Resolution, Escalation | None]:
        """Resolve ``action_id`` for ``actor``, raising unless work may proceed.

        Every path that starts work goes through here, including retry and the
        worker's dispatch recheck.  The covenant file is reloaded on each call
        so a corrected covenant governs work that is already queued, and every
        resolution is written to durable safety truth whatever it decided — a
        gate that records only its refusals cannot be audited for what it let
        through.

        Returns the resolution and, when a prior human approval was spent to
        get here, the escalation it came from.
        """

        covenant = self.reload_covenant()
        digest = request_digest(action_id, spec)
        resolution = resolve_action(action_id, covenant, actor=actor)
        self.authority.record_decision(
            actor=actor,
            action_id=action_id,
            surface=surface,
            outcome=resolution.outcome,
            binding_priority=resolution.binding_priority,
            reasons=resolution.reasons,
            covenant_sha256=resolution.covenant_sha256,
            request_digest_value=digest,
        )
        if resolution.outcome == "allow":
            return resolution, None
        if resolution.outcome == "refuse":
            # A refusal is the covenant's own decision about the work itself.
            # No approval path exists for it, deliberately.
            if authorization_denied(resolution):
                if not actor.is_authenticated:
                    raise AuthenticationRequiredError(
                        "this action requires a verified operator identity"
                    )
                raise AuthorizationDeniedError(
                    f"actor holds no delegation covering {action_id!r}",
                    details=resolution.model_dump(mode="json"),
                )
            raise CovenantRefusedError(resolution)

        approved = self.authority.consume_approval(
            actor=actor,
            action_id=action_id,
            request_digest_value=digest,
            covenant_sha256=resolution.covenant_sha256,
        )
        if approved is not None:
            return resolution, approved
        opened = self.authority.open_escalation(
            actor=actor,
            action_id=action_id,
            request_digest_value=digest,
            covenant_sha256=resolution.covenant_sha256,
            binding_priority=resolution.binding_priority,
            reasons=resolution.reasons,
        )
        raise CovenantEscalatedError(resolution, escalation_id=opened.id)

    def reload_covenant(self) -> Covenant:
        """Re-read the tracked covenant so a correction takes effect at once."""

        self.covenant = load_covenant()
        return self.covenant

    def actor_for(self, job: Job) -> Actor:
        """Re-derive a queued job's submitter at dispatch time.

        A delegation revoked or expired since admission yields the anonymous
        actor, which the covenant's second principle then refuses.  Trusting
        the recorded role instead would let a revoked operator's queued work
        keep running.
        """

        if job.submitted_by_id is None:
            return ANONYMOUS
        principals = {item.id: item for item in self.authority.principals()}
        principal = principals.get(job.submitted_by_id)
        if principal is None or not principal.active():
            return ANONYMOUS
        return principal.actor()

    def approved_at_dispatch(self, job: Job, resolution: Resolution) -> bool:
        """Whether a still-valid consumed approval covers this dispatch.

        The approval is spent once, at admission; the worker re-reads it rather
        than spending another.  It counts only while the covenant that granted
        it is still the covenant in force.
        """

        if job.escalation_id is None:
            return False
        try:
            escalation = self.authority.escalation(job.escalation_id)
        except ControlError:
            return False
        return (
            escalation.state is EscalationState.CONSUMED
            and escalation.covenant_sha256 == resolution.covenant_sha256
            and escalation.action_id == job.type.value
        )

    def get_job(self, job_id: str) -> Job:
        with self._sessions() as session:
            return self._require_job(session, job_id)

    def list_jobs(
        self,
        *,
        status: JobStatus | None = None,
        job_type: JobType | None = None,
        limit: int = 100,
    ) -> list[Job]:
        query: Select[tuple[Job]] = select(Job)
        if status is not None:
            query = query.where(Job.status == status)
        if job_type is not None:
            query = query.where(Job.type == job_type)
        query = query.order_by(Job.created_at.desc(), Job.id.desc()).limit(limit)
        with self._sessions() as session:
            return list(session.scalars(query))

    def list_events(self, job_id: str, *, after: int = 0, limit: int = 500) -> list[JobEvent]:
        with self._sessions() as session:
            self._require_job(session, job_id)
            query = (
                select(JobEvent)
                .where(JobEvent.job_id == job_id, JobEvent.sequence > after)
                .order_by(JobEvent.sequence)
                .limit(limit)
            )
            return list(session.scalars(query))

    def list_artifacts(self, job_id: str) -> list[JobArtifact]:
        with self._sessions() as session:
            self._require_job(session, job_id)
            query = (
                select(JobArtifact)
                .where(JobArtifact.job_id == job_id)
                .order_by(JobArtifact.created_at, JobArtifact.id)
            )
            return list(session.scalars(query))

    def claim_next(self, worker_id: str) -> Job | None:
        """Atomically claim the oldest queued job.

        A conditional UPDATE prevents two workers that observed the same
        candidate from both claiming it.  Losing workers retry the selection.
        """

        for _attempt in range(8):
            with self._write_session() as session:
                candidate = session.scalar(
                    select(Job.id)
                    .where(Job.status == JobStatus.QUEUED)
                    .order_by(Job.created_at, Job.id)
                    .limit(1)
                )
                if candidate is None:
                    return None
                now = utc_now()
                claimed = cast(
                    CursorResult[Any],
                    session.execute(
                        update(Job)
                        .where(Job.id == candidate, Job.status == JobStatus.QUEUED)
                        .values(
                            status=JobStatus.STARTING,
                            worker_id=worker_id,
                            started_at=now,
                            heartbeat_at=now,
                            updated_at=now,
                        )
                    ),
                )
                if claimed.rowcount != 1:
                    continue
                job = self._require_job(session, candidate)
                self._event(session, job, "job.starting", {"worker_id": worker_id})
                return job
        return None

    def mark_running(self, job_id: str, worker_id: str) -> Job:
        with self._write_session() as session:
            job = self._require_job(session, job_id)
            if job.worker_id != worker_id:
                raise InvalidTransitionError("job is claimed by another worker")
            now = utc_now()
            self._transition(job, JobStatus.RUNNING, now)
            self._event(session, job, "job.running", {"worker_id": worker_id})
            return job

    def heartbeat(self, job_id: str, worker_id: str) -> Job:
        with self._write_session() as session:
            job = self._require_job(session, job_id)
            if job.worker_id != worker_id:
                raise InvalidTransitionError("job is claimed by another worker")
            if job.status not in {
                JobStatus.STARTING,
                JobStatus.RUNNING,
                JobStatus.CANCELLING,
            }:
                raise InvalidTransitionError("only active jobs can emit a heartbeat")
            now = utc_now()
            job.heartbeat_at = now
            job.updated_at = now
            return job

    def emit_event(
        self,
        job_id: str,
        worker_id: str,
        kind: str,
        payload: dict[str, Any] | None = None,
    ) -> JobEvent:
        """Append a redacted worker event while preserving per-job ordering."""

        if not kind or len(kind) > 64:
            raise ValueError("event kind must contain between 1 and 64 characters")
        with self._write_session() as session:
            job = self._require_job(session, job_id)
            if job.worker_id != worker_id or job.status not in {
                JobStatus.STARTING,
                JobStatus.RUNNING,
                JobStatus.CANCELLING,
            }:
                raise InvalidTransitionError("job is not active on this worker")
            now = utc_now()
            job.heartbeat_at = now
            job.updated_at = now
            return self._event(session, job, kind, payload)

    def update_progress(
        self,
        job_id: str,
        worker_id: str,
        progress: float,
        payload: dict[str, Any] | None = None,
    ) -> Job:
        if not 0.0 <= progress <= 1.0:
            raise ValueError("progress must be between zero and one")
        with self._write_session() as session:
            job = self._require_job(session, job_id)
            if job.worker_id != worker_id or job.status not in {
                JobStatus.RUNNING,
                JobStatus.CANCELLING,
            }:
                raise InvalidTransitionError("job is not running on this worker")
            now = utc_now()
            job.progress = max(job.progress, progress)
            job.heartbeat_at = now
            job.updated_at = now
            event_payload = {"progress": job.progress, **(payload or {})}
            self._event(session, job, "job.progress", event_payload)
            return job

    def cancel(self, job_id: str) -> Job:
        with self._write_session() as session:
            job = self._require_job(session, job_id)
            if job.status in TERMINAL_JOB_STATUSES or job.status is JobStatus.CANCELLING:
                return job
            now = utc_now()
            job.cancel_requested_at = now
            if job.status is JobStatus.QUEUED:
                self._transition(job, JobStatus.CANCELLED, now)
                self._event(session, job, "job.cancelled", {"before_start": True})
            else:
                self._transition(job, JobStatus.CANCELLING, now)
                self._event(session, job, "job.cancelling")
            return job

    def cancellation_requested(self, job_id: str, worker_id: str) -> bool:
        with self._sessions() as session:
            job = self._require_job(session, job_id)
            return job.worker_id == worker_id and job.status is JobStatus.CANCELLING

    def complete_success(
        self, job_id: str, worker_id: str, result: dict[str, Any] | None = None
    ) -> Job:
        with self._write_session() as session:
            job = self._require_job(session, job_id)
            if job.worker_id != worker_id:
                raise InvalidTransitionError("job is claimed by another worker")
            now = utc_now()
            if job.status is JobStatus.CANCELLING:
                self._transition(job, JobStatus.CANCELLED, now)
                self._event(session, job, "job.cancelled", {"before_completion": True})
            else:
                self._transition(job, JobStatus.SUCCEEDED, now)
                job.result = redact(result or {})
                self._event(session, job, "job.succeeded")
            return job

    def complete_failure(
        self,
        job_id: str,
        worker_id: str,
        *,
        code: str,
        message: str,
    ) -> Job:
        with self._write_session() as session:
            job = self._require_job(session, job_id)
            if job.worker_id != worker_id:
                raise InvalidTransitionError("job is claimed by another worker")
            now = utc_now()
            self._transition(job, JobStatus.FAILED, now)
            job.error_code = re_safe_code(code)
            job.error_message = redact_text(message)[:4000]
            self._event(
                session,
                job,
                "job.failed",
                {"code": job.error_code, "message": job.error_message},
            )
            return job

    def reconcile_stale(self, *, now: datetime | None = None) -> list[str]:
        current = now or utc_now()
        cutoff = current - timedelta(seconds=self.settings.stale_after_seconds)
        interrupted: list[str] = []
        with self._write_session() as session:
            jobs = list(
                session.scalars(
                    select(Job).where(
                        Job.status.in_(
                            [
                                JobStatus.STARTING,
                                JobStatus.RUNNING,
                                JobStatus.CANCELLING,
                            ]
                        ),
                        Job.heartbeat_at.is_not(None),
                        Job.heartbeat_at < cutoff,
                    )
                )
            )
            for job in jobs:
                self._transition(job, JobStatus.INTERRUPTED, current)
                job.error_code = "stale_heartbeat"
                job.error_message = "worker heartbeat expired"
                self._event(session, job, "job.interrupted", {"reason": "stale_heartbeat"})
                interrupted.append(job.id)
        return interrupted

    def retry(self, job_id: str, *, actor: Actor = ANONYMOUS) -> Job:
        with self._sessions() as session:
            original = self._require_job(session, job_id)
            if original.status not in {
                JobStatus.FAILED,
                JobStatus.CANCELLED,
                JobStatus.INTERRUPTED,
            }:
                raise InvalidTransitionError(
                    "only failed, cancelled, or interrupted jobs can retry"
                )
            job_type = original.type
            spec = dict(original.spec)
        # A retry is a fresh submission by whoever asked for it, not a replay of
        # the original submitter's authority.
        return self.create_job(job_type, spec, actor=actor, retry_of_id=job_id)

    def restore_authority_index(self) -> int:
        """Rebuild the identity and escalation tables from hash-chained truth.

        Deterministic by construction: the same log always produces the same
        rows.  Deleting the SQLite file loses nothing, and a broken chain raises
        out of here rather than leaving a half-built index in place.
        """

        records = self.authority.records()
        principals = self.authority.principals()
        escalations = self.authority.escalations(limit=len(records) + 1)
        with self._write_session() as session:
            session.execute(delete(AuthorityEvent))
            session.execute(delete(PrincipalRow))
            session.execute(delete(EscalationRow))
            for principal in principals:
                session.add(
                    PrincipalRow(
                        id=principal.id,
                        role=principal.role.value,
                        label=principal.label,
                        scopes=sorted(principal.scopes),
                        credential_sha256=principal.credential_sha256,
                        created_at=principal.created_at,
                        expires_at=principal.expires_at,
                        revoked_at=principal.revoked_at,
                        rotated_at=principal.rotated_at,
                    )
                )
            for escalation in escalations:
                session.add(
                    EscalationRow(
                        id=escalation.id,
                        actor_id=escalation.actor_id,
                        actor_role=escalation.actor_role.value,
                        action_id=escalation.action_id,
                        request_digest=escalation.request_digest,
                        covenant_sha256=escalation.covenant_sha256,
                        binding_priority=escalation.binding_priority,
                        reasons=list(escalation.reasons),
                        state=escalation.effective_state().value,
                        decided_by=escalation.decided_by,
                        rationale=escalation.rationale,
                        created_at=escalation.created_at,
                        decided_at=escalation.decided_at,
                        approval_expires_at=escalation.approval_expires_at,
                        consumed_at=escalation.consumed_at,
                        consumed_job_id=escalation.consumed_job_id,
                    )
                )
            for record in records:
                created_at = parse_utc(record.get("recorded_at"))
                if created_at is None:
                    raise TruthStoreError("authority record requires recorded_at")
                body = record.get("principal") or record.get("escalation") or {}
                subject = body.get("id") if isinstance(body, dict) else None
                session.add(
                    AuthorityEvent(
                        sequence=int(record["sequence"]),
                        kind=str(record["kind"])[:48],
                        actor_id=record.get("by") or record.get("actor_id"),
                        subject_id=(
                            subject or record.get("principal_id") or record.get("escalation_id")
                        ),
                        payload=redact(_indexable(record)),
                        record_sha256=str(record["record_sha256"]),
                        created_at=created_at,
                    )
                )
        return len(records)

    def register_artifact(
        self,
        job_id: str,
        path: str | Path,
        *,
        kind: str = "file",
        media_type: str | None = None,
    ) -> JobArtifact:
        assert self.settings.artifact_root is not None
        try:
            resolved = contained_path(self.settings.artifact_root, path, must_exist=True)
        except (PathOutsideRootError, FileNotFoundError) as exc:
            raise ArtifactPathError(
                "artifact must be an existing file beneath artifact_root"
            ) from exc
        if not resolved.is_file():
            raise ArtifactPathError("artifact path must identify a regular file")
        digest = hashlib.sha256()
        with resolved.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        relative_path = resolved.relative_to(self.settings.artifact_root).as_posix()
        artifact = JobArtifact(
            id=str(uuid4()),
            job_id=job_id,
            kind=kind[:64],
            path=relative_path,
            sha256=digest.hexdigest(),
            size_bytes=resolved.stat().st_size,
            media_type=media_type or mimetypes.guess_type(resolved.name)[0],
            created_at=utc_now(),
        )
        with self._write_session() as session:
            job = self._require_job(session, job_id)
            session.add(artifact)
            self._event(
                session,
                job,
                "artifact.created",
                {
                    "artifact_id": artifact.id,
                    "kind": artifact.kind,
                    "path": artifact.path,
                    "sha256": artifact.sha256,
                    "size_bytes": artifact.size_bytes,
                    "media_type": artifact.media_type,
                    "created_at": artifact.created_at.isoformat().replace("+00:00", "Z"),
                },
            )
        return artifact


def _indexable(record: dict[str, Any]) -> dict[str, Any]:
    """Drop chain framing from a record before indexing its body."""

    framing = {"schema_version", "sequence", "previous_sha256", "record_sha256", "kind"}
    return {key: value for key, value in record.items() if key not in framing}


def re_safe_code(value: str) -> str:
    """Normalize worker exception classes into a stable, non-sensitive code."""

    normalized = "".join(ch.lower() if ch.isalnum() else "_" for ch in value)
    normalized = "_".join(part for part in normalized.split("_") if part)
    return (normalized or "job_failed")[:96]
