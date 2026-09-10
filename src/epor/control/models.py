"""SQLAlchemy 2 typed persistence models for the rebuildable local index."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum, StrEnum
from typing import Any

from sqlalchemy import (
    JSON,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy import (
    Enum as SqlEnum,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.types import TypeDecorator


def utc_now() -> datetime:
    return datetime.now(UTC)


class UTCDateTime(TypeDecorator[datetime]):
    """Persist UTC in SQLite and always restore timezone-aware values.

    SQLite has no timezone-aware datetime storage even when ``timezone=True``
    is requested.  Values are stored as naive UTC for sortable compatibility,
    then normalized back to aware UTC when materialized by the ORM.
    """

    impl = DateTime
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: object) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError("control-plane timestamps must be timezone-aware")
        return value.astimezone(UTC).replace(tzinfo=None)

    def process_result_value(self, value: datetime | None, dialect: object) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)


class JobType(StrEnum):
    SYSTEM_PROBE = "system_probe"
    RESEARCH_SYNC = "research_sync"
    DATA_INGEST = "data_ingest"
    DATA_BUILD = "data_build"
    DATA_REMOVE = "data_remove"
    DATA_AUDIT = "data_audit"
    TINY_TRAIN = "tiny_train"
    TINY_EVAL = "tiny_eval"


class JobStatus(StrEnum):
    QUEUED = "queued"
    STARTING = "starting"
    RUNNING = "running"
    CANCELLING = "cancelling"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"


TERMINAL_JOB_STATUSES = frozenset(
    {
        JobStatus.SUCCEEDED,
        JobStatus.FAILED,
        JobStatus.CANCELLED,
        JobStatus.INTERRUPTED,
    }
)


def _enum(enum_type: type[Enum], name: str) -> SqlEnum:
    return SqlEnum(
        enum_type,
        name=name,
        native_enum=False,
        create_constraint=False,
        validate_strings=True,
        values_callable=lambda members: [member.value for member in members],
    )


class Base(DeclarativeBase):
    pass


class Job(Base):
    __tablename__ = "jobs"
    __table_args__ = (
        CheckConstraint(
            "type IN ('system_probe', 'research_sync', 'data_ingest', 'data_build', "
            "'data_remove', 'data_audit', 'tiny_train', 'tiny_eval')",
            name="job_type",
        ),
        CheckConstraint(
            "status IN ('queued', 'starting', 'running', 'cancelling', "
            "'succeeded', 'failed', 'cancelled', 'interrupted')",
            name="job_status",
        ),
        Index("ix_jobs_status_created_at", "status", "created_at"),
        Index("ix_jobs_type_created_at", "type", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    type: Mapped[JobType] = mapped_column(_enum(JobType, "job_type"), nullable=False)
    status: Mapped[JobStatus] = mapped_column(
        _enum(JobStatus, "job_status"), nullable=False, default=JobStatus.QUEUED
    )
    spec: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    result: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    progress: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    error_code: Mapped[str | None] = mapped_column(String(96), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    worker_id: Mapped[str | None] = mapped_column(String(160), nullable=True)
    event_cursor: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    retry_of_id: Mapped[str | None] = mapped_column(
        ForeignKey("jobs.id", ondelete="SET NULL"), nullable=True
    )
    # Admission provenance. Nullable because v0.0.1 jobs predate identity and
    # must stay readable exactly as they were recorded.
    submitted_by_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    submitted_by_role: Mapped[str | None] = mapped_column(String(32), nullable=True)
    admission_covenant_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    escalation_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False, default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        UTCDateTime(), nullable=False, default=utc_now, onupdate=utc_now
    )
    started_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    finished_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    heartbeat_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    cancel_requested_at: Mapped[datetime | None] = mapped_column(UTCDateTime())

    retry_of: Mapped[Job | None] = relationship(
        "Job", remote_side="Job.id", foreign_keys=[retry_of_id]
    )
    events: Mapped[list[JobEvent]] = relationship(
        "JobEvent", back_populates="job", cascade="all, delete-orphan"
    )
    artifacts: Mapped[list[JobArtifact]] = relationship(
        "JobArtifact", back_populates="job", cascade="all, delete-orphan"
    )


class JobEvent(Base):
    __tablename__ = "job_events"
    __table_args__ = (
        UniqueConstraint("job_id", "sequence", name="uq_job_events_job_sequence"),
        Index("ix_job_events_job_id_sequence", "job_id", "sequence"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False)
    sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    kind: Mapped[str] = mapped_column(String(64), nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False, default=utc_now)

    job: Mapped[Job] = relationship("Job", back_populates="events")


class Principal(Base):
    """Rebuildable projection of one identity from hash-chained safety truth.

    Nothing authorizes anything from this table.  It exists so the console can
    list delegations without replaying the log per request; the log decides.
    """

    __tablename__ = "principals"
    __table_args__ = (Index("ix_principals_role_created_at", "role", "created_at"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    role: Mapped[str] = mapped_column(String(32), nullable=False)
    label: Mapped[str] = mapped_column(String(120), nullable=False)
    scopes: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    # The digest only. A credential value never reaches SQLite, a serializer,
    # a log line, or an error envelope.
    credential_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    expires_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    revoked_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    rotated_at: Mapped[datetime | None] = mapped_column(UTCDateTime())


class Escalation(Base):
    """Rebuildable projection of one human safety decision."""

    __tablename__ = "escalations"
    __table_args__ = (Index("ix_escalations_state_created_at", "state", "created_at"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    actor_id: Mapped[str] = mapped_column(String(36), nullable=False)
    actor_role: Mapped[str] = mapped_column(String(32), nullable=False)
    action_id: Mapped[str] = mapped_column(String(64), nullable=False)
    request_digest: Mapped[str] = mapped_column(String(64), nullable=False)
    covenant_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    binding_priority: Mapped[int | None] = mapped_column(Integer)
    reasons: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    state: Mapped[str] = mapped_column(String(16), nullable=False)
    decided_by: Mapped[str | None] = mapped_column(String(36))
    rationale: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    decided_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    approval_expires_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    consumed_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    consumed_job_id: Mapped[str | None] = mapped_column(String(36))


class AuthorityEvent(Base):
    """One indexed record of the authority chain: credential, decision, review.

    ``sequence`` mirrors the chain position, so a rebuilt index that disagrees
    with the file log is detectable rather than quietly authoritative.
    """

    __tablename__ = "authority_events"
    __table_args__ = (
        UniqueConstraint("sequence", name="uq_authority_events_sequence"),
        Index("ix_authority_events_kind_sequence", "kind", "sequence"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    kind: Mapped[str] = mapped_column(String(48), nullable=False)
    actor_id: Mapped[str | None] = mapped_column(String(36))
    subject_id: Mapped[str | None] = mapped_column(String(36))
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    record_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)


class JobArtifact(Base):
    __tablename__ = "job_artifacts"
    __table_args__ = (
        UniqueConstraint("job_id", "path", name="uq_job_artifacts_job_path"),
        Index("ix_job_artifacts_job_id_created_at", "job_id", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False)
    kind: Mapped[str] = mapped_column(String(64), nullable=False)
    path: Mapped[str] = mapped_column(String(1024), nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    media_type: Mapped[str | None] = mapped_column(String(160))
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False, default=utc_now)

    job: Mapped[Job] = relationship("Job", back_populates="artifacts")
