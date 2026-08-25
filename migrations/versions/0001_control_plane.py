"""Create the v0.0.1 local control-plane index.

Revision ID: 0001_control_plane
Revises: None
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001_control_plane"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


job_type = sa.Enum(
    "system_probe",
    "research_sync",
    "tiny_train",
    "tiny_eval",
    name="job_type",
    native_enum=False,
    create_constraint=False,
)
job_status = sa.Enum(
    "queued",
    "starting",
    "running",
    "cancelling",
    "succeeded",
    "failed",
    "cancelled",
    "interrupted",
    name="job_status",
    native_enum=False,
    create_constraint=False,
)


def upgrade() -> None:
    op.create_table(
        "jobs",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("type", job_type, nullable=False),
        sa.Column("status", job_status, nullable=False),
        sa.Column("spec", sa.JSON(), nullable=False),
        sa.Column("result", sa.JSON(), nullable=True),
        sa.Column("progress", sa.Float(), nullable=False),
        sa.Column("error_code", sa.String(length=96), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("worker_id", sa.String(length=160), nullable=True),
        sa.Column("event_cursor", sa.Integer(), nullable=False),
        sa.Column("retry_of_id", sa.String(length=36), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("heartbeat_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("cancel_requested_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "type IN ('system_probe', 'research_sync', 'tiny_train', 'tiny_eval')",
            name="job_type",
        ),
        sa.CheckConstraint(
            "status IN ('queued', 'starting', 'running', 'cancelling', "
            "'succeeded', 'failed', 'cancelled', 'interrupted')",
            name="job_status",
        ),
        sa.ForeignKeyConstraint(["retry_of_id"], ["jobs.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_jobs_status_created_at", "jobs", ["status", "created_at"])
    op.create_index("ix_jobs_type_created_at", "jobs", ["type", "created_at"])

    op.create_table(
        "job_events",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("job_id", sa.String(length=36), nullable=False),
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.Column("kind", sa.String(length=64), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["job_id"], ["jobs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("job_id", "sequence", name="uq_job_events_job_sequence"),
    )
    op.create_index("ix_job_events_job_id_sequence", "job_events", ["job_id", "sequence"])

    op.create_table(
        "job_artifacts",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("job_id", sa.String(length=36), nullable=False),
        sa.Column("kind", sa.String(length=64), nullable=False),
        sa.Column("path", sa.String(length=1024), nullable=False),
        sa.Column("sha256", sa.String(length=64), nullable=False),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column("media_type", sa.String(length=160), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["job_id"], ["jobs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("job_id", "path", name="uq_job_artifacts_job_path"),
    )
    op.create_index(
        "ix_job_artifacts_job_id_created_at",
        "job_artifacts",
        ["job_id", "created_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_job_artifacts_job_id_created_at", table_name="job_artifacts")
    op.drop_table("job_artifacts")
    op.drop_index("ix_job_events_job_id_sequence", table_name="job_events")
    op.drop_table("job_events")
    op.drop_index("ix_jobs_type_created_at", table_name="jobs")
    op.drop_index("ix_jobs_status_created_at", table_name="jobs")
    op.drop_table("jobs")
