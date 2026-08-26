"""Add the v0.0.2 identity, escalation, and admission-provenance index.

Revision ID: 0002_safety_identity
Revises: 0001_control_plane

These tables are a rebuildable query index over hash-chained safety truth.
Dropping them loses no authority; they are reconstructed from the log.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002_safety_identity"
down_revision: str | None = "0001_control_plane"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "principals",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("role", sa.String(length=32), nullable=False),
        sa.Column("label", sa.String(length=120), nullable=False),
        sa.Column("scopes", sa.JSON(), nullable=False),
        sa.Column("credential_sha256", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("rotated_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_principals_role_created_at", "principals", ["role", "created_at"])

    op.create_table(
        "escalations",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("actor_id", sa.String(length=36), nullable=False),
        sa.Column("actor_role", sa.String(length=32), nullable=False),
        sa.Column("action_id", sa.String(length=64), nullable=False),
        sa.Column("request_digest", sa.String(length=64), nullable=False),
        sa.Column("covenant_sha256", sa.String(length=64), nullable=False),
        sa.Column("binding_priority", sa.Integer(), nullable=True),
        sa.Column("reasons", sa.JSON(), nullable=False),
        sa.Column("state", sa.String(length=16), nullable=False),
        sa.Column("decided_by", sa.String(length=36), nullable=True),
        sa.Column("rationale", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("decided_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("approval_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("consumed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("consumed_job_id", sa.String(length=36), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_escalations_state_created_at", "escalations", ["state", "created_at"])

    op.create_table(
        "authority_events",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.Column("kind", sa.String(length=48), nullable=False),
        sa.Column("actor_id", sa.String(length=36), nullable=True),
        sa.Column("subject_id", sa.String(length=36), nullable=True),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("record_sha256", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("sequence", name="uq_authority_events_sequence"),
    )
    op.create_index(
        "ix_authority_events_kind_sequence",
        "authority_events",
        ["kind", "sequence"],
    )

    # Nullable throughout: a v0.0.1 job was admitted before identity existed and
    # must stay readable rather than be back-filled with an invented submitter.
    with op.batch_alter_table("jobs") as batch:
        batch.add_column(sa.Column("submitted_by_id", sa.String(length=36), nullable=True))
        batch.add_column(sa.Column("submitted_by_role", sa.String(length=32), nullable=True))
        batch.add_column(
            sa.Column("admission_covenant_sha256", sa.String(length=64), nullable=True)
        )
        batch.add_column(sa.Column("escalation_id", sa.String(length=36), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("jobs") as batch:
        batch.drop_column("escalation_id")
        batch.drop_column("admission_covenant_sha256")
        batch.drop_column("submitted_by_role")
        batch.drop_column("submitted_by_id")
    op.drop_index("ix_authority_events_kind_sequence", table_name="authority_events")
    op.drop_table("authority_events")
    op.drop_index("ix_escalations_state_created_at", table_name="escalations")
    op.drop_table("escalations")
    op.drop_index("ix_principals_role_created_at", table_name="principals")
    op.drop_table("principals")
