"""Allow the v0.0.3 provenance data-engine job types.

Revision ID: 0003_data_jobs
Revises: 0002_safety_identity
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003_data_jobs"
down_revision: str | None = "0002_safety_identity"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_V002_TYPES = "'system_probe', 'research_sync', 'tiny_train', 'tiny_eval'"
_V003_TYPES = (
    "'system_probe', 'research_sync', 'data_ingest', 'data_build', "
    "'data_remove', 'data_audit', 'tiny_train', 'tiny_eval'"
)


def _replace_job_type_constraint(values: str) -> None:
    with op.batch_alter_table("jobs") as batch:
        batch.drop_constraint("job_type", type_="check")
        batch.create_check_constraint("job_type", f"type IN ({values})")


def upgrade() -> None:
    _replace_job_type_constraint(_V003_TYPES)


def downgrade() -> None:
    bind = op.get_bind()
    newer_jobs = bind.execute(
        sa.text(
            "SELECT COUNT(*) FROM jobs WHERE type IN "
            "('data_ingest', 'data_build', 'data_remove', 'data_audit')"
        )
    ).scalar_one()
    if newer_jobs:
        raise RuntimeError("cannot downgrade while v0.0.3 data jobs exist")
    _replace_job_type_constraint(_V002_TYPES)
