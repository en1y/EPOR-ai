"""Allow the v0.0.5 training-kernel train and evaluate job types.

Revision ID: 0005_kernel_jobs
Revises: 0004_tokenizer_jobs
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0005_kernel_jobs"
down_revision: str | None = "0004_tokenizer_jobs"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_V004_TYPES = (
    "'system_probe', 'research_sync', 'data_ingest', 'data_build', "
    "'data_remove', 'data_audit', 'tokenizer_train', 'tokenizer_shard', "
    "'tokenizer_evaluate', 'tiny_train', 'tiny_eval'"
)
_V005_TYPES = (
    "'system_probe', 'research_sync', 'data_ingest', 'data_build', "
    "'data_remove', 'data_audit', 'tokenizer_train', 'tokenizer_shard', "
    "'tokenizer_evaluate', 'kernel_train', 'kernel_evaluate', 'tiny_train', "
    "'tiny_eval'"
)

# The non-native enum column is a VARCHAR sized to the longest member.
# ``kernel_evaluate`` (15 characters) is shorter than ``tokenizer_evaluate``
# (18), so the column keeps its v0.0.4 width and only the CHECK constraint
# changes; the no-drift migration test would catch a width change.


def _replace_job_type_constraint(values: str) -> None:
    with op.batch_alter_table("jobs") as batch:
        batch.drop_constraint("job_type", type_="check")
        batch.create_check_constraint("job_type", f"type IN ({values})")


def upgrade() -> None:
    _replace_job_type_constraint(_V005_TYPES)


def downgrade() -> None:
    bind = op.get_bind()
    newer_jobs = bind.execute(
        sa.text("SELECT COUNT(*) FROM jobs WHERE type IN ('kernel_train', 'kernel_evaluate')")
    ).scalar_one()
    if newer_jobs:
        raise RuntimeError("cannot downgrade while v0.0.5 training-kernel jobs exist")
    _replace_job_type_constraint(_V004_TYPES)
