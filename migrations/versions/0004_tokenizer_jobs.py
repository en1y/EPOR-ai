"""Allow the v0.0.4 tokenizer training, sharding, and fertility job types.

Revision ID: 0004_tokenizer_jobs
Revises: 0003_data_jobs
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0004_tokenizer_jobs"
down_revision: str | None = "0003_data_jobs"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_V003_TYPES = (
    "'system_probe', 'research_sync', 'data_ingest', 'data_build', "
    "'data_remove', 'data_audit', 'tiny_train', 'tiny_eval'"
)
_V004_TYPES = (
    "'system_probe', 'research_sync', 'data_ingest', 'data_build', "
    "'data_remove', 'data_audit', 'tokenizer_train', 'tokenizer_shard', "
    "'tokenizer_evaluate', 'tiny_train', 'tiny_eval'"
)


# The non-native enum column is a VARCHAR sized to the longest member, and
# ``tokenizer_evaluate`` is the first value longer than ``research_sync``.
_V003_WIDTH = 13
_V004_WIDTH = 18


def _replace_job_type_constraint(values: str, *, width: int, previous_width: int) -> None:
    with op.batch_alter_table("jobs") as batch:
        batch.drop_constraint("job_type", type_="check")
        batch.alter_column(
            "type",
            existing_type=sa.String(length=previous_width),
            type_=sa.String(length=width),
            existing_nullable=False,
        )
        batch.create_check_constraint("job_type", f"type IN ({values})")


def upgrade() -> None:
    _replace_job_type_constraint(_V004_TYPES, width=_V004_WIDTH, previous_width=_V003_WIDTH)


def downgrade() -> None:
    bind = op.get_bind()
    newer_jobs = bind.execute(
        sa.text(
            "SELECT COUNT(*) FROM jobs WHERE type IN "
            "('tokenizer_train', 'tokenizer_shard', 'tokenizer_evaluate')"
        )
    ).scalar_one()
    if newer_jobs:
        raise RuntimeError("cannot downgrade while v0.0.4 tokenizer jobs exist")
    _replace_job_type_constraint(_V003_TYPES, width=_V003_WIDTH, previous_width=_V004_WIDTH)
