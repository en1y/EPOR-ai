"""Allow the v0.0.8 context-laboratory job types.

Revision ID: 0008_context_jobs
Revises: 0007_alignment_jobs
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0008_context_jobs"
down_revision: str | None = "0007_alignment_jobs"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_V007_BASE = (
    "'system_probe', 'research_sync', 'data_ingest', 'data_build', "
    "'data_remove', 'data_audit', 'tokenizer_train', 'tokenizer_shard', "
    "'tokenizer_evaluate', 'kernel_train', 'kernel_evaluate', 'eval_suite', "
    "'eval_contamination', 'scale_profile', 'scale_sweep', 'scale_mixture', "
    "'align_generate', 'align_judge', 'align_dataset', 'align_sft', "
    "'align_preference', 'align_evaluate'"
)
_V008_ONLY = "'context_extend', 'context_evaluate', 'context_profile', 'context_sweep'"
_V007_TYPES = f"{_V007_BASE}, 'tiny_train', 'tiny_eval'"
_V008_TYPES = f"{_V007_BASE}, {_V008_ONLY}, 'tiny_train', 'tiny_eval'"

# The non-native enum column is a VARCHAR sized to the longest member. The
# longest new name, ``context_evaluate``, is 16 characters, under the existing
# 18-character ``tokenizer_evaluate``, so the column keeps its width and only
# the CHECK constraint changes; the no-drift migration test would catch a width
# change. Projections and validated-context certification are deliberately
# not job types: both stay on the owner's CLI.


def _replace_job_type_constraint(values: str) -> None:
    with op.batch_alter_table("jobs") as batch:
        batch.drop_constraint("job_type", type_="check")
        batch.create_check_constraint("job_type", f"type IN ({values})")


def upgrade() -> None:
    _replace_job_type_constraint(_V008_TYPES)


def downgrade() -> None:
    bind = op.get_bind()
    newer_jobs = bind.execute(
        sa.text(f"SELECT COUNT(*) FROM jobs WHERE type IN ({_V008_ONLY})")
    ).scalar_one()
    if newer_jobs:
        raise RuntimeError("cannot downgrade while v0.0.8 context jobs exist")
    _replace_job_type_constraint(_V007_TYPES)
