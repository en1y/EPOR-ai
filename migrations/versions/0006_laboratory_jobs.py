"""Allow the v0.0.6 evaluation and scaling-laboratory job types.

Revision ID: 0006_laboratory_jobs
Revises: 0005_kernel_jobs
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0006_laboratory_jobs"
down_revision: str | None = "0005_kernel_jobs"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_V005_TYPES = (
    "'system_probe', 'research_sync', 'data_ingest', 'data_build', "
    "'data_remove', 'data_audit', 'tokenizer_train', 'tokenizer_shard', "
    "'tokenizer_evaluate', 'kernel_train', 'kernel_evaluate', 'tiny_train', "
    "'tiny_eval'"
)
_V006_TYPES = (
    "'system_probe', 'research_sync', 'data_ingest', 'data_build', "
    "'data_remove', 'data_audit', 'tokenizer_train', 'tokenizer_shard', "
    "'tokenizer_evaluate', 'kernel_train', 'kernel_evaluate', 'eval_suite', "
    "'eval_contamination', 'scale_profile', 'scale_sweep', 'scale_mixture', "
    "'tiny_train', 'tiny_eval'"
)
_V006_ONLY = "'eval_suite', 'eval_contamination', 'scale_profile', 'scale_sweep', 'scale_mixture'"

# The non-native enum column is a VARCHAR sized to the longest member. The
# longest new name, ``eval_contamination``, is 18 characters, the same as
# ``tokenizer_evaluate``, so the column keeps its width and only the CHECK
# constraint changes; the no-drift migration test would catch a width change.
# Code execution is deliberately not a job type: it stays on the owner's CLI.


def _replace_job_type_constraint(values: str) -> None:
    with op.batch_alter_table("jobs") as batch:
        batch.drop_constraint("job_type", type_="check")
        batch.create_check_constraint("job_type", f"type IN ({values})")


def upgrade() -> None:
    _replace_job_type_constraint(_V006_TYPES)


def downgrade() -> None:
    bind = op.get_bind()
    newer_jobs = bind.execute(
        sa.text(f"SELECT COUNT(*) FROM jobs WHERE type IN ({_V006_ONLY})")
    ).scalar_one()
    if newer_jobs:
        raise RuntimeError("cannot downgrade while v0.0.6 laboratory jobs exist")
    _replace_job_type_constraint(_V005_TYPES)
