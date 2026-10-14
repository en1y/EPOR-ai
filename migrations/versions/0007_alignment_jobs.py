"""Allow the v0.0.7 knowledge-transfer and alignment job types.

Revision ID: 0007_alignment_jobs
Revises: 0006_laboratory_jobs
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0007_alignment_jobs"
down_revision: str | None = "0006_laboratory_jobs"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_V006_TYPES = (
    "'system_probe', 'research_sync', 'data_ingest', 'data_build', "
    "'data_remove', 'data_audit', 'tokenizer_train', 'tokenizer_shard', "
    "'tokenizer_evaluate', 'kernel_train', 'kernel_evaluate', 'eval_suite', "
    "'eval_contamination', 'scale_profile', 'scale_sweep', 'scale_mixture', "
    "'tiny_train', 'tiny_eval'"
)
_V007_ONLY = (
    "'align_generate', 'align_judge', 'align_dataset', 'align_sft', "
    "'align_preference', 'align_evaluate'"
)
_V007_TYPES = (
    "'system_probe', 'research_sync', 'data_ingest', 'data_build', "
    "'data_remove', 'data_audit', 'tokenizer_train', 'tokenizer_shard', "
    "'tokenizer_evaluate', 'kernel_train', 'kernel_evaluate', 'eval_suite', "
    "'eval_contamination', 'scale_profile', 'scale_sweep', 'scale_mixture', "
    f"{_V007_ONLY}, 'tiny_train', 'tiny_eval'"
)

# The non-native enum column is a VARCHAR sized to the longest member. The
# longest new name, ``align_preference``, is 16 characters, shorter than the
# existing 18-character ``tokenizer_evaluate``, so the column keeps its width
# and only the CHECK constraint changes; the no-drift migration test would
# catch a width change. Registering teacher material and recording human audit
# verdicts are deliberately not job types: the first stays on the owner's CLI,
# and a verdict is only ever recorded by a verified human.


def _replace_job_type_constraint(values: str) -> None:
    with op.batch_alter_table("jobs") as batch:
        batch.drop_constraint("job_type", type_="check")
        batch.create_check_constraint("job_type", f"type IN ({values})")


def upgrade() -> None:
    _replace_job_type_constraint(_V007_TYPES)


def downgrade() -> None:
    bind = op.get_bind()
    newer_jobs = bind.execute(
        sa.text(f"SELECT COUNT(*) FROM jobs WHERE type IN ({_V007_ONLY})")
    ).scalar_one()
    if newer_jobs:
        raise RuntimeError("cannot downgrade while v0.0.7 alignment jobs exist")
    _replace_job_type_constraint(_V006_TYPES)
