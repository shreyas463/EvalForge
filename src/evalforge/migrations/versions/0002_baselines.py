"""Append-only approval audit records and named baseline selection."""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "0002_baselines"
down_revision = "0001_core"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "evalforge_baseline_approvals",
        sa.Column("sequence", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("id", sa.String(), nullable=False, unique=True),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("run_id", sa.String(), sa.ForeignKey("evalforge_runs.id"), nullable=False),
        sa.Column("payload", sa.JSON().with_variant(JSONB(), "postgresql"), nullable=False),
    )
    op.create_index(
        "ix_evalforge_baseline_name", "evalforge_baseline_approvals", ["name", "sequence"]
    )


def downgrade():
    raise RuntimeError("destructive downgrades are intentionally unsupported; restore a backup")
