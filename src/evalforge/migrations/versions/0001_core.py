"""Initial V1 evidence schema. Definitions are frozen; do not import live metadata."""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "0001_core"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    payload = sa.JSON().with_variant(JSONB(), "postgresql")
    op.create_table(
        "evalforge_datasets",
        sa.Column("name", sa.String(), primary_key=True),
        sa.Column("version", sa.String(), primary_key=True),
        sa.Column("content_hash", sa.String(), nullable=False),
        sa.Column("payload", payload, nullable=False),
    )
    op.create_table(
        "evalforge_runs",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("dataset_name", sa.String(), nullable=False),
        sa.Column("dataset_version", sa.String(), nullable=False),
        sa.Column("payload", payload, nullable=False),
        sa.ForeignKeyConstraint(
            ["dataset_name", "dataset_version"],
            ["evalforge_datasets.name", "evalforge_datasets.version"],
        ),
    )
    op.create_table(
        "evalforge_experiments",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("baseline_id", sa.String(), sa.ForeignKey("evalforge_runs.id"), nullable=False),
        sa.Column("candidate_id", sa.String(), sa.ForeignKey("evalforge_runs.id"), nullable=False),
        sa.Column("payload", payload, nullable=False),
    )


def downgrade():
    raise RuntimeError("destructive downgrades are intentionally unsupported; restore a backup")
