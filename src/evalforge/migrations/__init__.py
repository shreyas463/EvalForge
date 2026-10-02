"""Packaged Alembic migrations, including a checked adoption path for V1 databases."""

from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import inspect, text

HEAD = "0002_baselines"
LEGACY = "0001_core"
CORE_TABLES = {
    "evalforge_datasets": {"name", "version", "content_hash", "payload"},
    "evalforge_runs": {"id", "dataset_name", "dataset_version", "payload"},
    "evalforge_experiments": {"id", "baseline_id", "candidate_id", "payload"},
}


def migration_config(connection):
    config = Config()
    config.set_main_option("script_location", str(Path(__file__).parent))
    config.attributes["connection"] = connection
    return config


def current_revision(engine):
    with engine.connect() as connection:
        if "alembic_version" not in inspect(connection).get_table_names():
            return None
        revisions = (
            connection.execute(text("SELECT version_num FROM alembic_version")).scalars().all()
        )
        return revisions[0] if len(revisions) == 1 else None


def _validate_legacy(connection):
    inspector = inspect(connection)
    tables = set(inspector.get_table_names())
    present = set(CORE_TABLES) & tables
    if not present:
        return False
    if present != set(CORE_TABLES):
        raise ValueError("partial legacy EvalForge schema; restore or migrate it manually")
    primary_keys = {
        "evalforge_datasets": {"name", "version"},
        "evalforge_runs": {"id"},
        "evalforge_experiments": {"id"},
    }
    for name, expected in CORE_TABLES.items():
        columns = inspector.get_columns(name)
        if {c["name"] for c in columns} != expected:
            raise ValueError(f"legacy schema mismatch: {name}")
        if set(inspector.get_pk_constraint(name)["constrained_columns"]) != primary_keys[name]:
            raise ValueError(f"legacy primary key mismatch: {name}")
        for column in columns:
            from sqlalchemy import JSON, String

            expected_type = JSON if column["name"] == "payload" else String
            if not isinstance(column["type"], expected_type) or column["nullable"]:
                raise ValueError(
                    f"legacy column type/nullability mismatch: {name}.{column['name']}"
                )
    expected_foreign = {
        "evalforge_runs": {
            (("dataset_name", "dataset_version"), "evalforge_datasets", ("name", "version"))
        },
        "evalforge_experiments": {
            (("baseline_id",), "evalforge_runs", ("id",)),
            (("candidate_id",), "evalforge_runs", ("id",)),
        },
    }
    for name, expected in expected_foreign.items():
        actual = {
            (tuple(f["constrained_columns"]), f["referred_table"], tuple(f["referred_columns"]))
            for f in inspector.get_foreign_keys(name)
        }
        if actual != expected:
            raise ValueError(f"legacy foreign key mismatch: {name}")
    return True


def upgrade_database(engine):
    with engine.begin() as connection:
        # Serialize PostgreSQL initialization/adoption/upgrades across processes.
        if connection.dialect.name == "postgresql":
            connection.execute(text("SELECT pg_advisory_xact_lock(731426018)"))
        config = migration_config(connection)
        if "alembic_version" not in inspect(connection).get_table_names() and _validate_legacy(
            connection
        ):
            command.stamp(config, LEGACY)
        command.upgrade(config, "head")
