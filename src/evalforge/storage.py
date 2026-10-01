"""Atomic JSON artifacts and transactional immutable SQL snapshots.

SQL tables intentionally store complete domain payloads in JSON/JSONB alongside indexed IDs.
This V1 schema is initialized with create_all; schema migrations are a future requirement.
"""

import json
import os
import tempfile
from pathlib import Path

from sqlalchemy import (
    JSON,
    Column,
    ForeignKey,
    ForeignKeyConstraint,
    MetaData,
    String,
    Table,
    create_engine,
    insert,
    select,
)
from sqlalchemy.dialects.postgresql import JSONB

from evalforge.datasets import dataset_hash, parse_json
from evalforge.models import Experiment, Run


class StorageError(RuntimeError):
    pass


def write_json(path: str | Path, payload: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as file:
            temporary = file.name
            json.dump(payload, file, ensure_ascii=False, indent=2, allow_nan=False)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def read_run(path: str | Path) -> Run:
    return Run.model_validate(parse_json(Path(path).read_text(encoding="utf-8")))


metadata = MetaData()
json_type = JSON().with_variant(JSONB(), "postgresql")
datasets = Table(
    "evalforge_datasets",
    metadata,
    Column("name", String, primary_key=True),
    Column("version", String, primary_key=True),
    Column("content_hash", String, nullable=False),
    Column("payload", json_type, nullable=False),
)
runs = Table(
    "evalforge_runs",
    metadata,
    Column("id", String, primary_key=True),
    Column("dataset_name", String, nullable=False),
    Column("dataset_version", String, nullable=False),
    Column("payload", json_type, nullable=False),
    ForeignKeyConstraint(
        ["dataset_name", "dataset_version"],
        ["evalforge_datasets.name", "evalforge_datasets.version"],
    ),
)
experiments = Table(
    "evalforge_experiments",
    metadata,
    Column("id", String, primary_key=True),
    Column("baseline_id", String, ForeignKey("evalforge_runs.id"), nullable=False),
    Column("candidate_id", String, ForeignKey("evalforge_runs.id"), nullable=False),
    Column("payload", json_type, nullable=False),
)


class SQLStore:
    def __init__(self, url: str):
        self.engine = None
        try:
            self.engine = create_engine(url, hide_parameters=True)
            metadata.create_all(self.engine)
        except Exception:
            if self.engine is not None:
                self.engine.dispose()
            raise StorageError("database initialization failed") from None

    def close(self):
        self.engine.dispose()

    @staticmethod
    def _immutable(connection, table, condition, values):
        existing = connection.execute(select(table).where(condition)).mappings().first()
        if existing:
            if any(existing[key] != value for key, value in values.items()):
                raise StorageError(f"immutable {table.name} snapshot conflicts with stored content")
        else:
            connection.execute(insert(table).values(**values))

    def _save_run(self, connection, run):
        cases = [r.case for r in run.cases]
        if dataset_hash(cases) != run.dataset_hash:
            raise StorageError("run dataset hash does not match evidence")
        payload = [c.model_dump(mode="json") for c in sorted(cases, key=lambda c: c.id)]
        self._immutable(
            connection,
            datasets,
            (datasets.c.name == run.dataset_name) & (datasets.c.version == run.dataset_version),
            {
                "name": run.dataset_name,
                "version": run.dataset_version,
                "content_hash": run.dataset_hash,
                "payload": payload,
            },
        )
        self._immutable(
            connection,
            runs,
            runs.c.id == run.id,
            {
                "id": run.id,
                "dataset_name": run.dataset_name,
                "dataset_version": run.dataset_version,
                "payload": run.model_dump(mode="json"),
            },
        )

    def save_run(self, run: Run):
        try:
            with self.engine.begin() as connection:
                self._save_run(connection, run)
        except StorageError:
            raise
        except Exception:
            raise StorageError("database write failed") from None

    def save_experiment(self, experiment: Experiment):
        try:
            with self.engine.begin() as connection:
                self._save_run(connection, experiment.baseline)
                self._save_run(connection, experiment.candidate)
                self._immutable(
                    connection,
                    experiments,
                    experiments.c.id == experiment.id,
                    {
                        "id": experiment.id,
                        "baseline_id": experiment.baseline.id,
                        "candidate_id": experiment.candidate.id,
                        "payload": experiment.model_dump(mode="json"),
                    },
                )
        except StorageError:
            raise
        except Exception:
            raise StorageError("database write failed") from None

    def _load(self, table, identifier, model):
        try:
            with self.engine.connect() as connection:
                payload = connection.execute(
                    select(table.c.payload).where(table.c.id == identifier)
                ).scalar_one_or_none()
        except Exception:
            raise StorageError("database read failed") from None
        if payload is None:
            raise StorageError(f"snapshot not found: {identifier}")
        return model.model_validate(payload)

    def load_run(self, identifier: str) -> Run:
        return self._load(runs, identifier, Run)

    def load_experiment(self, identifier: str) -> Experiment:
        return self._load(experiments, identifier, Experiment)
