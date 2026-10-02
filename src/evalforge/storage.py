"""Atomic JSON artifacts and transactional immutable SQL snapshots.

SQL tables intentionally store complete domain payloads in JSON/JSONB alongside indexed IDs.
Schema initialization and V1 adoption use packaged, versioned Alembic migrations.
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
    Integer,
    MetaData,
    String,
    Table,
    create_engine,
    insert,
    select,
)
from sqlalchemy.dialects.postgresql import JSONB

from evalforge.datasets import dataset_hash, parse_json
from evalforge.migrations import upgrade_database
from evalforge.models import BaselineApproval, Experiment, Run


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


approvals = Table(
    "evalforge_baseline_approvals",
    metadata,
    Column("sequence", Integer, primary_key=True, autoincrement=True),
    Column("id", String, nullable=False, unique=True),
    Column("name", String, nullable=False),
    Column("run_id", String, ForeignKey("evalforge_runs.id"), nullable=False),
    Column("payload", json_type, nullable=False),
)


class SQLStore:
    def __init__(self, url: str):
        self.engine = None
        try:
            self.engine = create_engine(url, hide_parameters=True)
            upgrade_database(self.engine)
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

    def approve_baseline(
        self, run: Run, *, name: str, approved_by: str, note: str = ""
    ) -> BaselineApproval:
        validate_baseline(run)
        approval = BaselineApproval(name=name, run_id=run.id, approved_by=approved_by, note=note)
        try:
            with self.engine.begin() as connection:
                self._save_run(connection, run)
                connection.execute(
                    insert(approvals).values(
                        id=approval.id,
                        name=name,
                        run_id=run.id,
                        payload=approval.model_dump(mode="json"),
                    )
                )
        except StorageError:
            raise
        except Exception:
            raise StorageError("baseline approval write failed") from None
        return approval

    def resolve_baseline(self, *, name: str | None = None, run_id: str | None = None) -> Run:
        if (name is None) == (run_id is None):
            raise ValueError("select exactly one baseline name or run ID")
        condition = approvals.c.name == name if name is not None else approvals.c.run_id == run_id
        try:
            with self.engine.connect() as connection:
                approved_run = connection.execute(
                    select(approvals.c.run_id)
                    .where(condition)
                    .order_by(approvals.c.sequence.desc())
                    .limit(1)
                ).scalar_one_or_none()
        except Exception:
            raise StorageError("baseline lookup failed") from None
        if approved_run is None:
            raise StorageError("approved baseline not found")
        run = self.load_run(approved_run)
        validate_baseline(run)
        return run

    def baseline_history(self, name: str) -> list[BaselineApproval]:
        with self.engine.connect() as connection:
            payloads = (
                connection.execute(
                    select(approvals.c.payload)
                    .where(approvals.c.name == name)
                    .order_by(approvals.c.sequence)
                )
                .scalars()
                .all()
            )
        return [BaselineApproval.model_validate(p) for p in payloads]


def validate_baseline(run: Run):
    # Revalidate nested objects because callers may have mutated their evidence in memory.
    run = Run.model_validate(run.model_dump())
    if dataset_hash([r.case for r in run.cases]) != run.dataset_hash:
        raise ValueError("baseline dataset hash does not match evidence")
    if run.status == "ERROR" or any(
        e.status in {"ERROR", "UNKNOWN"} for row in run.cases for e in row.evaluations
    ):
        raise ValueError("cannot approve a baseline with execution errors or unknown judgments")
    if not any(e.score is not None for row in run.cases for e in row.evaluations):
        raise ValueError("cannot approve a baseline without scored evidence")
    for row in run.cases:
        if row.case.critical and (
            any(e.status == "FAIL" for e in row.evaluations)
            or not any(e.score is not None for e in row.evaluations)
        ):
            raise ValueError(f"cannot approve failed or unscored critical case: {row.case.id}")
