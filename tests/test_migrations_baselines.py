import json

import pytest
from alembic import command
from sqlalchemy import create_engine, inspect, select, text
from test_storage_cli import sample_experiment

from evalforge.cli import main
from evalforge.migrations import HEAD, current_revision, migration_config, upgrade_database
from evalforge.models import EvaluatorResult
from evalforge.storage import SQLStore, StorageError, approvals, datasets, experiments, runs


def test_fresh_migration_and_repeat(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'fresh.db'}")
    assert current_revision(engine) is None
    upgrade_database(engine)
    assert current_revision(engine) == HEAD
    upgrade_database(engine)
    assert set(inspect(engine).get_table_names()) == {
        "alembic_version",
        "evalforge_datasets",
        "evalforge_runs",
        "evalforge_experiments",
        "evalforge_baseline_approvals",
    }
    engine.dispose()


def test_legacy_adoption_preserves_evidence(tmp_path):
    url = f"sqlite:///{tmp_path / 'legacy.db'}"
    engine = create_engine(url)
    for table in (datasets, runs, experiments):
        table.create(engine)
    with engine.begin() as connection:
        connection.execute(
            datasets.insert().values(name="legacy", version="v1", content_hash="hash", payload=[])
        )
    upgrade_database(engine)
    assert current_revision(engine) == HEAD
    with engine.connect() as connection:
        assert connection.execute(select(datasets.c.content_hash)).scalar_one() == "hash"
    engine.dispose()


def test_partial_or_mismatched_legacy_schema_rejected(tmp_path):
    for mode in ("partial", "mismatch"):
        url = f"sqlite:///{tmp_path / (mode + '.db')}"
        engine = create_engine(url)
        if mode == "partial":
            datasets.create(engine)
        else:
            for table in (datasets, runs, experiments):
                table.create(engine)
            with engine.begin() as connection:
                connection.execute(text("ALTER TABLE evalforge_runs ADD COLUMN unexpected VARCHAR"))
        with pytest.raises(StorageError, match="initialization"):
            SQLStore(url)
        assert current_revision(engine) is None
        engine.dispose()


def test_upgrade_versioned_core_and_reject_unknown_revision(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'versioned.db'}")
    with engine.begin() as connection:
        command.upgrade(migration_config(connection), "0001_core")
    assert current_revision(engine) == "0001_core"
    upgrade_database(engine)
    assert current_revision(engine) == HEAD
    with engine.begin() as connection:
        connection.execute(text("UPDATE alembic_version SET version_num='unknown'"))
    with pytest.raises(Exception, match="unknown"):
        upgrade_database(engine)
    engine.dispose()


def test_approval_history_and_run_selection(tmp_path):
    database = SQLStore(f"sqlite:///{tmp_path / 'approval.db'}")
    try:
        baseline = sample_experiment().baseline
        database.save_run(baseline)
        with pytest.raises(StorageError, match="not found"):
            database.resolve_baseline(run_id=baseline.id)
        first = database.approve_baseline(
            baseline, name="support", approved_by="tester", note="reviewed"
        )
        replacement = baseline.model_copy(deep=True)
        replacement.id = "new-run"
        second = database.approve_baseline(replacement, name="support", approved_by="tester")
        assert database.resolve_baseline(name="support").id == replacement.id
        assert database.resolve_baseline(run_id=baseline.id).id == baseline.id
        assert database.baseline_history("support") == [first, second]
        assert database.load_run(baseline.id) == baseline
    finally:
        database.close()


@pytest.mark.parametrize("problem", ["error", "unknown", "empty", "hash", "critical_fail"])
def test_invalid_baseline_cannot_be_approved(tmp_path, problem):
    run = sample_experiment().baseline
    if problem == "error":
        run.cases[0].evaluations[0] = EvaluatorResult(
            metric="match", evaluator_version="1", status="ERROR"
        )
        run.status = "ERROR"
    elif problem in {"unknown", "empty"}:
        run.cases[0].evaluations[0] = EvaluatorResult(
            metric="match",
            evaluator_version="1",
            status="UNKNOWN" if problem == "unknown" else "SKIPPED",
        )
    elif problem == "hash":
        run.dataset_hash = "tampered"
    else:
        from evalforge.datasets import dataset_hash

        run.cases[0].case.critical = True
        run.dataset_hash = dataset_hash([run.cases[0].case])
        run.cases[0].evaluations[0] = EvaluatorResult(
            metric="match", evaluator_version="1", status="FAIL", score=0
        )
    store = SQLStore(f"sqlite:///{tmp_path / 'invalid.db'}")
    try:
        with pytest.raises(ValueError):
            store.approve_baseline(run, name="test", approved_by="tester")
        with store.engine.connect() as connection:
            assert connection.execute(select(approvals)).all() == []
    finally:
        store.close()


def test_baseline_cli_and_candidate_only_execution(tmp_path, capsys):
    from evalforge.storage import write_json

    dataset = tmp_path / "cases.jsonl"
    dataset.write_text('{"id":"a","input":"x","reference_answer":"yes"}\n')
    config = tmp_path / "config.json"
    contents = {
        "dataset": "cases.jsonl",
        "baseline": {"kind": "mock", "name": "b", "responses": {"a": "yes"}},
        "candidate": {"kind": "mock", "name": "c", "responses": {"a": "yes"}},
        "evaluators": [{"metric": "match", "kind": "exact_match"}],
        "gates": [{"metric": "match", "min_score": 1}],
    }
    write_json(config, contents)
    url = f"sqlite:///{tmp_path / 'cli.db'}"
    assert main(["db", "status", "--database-url", url]) == 3
    assert main(["db", "upgrade", "--database-url", url]) == 0
    assert main(["db", "status", "--database-url", url]) == 0
    assert main(["run", "--config", str(config), "--database-url", url]) == 0
    baseline_file = next((tmp_path / ".evalforge").glob("*/baseline.json"))
    baseline_id = json.loads(baseline_file.read_text())["id"]
    assert (
        main(
            [
                "baseline",
                "approve",
                "--run",
                str(baseline_file),
                "--name",
                "support",
                "--approved-by",
                "tester",
                "--database-url",
                url,
            ]
        )
        == 0
    )
    assert main(["baseline", "show", "--name", "support", "--database-url", url]) == 0
    assert main(["baseline", "history", "--name", "support", "--database-url", url]) == 0
    # Broken baseline config must never be executed when a stored approved run is selected.
    contents["baseline"]["responses"] = {}
    write_json(config, contents)
    for selector in (["--baseline-name", "support"], ["--baseline-id", baseline_id]):
        assert main(["run", "--config", str(config), "--database-url", url, *selector]) == 0
    assert main(["run", "--config", str(config), "--baseline-name", "support"]) == 2
    capsys.readouterr()
    # Detect incompatible dataset before executing candidate.
    dataset.write_text('{"id":"a","input":"changed","reference_answer":"yes"}\n')
    assert (
        main(["run", "--config", str(config), "--database-url", url, "--baseline-name", "support"])
        == 2
    )
    assert "dataset differs" in capsys.readouterr().err


def test_v1_run_without_execution_field_remains_immutable(tmp_path):
    database = SQLStore(f"sqlite:///{tmp_path / 'old-run.db'}")
    run = sample_experiment().baseline
    database.save_run(run)
    old_payload = run.model_dump(mode="json")
    old_payload.pop("execution")
    with database.engine.begin() as connection:
        connection.execute(runs.update().where(runs.c.id == run.id).values(payload=old_payload))
    reloaded = database.load_run(run.id)
    database.save_run(reloaded)
    database.approve_baseline(reloaded, name="old", approved_by="tester")
    assert database.resolve_baseline(name="old") == reloaded
    with database.engine.connect() as connection:
        assert (
            "execution"
            not in connection.execute(
                select(runs.c.payload).where(runs.c.id == run.id)
            ).scalar_one()
        )
    database.close()


@pytest.mark.postgres
def test_postgres_baseline_approval_and_migration():
    import os
    from uuid import uuid4

    url = os.environ.get("EVALFORGE_TEST_DATABASE_URL")
    if not url:
        pytest.skip("EVALFORGE_TEST_DATABASE_URL not configured")
    store = SQLStore(url)
    try:
        assert current_revision(store.engine) == HEAD
        baseline = sample_experiment().baseline
        name = "postgres-" + str(uuid4())
        record = store.approve_baseline(baseline, name=name, approved_by="integration-test")
        assert store.resolve_baseline(name=name) == baseline
        assert store.resolve_baseline(run_id=baseline.id) == baseline
        assert store.baseline_history(name) == [record]
    finally:
        store.close()
