import json
import os

import pytest
from sqlalchemy import select

from evalforge.cli import main
from evalforge.config import ProviderConfig, build_target, load_config
from evalforge.datasets import dataset_hash
from evalforge.evaluators import DeterministicEvaluator
from evalforge.models import Dataset, EvalCase, EvaluatorSpec, Experiment, RegressionRule
from evalforge.regression import compare
from evalforge.runner import run_experiment
from evalforge.storage import SQLStore, StorageError, experiments, read_run, runs, write_json
from evalforge.targets import MockTarget


def sample_experiment():
    dataset = Dataset(
        name="test", version="v1", cases=[EvalCase(id="a", input="x", reference_answer="yes")]
    )
    evaluator = DeterministicEvaluator(EvaluatorSpec(metric="match", kind="exact_match"))
    baseline = run_experiment(
        dataset, MockTarget({"a": "yes"}), [evaluator], target_name="baseline"
    )
    candidate = run_experiment(
        dataset, MockTarget({"a": "no"}), [evaluator], target_name="candidate"
    )
    return Experiment(
        baseline=baseline,
        candidate=candidate,
        comparison=compare(baseline, candidate, [RegressionRule(max_regression=0)]),
    )


@pytest.fixture
def store(tmp_path):
    database = SQLStore(f"sqlite:///{tmp_path / 'runs.db'}")
    yield database
    database.close()


def verify_store(database):
    experiment = sample_experiment()
    database.save_experiment(experiment)
    database.save_experiment(experiment)
    assert database.load_experiment(experiment.id) == experiment
    assert database.load_run(experiment.baseline.id) == experiment.baseline
    conflicting = experiment.baseline.model_copy(deep=True)
    conflicting.target_name = "changed"
    with pytest.raises(StorageError, match="immutable"):
        database.save_run(conflicting)
    with pytest.raises(StorageError, match="not found"):
        database.load_run("missing")


def test_sql_roundtrip_and_idempotence(store):
    verify_store(store)


def test_sql_transaction_rollback_on_version_conflict(store):
    experiment = sample_experiment()
    experiment.candidate.cases[0].case.input = "changed"
    experiment.candidate.dataset_hash = dataset_hash([r.case for r in experiment.candidate.cases])
    with pytest.raises(StorageError, match="immutable"):
        store.save_experiment(experiment)
    with store.engine.connect() as connection:
        assert connection.execute(select(runs)).all() == []
        assert connection.execute(select(experiments)).all() == []


@pytest.mark.postgres
def test_postgres_roundtrip():
    url = os.environ.get("EVALFORGE_TEST_DATABASE_URL")
    if not url:
        pytest.skip("EVALFORGE_TEST_DATABASE_URL not configured")
    database = SQLStore(url)
    try:
        verify_store(database)
    finally:
        database.close()


def test_atomic_json_roundtrip(tmp_path):
    experiment = sample_experiment()
    path = tmp_path / "nested" / "run.json"
    write_json(path, experiment.baseline.model_dump(mode="json"))
    assert read_run(path) == experiment.baseline
    with pytest.raises(ValueError):
        write_json(path, {"invalid": float("nan")})
    assert read_run(path) == experiment.baseline
    assert list(path.parent.iterdir()) == [path]


@pytest.fixture
def config_path(tmp_path):
    path = tmp_path / "experiment.json"
    (tmp_path / "cases.jsonl").write_text(
        '{"id":"a","input":"x","reference_answer":"yes","critical":true}\n'
    )
    write_json(
        path,
        {
            "dataset": "cases.jsonl",
            "baseline": {"kind": "mock", "name": "base", "responses": {"a": "yes"}},
            "candidate": {"kind": "mock", "name": "new", "responses": {"a": "no"}},
            "evaluators": [{"kind": "exact_match", "metric": "accuracy"}],
            "gates": [{"metric": "accuracy", "max_regression": 0, "critical": True}],
        },
    )
    return path


@pytest.mark.parametrize("expected", [0, 1, 3])
def test_cli_exit_codes_and_reports(config_path, capsys, expected):
    config = json.loads(config_path.read_text())
    if expected == 0:
        config["candidate"]["responses"]["a"] = "yes"
    elif expected == 3:
        config["candidate"]["responses"] = {}
    write_json(config_path, config)
    assert main(["run", "--config", str(config_path)]) == expected
    output = capsys.readouterr().out
    assert "EvalForge Quality Gate:" in output
    directories = list((config_path.parent / ".evalforge").iterdir())
    assert len(directories) == 1
    directory = directories[0]
    assert {f.name for f in directory.iterdir()} == {
        "baseline.json",
        "candidate.json",
        "comparison.json",
        "experiment.json",
        "metrics.json",
        "report.txt",
    }
    assert read_run(directory / "baseline.json").status == "COMPLETED"
    assert (
        main(
            [
                "compare",
                "--config",
                str(config_path),
                "--baseline",
                str(directory / "baseline.json"),
                "--candidate",
                str(directory / "candidate.json"),
            ]
        )
        == expected
    )


def test_cli_validate_and_invalid_input(config_path, capsys):
    assert main(["validate", str(config_path.parent / "cases.jsonl")]) == 0
    assert "1 cases" in capsys.readouterr().out
    config = json.loads(config_path.read_text())
    config["gates"][0]["metric"] = "typo"
    write_json(config_path, config)
    assert main(["run", "--config", str(config_path)]) == 2
    assert "Invalid input" in capsys.readouterr().err
    assert not (config_path.parent / ".evalforge").exists()


def test_cli_sql_persistence(config_path):
    url = f"sqlite:///{config_path.parent / 'runs.db'}"
    assert main(["run", "--config", str(config_path), "--database-url", url]) == 1
    saved = next((config_path.parent / ".evalforge").glob("*/experiment.json"))
    experiment = Experiment.model_validate_json(saved.read_text())
    store = SQLStore(url)
    try:
        assert store.load_experiment(experiment.id) == experiment
    finally:
        store.close()


def test_invalid_provider_config():
    for extra in [
        {"base_url": "https://user:secret@example.org/v1"},
        {"base_url": "https://example.org/v1?key=secret"},
        {"timeout": 0},
        {"input_cost_per_million": 1},
        {"api_key": "secret"},
    ]:
        with pytest.raises(ValueError):
            ProviderConfig(model="test", **extra)


def test_local_target_config(config_path, monkeypatch):
    module = config_path.parent / "demo_target.py"
    module.write_text('def respond(case):\n    return "yes"\n')
    monkeypatch.syspath_prepend(str(config_path.parent))
    config = json.loads(config_path.read_text())
    config["candidate"] = {"kind": "local", "name": "local-v1", "callable": "demo_target:respond"}
    write_json(config_path, config)
    target, snapshot = build_target(load_config(config_path).candidate)
    assert snapshot["module_hash"]
    assert target.execute(EvalCase(id="a", input="x")).output == "yes"
    assert main(["run", "--config", str(config_path)]) == 0


def test_cli_missing_files_and_invalid_schema(config_path, capsys):
    assert main(["validate", str(config_path.parent / "missing.jsonl")]) == 3
    assert "FileNotFoundError" in capsys.readouterr().err
    config = json.loads(config_path.read_text())
    config["evaluators"] = [
        {"metric": "accuracy", "kind": "json_schema", "options": {"schema": {"type": "bad"}}}
    ]
    write_json(config_path, config)
    assert main(["run", "--config", str(config_path)]) == 2


def test_judge_provider_identity_captured(config_path):
    from evalforge.config import build_evaluators

    config = json.loads(config_path.read_text())
    config["evaluators"] = [
        {"metric": "accuracy", "kind": "judge", "options": {"rubric": "correctness"}}
    ]
    config["judge_provider"] = {"model": "test-model"}
    write_json(config_path, config)
    judge = build_evaluators(load_config(config_path))[0]
    assert judge.spec.provider_config["model"] == "test-model"
    assert "api_key" not in judge.spec.provider_config


def test_database_initialization_failure_is_operational(config_path, capsys):
    assert main(["run", "--config", str(config_path), "--database-url", "invalid://bad"]) == 3
    assert "database initialization failed" in capsys.readouterr().err
    assert next((config_path.parent / ".evalforge").glob("*/experiment.json")).exists()


def test_config_diagnostics_do_not_echo_secret(config_path, capsys):
    config = json.loads(config_path.read_text())
    config["candidate"] = {
        "kind": "chat",
        "name": "test",
        "provider": {"model": "test", "api_key": "secret-value"},
    }
    write_json(config_path, config)
    assert main(["run", "--config", str(config_path)]) == 2
    error = capsys.readouterr().err
    assert "api_key" in error
    assert "secret-value" not in error


def test_pre_rag_snapshots_remain_idempotent_without_rewriting(store):
    from sqlalchemy import update

    experiment = sample_experiment()
    store.save_experiment(experiment)
    old = experiment.model_dump(mode="json")
    for label in ("baseline", "candidate"):
        for row in old[label]["cases"]:
            row["target"].pop("retrieval")
    with store.engine.begin() as connection:
        for label in ("baseline", "candidate"):
            connection.execute(
                update(runs).where(runs.c.id == old[label]["id"]).values(payload=old[label])
            )
        connection.execute(
            update(experiments).where(experiments.c.id == experiment.id).values(payload=old)
        )
    store.save_experiment(experiment)
    assert store.load_experiment(experiment.id) == experiment
    with store.engine.connect() as connection:
        saved = connection.execute(select(experiments.c.payload)).scalar_one()
    assert "retrieval" not in saved["baseline"]["cases"][0]["target"]
