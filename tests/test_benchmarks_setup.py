import hashlib
import json
from pathlib import Path

import pytest

from evalforge.benchmarks import import_squad
from evalforge.cli import main
from evalforge.config import build_evaluators, build_target, load_config
from evalforge.datasets import load_jsonl
from evalforge.models import EvalCase
from evalforge.regression import compare
from evalforge.runner import run_experiment
from evalforge.setup import create_config

ROOT = Path(__file__).resolve().parents[1]


def test_real_bundled_squad_integrity_and_retrieval_regression():
    folder = ROOT / "examples/squad"
    manifest = json.loads((folder / "manifest.json").read_text())
    for path, digest in manifest["files"].items():
        assert hashlib.sha256((folder / path).read_bytes()).hexdigest() == digest
    dataset = load_jsonl(folder / "cases.jsonl", name="squad-dev-retrieval-subset")
    assert len(dataset.cases) == manifest["questions"] == 200
    assert len({case.category for case in dataset.cases}) == manifest["articles"]
    config = load_config(folder / "retrieval.json")
    runs = []
    for spec in (config.baseline, config.candidate):
        target, snapshot = build_target(spec, base_dir=folder)
        runs.append(
            run_experiment(
                dataset,
                target,
                build_evaluators(config),
                target_name=spec.name,
                target_config=snapshot,
            )
        )
    result = compare(*runs, config.gates)
    assert result.status == "FAIL"
    assert result.gates[0].baseline > result.gates[0].candidate
    assert all(row.target.metadata["mode"] == "retrieval_only" for run in runs for row in run.cases)
    unchanged = load_config(folder / "unchanged.json")
    target, snapshot = build_target(unchanged.candidate, base_dir=folder)
    unchanged_run = run_experiment(
        dataset,
        target,
        build_evaluators(unchanged),
        target_name=unchanged.candidate.name,
        target_config=snapshot,
    )
    assert compare(runs[0], unchanged_run, unchanged.gates).status == "PASS"
    # Mutating all answer labels must not change application retrieval.
    target, _ = build_target(config.baseline, base_dir=folder)
    case = dataset.cases[0]
    altered = case.model_copy(update={"reference_answer": "unrelated", "metadata": {}})
    assert target.execute(case) == target.execute(altered)


def test_converter_checks_source_and_refuses_overwrite(tmp_path, monkeypatch):
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps(
            {
                "data": [
                    {
                        "title": "Authored test fixture",
                        "paragraphs": [
                            {
                                "context": "A test passage.",
                                "qas": [
                                    {
                                        "id": "test1",
                                        "question": "What?",
                                        "answers": [{"text": "test", "answer_start": 2}],
                                    }
                                ],
                            }
                        ],
                    }
                ]
            }
        )
    )
    with pytest.raises(ValueError, match="checksum"):
        import_squad(source, tmp_path / "out")
    monkeypatch.setattr(
        "evalforge.benchmarks.SQUAD_SHA256", hashlib.sha256(source.read_bytes()).hexdigest()
    )
    output = tmp_path / "out"
    assert import_squad(source, output, paragraphs=1)["questions"] == 1
    with pytest.raises(ValueError, match="already exists"):
        import_squad(source, output)
    with pytest.raises(ValueError, match="paragraphs"):
        import_squad(source, tmp_path / "other", paragraphs=0)


def test_setup_generates_validated_offline_and_model_configs_without_calls(tmp_path, monkeypatch):
    source = ROOT / "examples/squad"
    calls = []
    monkeypatch.setattr(
        "evalforge.providers.ChatProvider.complete", lambda *a, **k: calls.append(a)
    )
    for model, judge in ((None, None), ("test-chat", "test-judge")):
        output = tmp_path / ("offline.json" if model is None else "model.json")
        create_config(
            source / "cases.jsonl",
            source / "documents/baseline",
            source / "documents/candidate",
            output,
            model=model,
            judge_model=judge,
        )
        config = load_config(output)
        assert config.baseline.kind == ("rag" if model else "retrieval")
        assert Path(config.dataset).is_absolute()
        assert config.execution.max_provider_requests > 0
        build_evaluators(config)
        with pytest.raises(ValueError, match="already exists"):
            create_config(
                source / "cases.jsonl",
                source / "documents/baseline",
                source / "documents/candidate",
                output,
            )
    assert not calls
    assert (
        main(
            [
                "setup",
                "--dataset",
                str(source / "cases.jsonl"),
                "--baseline-documents",
                str(source / "documents/baseline"),
                "--candidate-documents",
                str(source / "documents/candidate"),
                "--output",
                str(tmp_path / "cli.json"),
            ]
        )
        == 0
    )
    with pytest.raises(ValueError, match="answer model"):
        create_config(
            source / "cases.jsonl",
            source / "documents/baseline",
            source / "documents/candidate",
            tmp_path / "bad.json",
            judge_model="judge",
        )


def test_setup_rejects_missing_relevance_labels(tmp_path):
    dataset = tmp_path / "cases.jsonl"
    dataset.write_text(EvalCase(id="1", input="question").model_dump_json() + "\n")
    with pytest.raises(ValueError, match="supporting document labels"):
        create_config(dataset, tmp_path, tmp_path, tmp_path / "config.json")
