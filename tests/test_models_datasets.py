import json

import pytest
from pydantic import ValidationError

from evalforge.datasets import DatasetError, dataset_hash, load_jsonl, parse_json
from evalforge.models import EvalCase, EvaluatorResult, RegressionRule, TargetResult


def test_jsonl_and_content_version(tmp_path):
    path = tmp_path / "suite.jsonl"
    path.write_text('\n{"id":"a","input":"héllo"}\n\n{"id":"b","input":{}}\n')
    dataset = load_jsonl(path)
    assert dataset.name == "suite"
    assert dataset.version == dataset_hash(dataset.cases[::-1])
    assert load_jsonl(path, version="v1").version == "v1"
    edited = dataset.cases[0].model_copy(update={"input": "changed"})
    assert dataset_hash([edited, dataset.cases[1]]) != dataset.version


@pytest.mark.parametrize(
    "text",
    [
        "",
        "\n",
        "{bad}",
        '{"id":"a","input":"x","unknown":1}',
        '{"id":"a","input":"x"}\n{"id":"a","input":"y"}',
        '{"id":"a","input":"x","weight":0}',
        '{"id":"a","input":"x","critical":"false"}',
        '{"id":"a","input":"x","weight":NaN}',
        '{"id":"a","id":"b","input":"x"}',
    ],
)
def test_invalid_datasets(tmp_path, text):
    path = tmp_path / "bad.jsonl"
    path.write_text(text)
    with pytest.raises(DatasetError, match="bad.jsonl"):
        load_jsonl(path)


def test_line_number(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"id":"a","input":"x"}\n\n{}')
    with pytest.raises(DatasetError, match=":3:"):
        load_jsonl(path)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"output": None},
        {"output": "x", "latency_ms": -1},
        {"status": "ERROR"},
        {"output": "x", "cost": float("inf")},
        {"output": "x", "error": "bad"},
    ],
)
def test_target_invariants(kwargs):
    with pytest.raises(ValidationError):
        TargetResult(**kwargs)


@pytest.mark.parametrize("status,score", [("PASS", None), ("ERROR", 0), ("FAIL", 2)])
def test_evaluator_invariants(status, score):
    with pytest.raises(ValidationError):
        EvaluatorResult(metric="m", evaluator_version="1", status=status, score=score)


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"min_samples": 0, "critical": True},
        {"metric": "latency_ms", "min_score": 0.5},
        {"metric": "m", "max_value": 0.5},
    ],
)
def test_rule_invariants(kwargs):
    with pytest.raises(ValidationError):
        RegressionRule(**kwargs)


def test_json_roundtrip():
    case = EvalCase(id="a", input={"question": "x"}, critical=True)
    assert EvalCase.model_validate_json(case.model_dump_json()) == case
    assert parse_json(json.dumps({"a": 1})) == {"a": 1}
