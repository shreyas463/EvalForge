import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from evalforge.agents import AgentEvaluator, AgentRules
from evalforge.cli import main
from evalforge.config import build_evaluators, build_target, load_config
from evalforge.datasets import load_jsonl
from evalforge.models import AgentTrace, Dataset, EvalCase, EvaluatorSpec, TargetResult, ToolCall
from evalforge.regression import compare
from evalforge.review import calibration_report, export_review
from evalforge.runner import run_experiment
from evalforge.storage import write_json
from evalforge.traces import RecordedTarget

ROOT = Path(__file__).resolve().parents[1]


def call(tool, number=1, *, status="SUCCESS", arguments=None):
    return ToolCall(
        id=str(number),
        tool=tool,
        arguments=arguments or {},
        status=status,
        error="failed" if status == "ERROR" else None,
    )


def evaluate(kind, calls, rules=None, completed=True):
    case = EvalCase(
        id="case", input="task", metadata={} if rules is None else {"agent_rules": rules}
    )
    result = TargetResult(
        output="done", trajectory=AgentTrace(calls=calls, task_completed=completed)
    )
    return AgentEvaluator(EvaluatorSpec(metric=kind, kind=kind)).evaluate(case, result)


@pytest.mark.parametrize(
    "kind,calls,rules,expected",
    [
        ("required_tools", [call("lookup")], {"required_tools": ["lookup"]}, "PASS"),
        (
            "required_tools",
            [call("lookup", status="ERROR")],
            {"required_tools": ["lookup"]},
            "FAIL",
        ),
        (
            "forbidden_tools",
            [call("delete", status="ERROR")],
            {"forbidden_tools": ["delete"]},
            "FAIL",
        ),
        ("forbidden_tools", [call("lookup")], {"forbidden_tools": ["delete"]}, "PASS"),
        ("tool_order", [call("b"), call("a", 2)], {"ordered_tools": ["a", "b"]}, "FAIL"),
        (
            "tool_order",
            [call("a"), call("a", 2), call("b", 3)],
            {"ordered_tools": ["a", "a", "b"]},
            "PASS",
        ),
        (
            "tool_arguments",
            [call("lookup", arguments={"account": "wrong"})],
            {"expected_arguments": {"lookup": {"account": "right"}}},
            "FAIL",
        ),
        (
            "tool_arguments",
            [call("lookup", arguments={"account": "right", "extra": 1})],
            {"expected_arguments": {"lookup": {"account": "right"}}},
            "PASS",
        ),
        (
            "tool_arguments",
            [call("lookup", arguments={"confirmed": 1})],
            {"expected_arguments": {"lookup": {"confirmed": True}}},
            "FAIL",
        ),
        ("tool_arguments", [], {"expected_arguments": {"lookup": {"id": 1}}}, "FAIL"),
        ("tool_efficiency", [call("lookup"), call("lookup", 2)], {"max_calls": 1}, "FAIL"),
        ("tool_efficiency", [], {"max_calls": 0}, "PASS"),
        ("tool_recovery", [call("a", status="ERROR"), call("b", 2)], None, "FAIL"),
        ("tool_recovery", [call("a", status="ERROR"), call("a", 2)], None, "PASS"),
        ("tool_recovery", [], None, "PASS"),
        ("required_tools", [], None, "SKIPPED"),
        ("tool_order", [], {}, "SKIPPED"),
        ("required_tools", [], {"max_calls": True}, "ERROR"),
    ],
)
def test_observable_tool_policy_checks(kind, calls, rules, expected):
    assert evaluate(kind, calls, rules).status == expected


@pytest.mark.parametrize(
    "kind", ["required_tools", "forbidden_tools", "tool_order", "tool_arguments", "tool_efficiency"]
)
def test_missing_policy_fields_skip_and_missing_evidence_errors(kind):
    evaluator = AgentEvaluator(EvaluatorSpec(metric=kind, kind=kind))
    case = EvalCase(id="x", input="task", metadata={"agent_rules": {}})
    assert evaluate(kind, [], {}).status == "SKIPPED"
    assert evaluator.evaluate(case, TargetResult(output="answer")).status == "ERROR"
    assert (
        evaluator.evaluate(case, TargetResult(status="ERROR", error="failed")).status == "SKIPPED"
    )


def test_trace_validation_and_completion_flags():
    with pytest.raises(ValidationError):
        AgentTrace(calls=[call("a"), call("b")])
    with pytest.raises(ValidationError):
        ToolCall(id="1", tool="a", status="ERROR")
    with pytest.raises(ValidationError):
        ToolCall(id="1", tool="a", error="failed")
    with pytest.raises(ValidationError):
        AgentRules(required_tools=["a"], forbidden_tools=["a"])
    with pytest.raises(ValidationError):
        AgentRules(expected_arguments={" ": {}})
    with pytest.raises(ValueError):
        AgentEvaluator(EvaluatorSpec(metric="x", kind="tool_order", options={"fake": 1}))
    assert evaluate("task_completion", [], completed=None).status == "UNKNOWN"
    assert evaluate("task_completion", [], completed=False).status == "FAIL"
    assert evaluate("task_completion", []).status == "PASS"


def test_local_tools_real_execution_and_regression(tmp_path):
    config = load_config(ROOT / "examples/agents/regression.json")
    base = ROOT / "examples/agents"
    dataset = load_jsonl(base / config.dataset)
    runs = []
    for spec in (config.baseline, config.candidate):
        target, snapshot = build_target(spec)
        runs.append(
            run_experiment(
                dataset,
                target,
                build_evaluators(config),
                target_name=spec.name,
                target_config=snapshot,
            )
        )
    assert compare(*runs, config.gates).status == "FAIL"
    assert all(e.status == "PASS" for row in runs[0].cases for e in row.evaluations)
    assert runs[0].cases[0].target.trajectory.calls[1].output == {"total_cents": 1000}
    assert runs[1].cases[0].target.trajectory.calls[1].output == {"total_cents": 1500}
    assert (
        main(
            [
                "run",
                "--config",
                str(base / "regression.json"),
                "--output-dir",
                str(tmp_path / "blocked"),
            ]
        )
        == 1
    )
    assert (
        main(
            [
                "run",
                "--config",
                str(base / "unchanged.json"),
                "--output-dir",
                str(tmp_path / "passing"),
            ]
        )
        == 0
    )


def write_records(path, input="task", result=None):
    record = {
        "case_id": "x",
        "input": input,
        "result": (result or TargetResult(output="observed", latency_ms=42)).model_dump(
            mode="json"
        ),
    }
    path.write_text(json.dumps(record) + "\n")
    return record


def test_recorded_target_exact_inputs_immutable_evidence_and_hash(tmp_path):
    path = tmp_path / "traces.jsonl"
    write_records(path, input={"b": 2, "a": True})
    target = RecordedTarget(path)
    case = EvalCase(id="x", input={"a": True, "b": 2}, reference_answer="never use")
    result = target.execute(case)
    assert result.metadata["observed_latency_ms"] == 42
    result.output = "mutated"
    assert target.execute(case).output == "observed"
    with pytest.raises(ValueError):
        target.execute(case.model_copy(update={"input": {"a": 1, "b": 2}}))
    with pytest.raises(ValueError):
        target.execute(case.model_copy(update={"id": "missing"}))
    from evalforge.config import RecordedConfig

    _, snapshot = build_target(RecordedConfig(kind="recorded", name="saved", records=str(path)))
    assert snapshot["records_sha256"] == target.sha256
    assert snapshot["evidence_mode"] == "recorded"


@pytest.mark.parametrize("contents", ["", "{}\n", '{"case_id":"x","case_id":"x"}\n', "NaN\n"])
def test_recorded_target_rejects_invalid_files(tmp_path, contents):
    path = tmp_path / "trace.jsonl"
    path.write_text(contents)
    with pytest.raises(ValueError):
        RecordedTarget(path)


def test_duplicate_records_and_missing_cases_are_execution_errors(tmp_path):
    path = tmp_path / "trace.jsonl"
    record = write_records(path)
    path.write_text(json.dumps(record) + "\n" + json.dumps(record) + "\n")
    with pytest.raises(ValueError, match="line 2"):
        RecordedTarget(path)
    write_records(path)
    from evalforge.evaluators import DeterministicEvaluator

    evaluator = DeterministicEvaluator(EvaluatorSpec(metric="exact", kind="exact_match"))
    dataset = Dataset(name="saved", version="1", cases=[EvalCase(id="missing", input="task")])
    run = run_experiment(dataset, RecordedTarget(path), [evaluator], target_name="replay")
    assert run.status == "ERROR"
    assert run.cases[0].evaluations[0].status == "SKIPPED"


def make_review_run(tmp_path):
    from evalforge.evaluators import DeterministicEvaluator
    from evalforge.targets import MockTarget

    cases = [
        EvalCase(id="a", input="a", reference_answer="yes"),
        EvalCase(id="b", input="b", reference_answer="yes"),
        EvalCase(id="c", input="c"),
    ]
    run = run_experiment(
        Dataset(name="review", version="1", cases=cases),
        MockTarget({"a": "yes", "b": "no", "c": "no"}),
        [DeterministicEvaluator(EvaluatorSpec(metric="exact", kind="exact_match"))],
        target_name="fixture",
    )
    path = tmp_path / "run.json"
    write_json(path, run.model_dump(mode="json"))
    export_review(path, tmp_path / "review", metric="exact")
    return path, tmp_path / "review/labels.jsonl"


def test_blind_export_and_calibration_audit_without_providers(tmp_path):
    path, labels = make_review_run(tmp_path)
    evidence = json.loads((labels.parent / "evidence.json").read_text())
    assert "evaluations" not in evidence["cases"][0]
    assert calibration_report(path, labels)["metrics"]["exact"]["paired"] == 0
    rows = [json.loads(line) for line in labels.read_text().splitlines()]
    for row, score in zip(rows, [0.0, 0.0, 1.0], strict=True):
        row.update(human_score=score, reviewer="Authored test label, not human validation")
    labels.write_text("".join(json.dumps(row) + "\n" for row in rows))
    report = calibration_report(path, labels)
    group = report["metrics"]["exact"]
    assert group["paired"] == 2 and group["unscored"] == 1
    assert group["status_agreement"] == group["mean_absolute_error"] == 0.5
    assert group["disagreements"][0]["case_id"] == "a"
    assert group["review_coverage"] == 1
    assert len(group["labels"]) == 3
    assert (
        main(
            [
                "review-report",
                "--run",
                str(path),
                "--labels",
                str(labels),
                "--output",
                str(tmp_path / "report.json"),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "review-report",
                "--run",
                str(path),
                "--labels",
                str(labels),
                "--output",
                str(tmp_path / "report.json"),
            ]
        )
        == 2
    )
    assert (
        main(
            [
                "review-export",
                "--run",
                str(path),
                "--metric",
                "exact",
                "--destination",
                str(tmp_path / "another"),
            ]
        )
        == 0
    )
    with pytest.raises(ValueError):
        export_review(path, labels.parent, metric="exact")
    with pytest.raises(ValueError):
        export_review(path, tmp_path / "other", metric="unknown")
    with pytest.raises(ValueError):
        calibration_report(path, labels, threshold=float("nan"))


@pytest.mark.parametrize(
    "mutation", ["duplicate", "run", "case", "metric", "reviewer", "score", "empty"]
)
def test_calibration_rejects_mismatched_or_invalid_labels(tmp_path, mutation):
    path, labels = make_review_run(tmp_path)
    rows = [json.loads(line) for line in labels.read_text().splitlines()]
    if mutation == "duplicate":
        rows.append(rows[0])
    elif mutation in {"run", "case", "metric"}:
        rows[0][{"run": "run_id", "case": "case_id", "metric": "metric"}[mutation]] = "wrong"
    elif mutation == "reviewer":
        rows[0]["human_score"] = 1.0
    elif mutation == "score":
        rows[0]["human_score"] = 2.0
    else:
        rows = []
    labels.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError):
        calibration_report(path, labels)


def test_trace_export_roundtrip_and_cli_recorded_quality_gates(tmp_path):
    from evalforge.traces import export_traces

    path, _ = make_review_run(tmp_path)
    output = tmp_path / "capture.jsonl"
    export_traces(path, output)
    result = RecordedTarget(output).execute(EvalCase(id="a", input="a"))
    assert result.output == "yes"
    assert result.metadata["source_dataset_hash"]
    with pytest.raises(ValueError):
        export_traces(path, output)
    assert main(["trace-export", "--run", str(path), "--output", str(tmp_path / "cli.jsonl")]) == 0
    assert (
        main(
            [
                "run",
                "--config",
                str(ROOT / "examples/agents/recorded.json"),
                "--output-dir",
                str(tmp_path / "recorded"),
            ]
        )
        == 1
    )


def test_old_target_snapshot_remains_identical_without_trajectory():
    result = TargetResult(output="existing")
    assert "trajectory" not in result.model_dump()
    assert TargetResult.model_validate(result.model_dump()).trajectory is None


def test_recorded_file_size_bound_and_invalid_utf8(tmp_path):
    path = tmp_path / "oversized.jsonl"
    with path.open("wb") as destination:
        destination.truncate(16 * 1024 * 1024 + 1)
    with pytest.raises(ValueError, match="16 MiB"):
        RecordedTarget(path)
    path.write_bytes(b"\xff")
    with pytest.raises(ValueError):
        RecordedTarget(path)


def test_local_tool_demo_does_not_use_policy_or_reference_labels():
    from evalforge.demos.tools import quote_case

    case = load_jsonl(ROOT / "examples/agents/cases.jsonl").cases[0]
    altered = case.model_copy(update={"metadata": {}, "reference_answer": "unrelated"})
    assert quote_case(case) == quote_case(altered)


def test_original_latency_survives_run_export_and_repeated_replays(tmp_path):
    from evalforge.evaluators import DeterministicEvaluator
    from evalforge.traces import export_traces

    capture = tmp_path / "original.jsonl"
    write_records(capture, result=TargetResult(output="observed", latency_ms=9876.5))
    dataset = Dataset(
        name="capture",
        version="1",
        cases=[EvalCase(id="x", input="task", reference_answer="observed")],
    )
    evaluator = DeterministicEvaluator(EvaluatorSpec(metric="exact", kind="exact_match"))
    for index in range(3):
        target = RecordedTarget(capture)
        run = run_experiment(dataset, target, [evaluator], target_name="replay")
        result = run.cases[0].target
        assert result.metadata["observed_latency_ms"] == 9876.5
        assert result.latency_ms != 9876.5
        run_path = tmp_path / f"run-{index}.json"
        write_json(run_path, run.model_dump(mode="json"))
        capture = tmp_path / f"capture-{index}.jsonl"
        export_traces(run_path, capture)
    assert (
        RecordedTarget(capture).execute(dataset.cases[0]).metadata["observed_latency_ms"] == 9876.5
    )


@pytest.mark.parametrize("latency", [None, "42", True, -1])
def test_invalid_preserved_latency_rejects_capture(tmp_path, latency):
    capture = tmp_path / "capture.jsonl"
    write_records(
        capture, result=TargetResult(output="observed", metadata={"observed_latency_ms": latency})
    )
    with pytest.raises(ValueError, match="invalid recorded trace"):
        RecordedTarget(capture)


def test_zero_original_latency_is_preserved(tmp_path):
    capture = tmp_path / "capture.jsonl"
    write_records(
        capture,
        result=TargetResult(output="observed", latency_ms=42, metadata={"observed_latency_ms": 0}),
    )
    assert (
        RecordedTarget(capture)
        .execute(EvalCase(id="x", input="task"))
        .metadata["observed_latency_ms"]
        == 0
    )
