import pytest

from evalforge.datasets import dataset_hash
from evalforge.evaluators import DeterministicEvaluator, LLMJudge
from evalforge.models import Dataset, EvalCase, EvaluatorSpec, RegressionRule, TargetResult
from evalforge.regression import ComparisonError, compare
from evalforge.runner import run_experiment
from evalforge.scoring import aggregate, summarize
from evalforge.targets import LocalTarget, MockTarget


@pytest.fixture
def dataset():
    return Dataset(
        name="support",
        version="v1",
        cases=[
            EvalCase(
                id="a", input="refund", reference_answer="14 days", critical=True, category="refund"
            ),
            EvalCase(id="b", input="login", reference_answer="reset", category="account", weight=3),
        ],
    )


def run(dataset, responses):
    evaluator = DeterministicEvaluator(EvaluatorSpec(metric="accuracy", kind="exact_match"))
    return run_experiment(dataset, MockTarget(responses), [evaluator], target_name="test")


def test_paired_regression_and_aggregation(dataset):
    baseline = run(dataset, {"a": "14 days", "b": "reset"})
    candidate = run(dataset, {"a": "30 days", "b": "reset"})
    assert summarize(candidate).value == 0.75
    assert aggregate(candidate)["category:refund"]["accuracy"].value == 0
    comparison = compare(
        baseline,
        candidate,
        [
            RegressionRule(metric="accuracy", max_regression=0.1),
            RegressionRule(metric="accuracy", category="refund", critical=True),
        ],
    )
    assert comparison.status == "FAIL"
    assert comparison.gates[0].delta == -0.25
    assert comparison.gates[1].case_ids == ["a"]


def test_threshold_boundary_and_warn(dataset):
    baseline = run(dataset, {"a": "14 days", "b": "reset"})
    candidate = run(dataset, {"a": "30 days", "b": "reset"})
    assert (
        compare(baseline, candidate, [RegressionRule(max_regression=0.25, min_score=0.75)]).status
        == "PASS"
    )
    assert (
        compare(baseline, candidate, [RegressionRule(max_regression=0.1, severity="warn")]).status
        == "PASS"
    )
    assert (
        compare(baseline, candidate, [RegressionRule(max_relative_regression=0.2)]).status == "FAIL"
    )


def test_errors_do_not_become_scores(dataset):
    baseline = run(dataset, {"a": "14 days", "b": "reset"})
    candidate = run(dataset, {"b": "reset"})
    assert candidate.status == "ERROR"
    assert candidate.cases[0].target.error == "ValueError: execution failed"
    assert candidate.cases[0].evaluations[0].score is None
    assert summarize(candidate, "error_rate").value == 0.5
    assert compare(baseline, candidate, [RegressionRule(min_score=0.1)]).status == "ERROR"


def test_judge_errors_are_preserved(dataset):
    class BrokenProvider:
        def complete(self, *args, **kwargs):
            raise RuntimeError("secret must not appear")

    judge = LLMJudge(
        EvaluatorSpec(metric="judge", kind="judge", options={"rubric": "correct?"}),
        BrokenProvider(),
    )
    result = run_experiment(dataset, MockTarget({"a": "x", "b": "y"}), [judge], target_name="test")
    assert result.status == "ERROR"
    assert result.cases[0].evaluations[0].status == "ERROR"
    assert "secret" not in result.model_dump_json()


@pytest.mark.parametrize("change", ["dataset", "version", "evaluator", "hash"])
def test_incompatible_runs_rejected(dataset, change):
    baseline = run(dataset, {"a": "14 days", "b": "reset"})
    candidate = baseline.model_copy(deep=True)
    if change == "dataset":
        candidate.cases[0].case.input = "changed"
        candidate.dataset_hash = dataset_hash([r.case for r in candidate.cases])
    elif change == "version":
        candidate.dataset_version = "v2"
    elif change == "evaluator":
        candidate.evaluators[0].options = {"strip": True}
    else:
        candidate.dataset_hash = "wrong"
    with pytest.raises(ComparisonError):
        compare(baseline, candidate, [RegressionRule(min_score=0.5)])


def test_missing_metrics_samples_and_cost_fail_closed(dataset):
    baseline = run(dataset, {"a": "14 days", "b": "reset"})
    for rule in [
        RegressionRule(metric="unknown", min_score=0.5),
        RegressionRule(min_score=0.5, min_samples=3),
        RegressionRule(metric="cost", max_value=1),
        RegressionRule(category="unknown", critical=True),
    ]:
        assert compare(baseline, baseline, [rule]).status == "ERROR"


def test_latency_percent_and_zero_baseline(dataset):
    baseline = run(dataset, {"a": "14 days", "b": "reset"})
    candidate = baseline.model_copy(deep=True)
    for row in baseline.cases:
        row.target.latency_ms = 0
    for row in candidate.cases:
        row.target.latency_ms = 1
    result = compare(
        baseline, candidate, [RegressionRule(metric="latency_ms", max_increase_percent=25)]
    )
    assert result.status == "FAIL"
    assert summarize(candidate, "p95_latency_ms").value == 1


def test_skips_and_mutation_isolation(dataset):
    evaluator = DeterministicEvaluator(EvaluatorSpec(metric="facts", kind="contains"))

    def mutate(case):
        case.expected_facts.append("corrupted")
        return TargetResult(output="x", cost=0.1)

    result = run_experiment(dataset, LocalTarget(mutate), [evaluator], target_name="local")
    assert result.cases[0].case.expected_facts == []
    assert summarize(result, "facts").value is None
    assert summarize(result, "cost").value == 0.1
    assert compare(result, result, [RegressionRule(metric="facts", min_score=0)]).status == "ERROR"


def test_weighted_evaluators_and_critical_failure(dataset):
    specs = [
        EvaluatorSpec(metric="exact", kind="exact_match", weight=3),
        EvaluatorSpec(metric="format", kind="regex", options={"pattern": ".*"}),
    ]
    result = run_experiment(
        dataset,
        MockTarget({"a": "wrong", "b": "reset"}),
        [DeterministicEvaluator(s) for s in specs],
        target_name="test",
    )
    assert summarize(result).value == 0.8125
    assert compare(result, result, [RegressionRule(critical=True)]).status == "FAIL"


def test_duplicate_and_reserved_metrics_rejected(dataset):
    evaluator = DeterministicEvaluator(EvaluatorSpec(metric="overall", kind="exact_match"))
    with pytest.raises(ValueError):
        run_experiment(dataset, MockTarget({}), [evaluator], target_name="test")
    with pytest.raises(ComparisonError):
        result = run(dataset, {"a": "14 days", "b": "reset"})
        compare(result, result, [])


def test_options_key_order_does_not_change_pair_compatibility(dataset):
    evaluator = DeterministicEvaluator(
        EvaluatorSpec(
            metric="accuracy", kind="exact_match", options={"strip": True, "case_sensitive": False}
        )
    )
    baseline = run_experiment(
        dataset, MockTarget({"a": "14 days", "b": "reset"}), [evaluator], target_name="test"
    )
    candidate = baseline.model_copy(deep=True)
    candidate.evaluators[0].options = {"case_sensitive": False, "strip": True}
    assert compare(baseline, candidate, [RegressionRule(min_score=1)]).status == "PASS"


def test_unknown_judge_status_and_changed_applicability(dataset):
    baseline = run(dataset, {"a": "14 days", "b": "reset"})
    candidate = baseline.model_copy(deep=True)
    from evalforge.models import EvaluatorResult

    candidate.cases[0].evaluations[0] = EvaluatorResult(
        metric="accuracy", evaluator_version="1", status="UNKNOWN"
    )
    assert compare(baseline, candidate, [RegressionRule(min_score=0.1)]).status == "ERROR"
    candidate.cases[0].evaluations[0] = EvaluatorResult(
        metric="accuracy", evaluator_version="1", status="SKIPPED"
    )
    with pytest.raises(ComparisonError, match="applicability"):
        compare(baseline, candidate, [RegressionRule(min_score=0.1)])


def test_invalid_plugin_results_and_evaluator_failure(dataset):
    class BadTarget:
        def execute(self, case):
            return "not a TargetResult"

    class BadEvaluator:
        spec = EvaluatorSpec(metric="bad", kind="exact_match")

        def evaluate(self, case, target):
            from evalforge.models import EvaluatorResult

            return EvaluatorResult(metric="wrong", evaluator_version="1", status="PASS", score=1)

    result = run_experiment(dataset, BadTarget(), [BadEvaluator()], target_name="bad")
    assert result.status == "ERROR"
    result = run_experiment(
        dataset, MockTarget({"a": "yes", "b": "yes"}), [BadEvaluator()], target_name="bad"
    )
    assert result.cases[0].evaluations[0].status == "ERROR"


def test_p95_nearest_rank_and_cost_increase(dataset):
    baseline = run(dataset, {"a": "14 days", "b": "reset"})
    candidate = baseline.model_copy(deep=True)
    for row in baseline.cases:
        row.target.cost = 0.1
    candidate.cases[0].target.cost = 0.2
    candidate.cases[1].target.cost = 0.3
    candidate.cases[0].target.latency_ms = 1
    candidate.cases[1].target.latency_ms = 100
    assert summarize(candidate, "p95_latency_ms").value == 100
    result = compare(
        baseline, candidate, [RegressionRule(metric="cost", max_value=0.2, max_increase_percent=40)]
    )
    assert result.status == "FAIL"
    candidate.cases[1].target.cost = None
    assert (
        compare(baseline, candidate, [RegressionRule(metric="cost", max_value=1)]).status == "ERROR"
    )


def test_baseline_error_report_is_visible(dataset):
    from evalforge.report import render_report

    baseline = run(dataset, {"b": "reset"})
    candidate = run(dataset, {"a": "14 days", "b": "reset"})
    comparison = compare(baseline, candidate, [RegressionRule(min_score=0.5)])
    report = render_report(baseline, candidate, comparison)
    assert "Baseline target error [a]" in report
    assert comparison.status == "ERROR"
