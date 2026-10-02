import json
import threading
import time

import httpx
import pytest

from evalforge.budgets import CURRENT_BUDGET, BudgetExceeded, RunBudget
from evalforge.evaluators import DeterministicEvaluator, LLMJudge
from evalforge.models import Dataset, EvalCase, EvaluatorSpec, ExecutionLimits, RegressionRule
from evalforge.providers import ChatProvider, ProviderError, retry_after_seconds
from evalforge.regression import compare
from evalforge.runner import run_experiment
from evalforge.targets import LLMTarget, LocalTarget


def successful_response(text="yes", usage=True):
    payload = {"choices": [{"message": {"content": text}, "finish_reason": "stop"}]}
    if usage:
        payload["usage"] = {"prompt_tokens": 10, "completion_tokens": 5}
    return httpx.Response(200, json=payload)


def provider(handler, **kwargs):
    return ChatProvider(
        model="test", api_key_env="TEST_KEY", transport=httpx.MockTransport(handler), **kwargs
    )


@pytest.fixture(autouse=True)
def credential(monkeypatch):
    monkeypatch.setenv("TEST_KEY", "test-value-not-a-real-key")


def test_transient_retries_and_output_limit(monkeypatch):
    calls, sleeps = [], []
    monkeypatch.setattr("evalforge.providers.time.sleep", sleeps.append)

    def handler(request):
        calls.append(json.loads(request.content))
        if len(calls) == 1:
            return httpx.Response(
                429, headers={"Retry-After": "1"}, json={"error": {"code": "rate_limit"}}
            )
        if len(calls) == 2:
            return httpx.Response(503, json={"error": {}})
        return successful_response()

    result = provider(handler, max_attempts=3, max_completion_tokens=100).complete([])
    assert result.attempts == 3 and len(calls) == 3
    assert sleeps[0] == 1
    assert calls[0]["max_completion_tokens"] == 100


@pytest.mark.parametrize(
    "status,code",
    [
        (400, None),
        (401, None),
        (403, None),
        (429, "insufficient_quota"),
        (429, "billing_hard_limit_reached"),
        (429, "billing_not_active"),
    ],
)
def test_terminal_failures_never_retry(status, code):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(status, json={"error": {"code": code, "message": "sensitive"}})

    with pytest.raises(ProviderError) as error:
        provider(handler, max_attempts=3).complete([])
    assert len(calls) == 1 and "sensitive" not in str(error.value)


def test_retry_after_above_limit_is_not_ignored():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(503, headers={"Retry-After": "100"})

    with pytest.raises(ProviderError, match="delay exceeds"):
        provider(handler, max_attempts=3, retry_max_seconds=1).complete([])
    assert len(calls) == 1


def test_completion_deadline_prevents_retry_sleep():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(429, headers={"Retry-After": "5"})

    with pytest.raises(ProviderError, match="deadline"):
        provider(handler, max_attempts=3, timeout=0.01).complete([])
    assert len(calls) == 1


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, None),
        ("bad", None),
        ("-1", None),
        ("NaN", None),
        ("2", 2),
        ("Thu, 01 Jan 1970 00:00:00 GMT", None),
    ],
)
def test_retry_hint_parser(value, expected):
    assert retry_after_seconds(value) == expected


def test_request_budget_counts_retries_and_judge():
    dataset = Dataset(
        name="test", version="v1", cases=[EvalCase(id=str(i), input="q") for i in range(5)]
    )
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(503)
        payload = json.loads(request.content)
        if "response_format" in payload:
            return successful_response('{"score":1,"reason":"correct"}')
        return successful_response()

    p = provider(handler, max_attempts=2, retry_base_seconds=0)
    judge = LLMJudge(
        EvaluatorSpec(metric="judge", kind="judge", options={"rubric": "correctness"}), p
    )
    result = run_experiment(
        dataset,
        LLMTarget(p),
        [judge],
        target_name="test",
        limits=ExecutionLimits(max_provider_requests=4),
    )
    assert len(calls) == 4
    assert result.execution["provider_requests"] == 4
    assert result.execution["provider_retries"] == 1
    assert result.status == "ERROR"
    assert len(result.cases) == 5
    assert result.cases[0].evaluations[0].status == "PASS"
    assert CURRENT_BUDGET.get() is None


def test_concurrency_is_bounded_and_order_is_stable():
    dataset = Dataset(
        name="test",
        version="v1",
        cases=[EvalCase(id=str(i), input="q", reference_answer="yes") for i in range(12)],
    )
    lock = threading.Lock()
    barrier = threading.Barrier(3)
    active = 0
    maximum = 0

    def target(case):
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(active, maximum)
        if int(case.id) < 3:
            barrier.wait(timeout=5)
        time.sleep(0.001 * (12 - int(case.id)))
        with lock:
            active -= 1
        return "yes"

    evaluator = DeterministicEvaluator(EvaluatorSpec(metric="match", kind="exact_match"))
    result = run_experiment(
        dataset,
        LocalTarget(target),
        [evaluator],
        target_name="test",
        limits=ExecutionLimits(concurrency=3),
    )
    assert result.status == "COMPLETED"
    assert maximum == 3
    assert [r.case.id for r in result.cases] == [c.id for c in dataset.cases]


def test_concurrent_request_reservation_never_overshoots():
    dataset = Dataset(
        name="test",
        version="v1",
        cases=[EvalCase(id=str(i), input="q", reference_answer="yes") for i in range(30)],
    )
    calls = []
    lock = threading.Lock()

    def handler(request):
        with lock:
            calls.append(request)
        time.sleep(0.002)
        return successful_response()

    evaluator = DeterministicEvaluator(EvaluatorSpec(metric="match", kind="exact_match"))
    result = run_experiment(
        dataset,
        LLMTarget(provider(handler)),
        [evaluator],
        target_name="test",
        limits=ExecutionLimits(concurrency=8, max_provider_requests=5),
    )
    assert len(calls) == result.execution["provider_requests"] == 5
    assert result.status == "ERROR"


def test_cooperative_deadline_preserves_completed_output(monkeypatch):
    dataset = Dataset(
        name="test", version="v1", cases=[EvalCase(id="a", input="q", reference_answer="yes")]
    )

    clock = [0.0]
    monkeypatch.setattr("evalforge.budgets.time.monotonic", lambda: clock[0])

    def slow(case):
        clock[0] = 1.0
        return "yes"

    evaluator = DeterministicEvaluator(EvaluatorSpec(metric="match", kind="exact_match"))
    result = run_experiment(
        dataset,
        LocalTarget(slow),
        [evaluator],
        target_name="slow",
        limits=ExecutionLimits(max_seconds=0.5),
    )
    assert result.status == "ERROR"
    assert result.cases[0].target.output == "yes"
    assert "time budget" in result.cases[0].target.error
    assert compare(result, result, [RegressionRule(min_score=0)]).status == "ERROR"


def test_observed_cost_stops_future_calls_and_unknown_usage():
    budget = RunBudget(ExecutionLimits(max_observed_cost=0.00002))
    token = CURRENT_BUDGET.set(budget)
    try:
        p = provider(
            lambda r: successful_response(), input_cost_per_million=1, output_cost_per_million=2
        )
        assert p.complete([]).cost == 0.00002
        with pytest.raises(BudgetExceeded, match="cost budget"):
            p.complete([])
        assert budget.requests == 1
    finally:
        CURRENT_BUDGET.reset(token)
    unknown = RunBudget(ExecutionLimits(max_observed_cost=1))
    token = CURRENT_BUDGET.set(unknown)
    try:
        p = provider(
            lambda r: successful_response(usage=False),
            input_cost_per_million=1,
            output_cost_per_million=2,
        )
        p.complete([])
        with pytest.raises(BudgetExceeded, match="unknown"):
            p.complete([])
    finally:
        CURRENT_BUDGET.reset(token)


def test_shared_budget_spans_baseline_and_candidate():
    dataset = Dataset(
        name="test", version="v1", cases=[EvalCase(id="a", input="q", reference_answer="yes")]
    )
    evaluator = DeterministicEvaluator(EvaluatorSpec(metric="match", kind="exact_match"))
    budget = RunBudget(ExecutionLimits(max_provider_requests=1))
    p = provider(lambda r: successful_response())
    baseline = run_experiment(dataset, LLMTarget(p), [evaluator], target_name="b", budget=budget)
    candidate = run_experiment(dataset, LLMTarget(p), [evaluator], target_name="c", budget=budget)
    assert baseline.status == "COMPLETED" and candidate.status == "ERROR"
    assert candidate.execution["provider_requests"] == 1


def test_final_call_cost_overrun_cannot_pass():
    dataset = Dataset(
        name="test", version="v1", cases=[EvalCase(id="a", input="q", reference_answer="yes")]
    )
    evaluator = DeterministicEvaluator(EvaluatorSpec(metric="match", kind="exact_match"))
    p = provider(
        lambda r: successful_response(), input_cost_per_million=1, output_cost_per_million=2
    )
    result = run_experiment(
        dataset,
        LLMTarget(p),
        [evaluator],
        target_name="test",
        limits=ExecutionLimits(max_observed_cost=0.00001),
    )
    assert result.status == "ERROR"
    assert result.cases[0].target.output == "yes"
    assert result.cases[0].target.cost == 0.00002
    assert "cost budget exceeded" in result.cases[0].target.error


def test_provider_deadline_detects_slow_transport():
    def handler(request):
        time.sleep(0.01)
        return successful_response()

    with pytest.raises(ProviderError, match="deadline"):
        provider(handler, timeout=0.001).complete([])


def test_timeout_retry_bounded(monkeypatch):
    calls = []
    monkeypatch.setattr("evalforge.providers.time.sleep", lambda delay: None)

    def handler(request):
        calls.append(request)
        if len(calls) < 3:
            raise httpx.ReadTimeout("sensitive", request=request)
        return successful_response()

    assert provider(handler, max_attempts=3).complete([]).attempts == 3
    assert len(calls) == 3


def test_truncated_output_accounts_known_usage():
    budget = RunBudget(ExecutionLimits(max_observed_cost=1))
    token = CURRENT_BUDGET.set(budget)
    try:

        def handler(request):
            response = successful_response()
            payload = json.loads(response.content)
            payload["choices"][0]["finish_reason"] = "length"
            return httpx.Response(200, json=payload)

        with pytest.raises(ProviderError, match="incomplete"):
            provider(handler, input_cost_per_million=1, output_cost_per_million=2).complete([])
        assert budget.observed_cost == 0.00002 and not budget.unknown_cost
    finally:
        CURRENT_BUDGET.reset(token)


def test_cost_budget_requires_prices_before_any_request():
    budget = RunBudget(ExecutionLimits(max_observed_cost=1))
    token = CURRENT_BUDGET.set(budget)
    try:
        with pytest.raises(BudgetExceeded, match="prices"):
            provider(lambda r: successful_response()).complete([])
        assert budget.requests == 0
    finally:
        CURRENT_BUDGET.reset(token)
