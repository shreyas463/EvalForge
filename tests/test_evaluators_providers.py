import json

import httpx
import pytest
from jsonschema.exceptions import SchemaError

from evalforge.evaluators import DeterministicEvaluator, LLMJudge
from evalforge.models import EvalCase, EvaluatorSpec, TargetResult
from evalforge.providers import ChatProvider, Completion, ProviderError
from evalforge.targets import LLMTarget, LocalTarget, MockTarget

CASE = EvalCase(
    id="a", input="question", reference_answer="Yes", expected_facts=["Yes"], forbidden_facts=["No"]
)


@pytest.mark.parametrize(
    "kind,options,output,status",
    [
        ("exact_match", {}, "Yes", "PASS"),
        ("exact_match", {}, "yes", "FAIL"),
        ("exact_match", {"case_sensitive": False, "strip": True}, " yes ", "PASS"),
        ("contains", {}, "Yes indeed", "PASS"),
        ("contains", {}, "Maybe", "FAIL"),
        ("excludes", {}, "Yes", "PASS"),
        ("excludes", {}, "No", "FAIL"),
        ("regex", {"pattern": r"\d+"}, "123", "PASS"),
        ("regex", {"pattern": r"\d+"}, "id123", "FAIL"),
        ("regex", {"pattern": r"\d+", "fullmatch": False}, "id123", "PASS"),
        ("json_schema", {"schema": {"type": "object", "required": ["x"]}}, '{"x":1}', "PASS"),
        ("json_schema", {"schema": {"type": "object", "required": ["x"]}}, "{}", "FAIL"),
        ("json_schema", {"schema": {}}, "{bad}", "FAIL"),
        ("json_schema", {"schema": {}}, '{"x":NaN}', "FAIL"),
        ("numeric", {"expected": 12.5, "tolerance": 0.01}, "12.505", "PASS"),
        ("numeric", {"expected": 12.5}, "12.6", "FAIL"),
        ("numeric", {"expected": 12.5}, "NaN", "FAIL"),
        ("numeric", {"expected": 12.5}, "hello", "FAIL"),
    ],
)
def test_checks(kind, options, output, status):
    evaluator = DeterministicEvaluator(EvaluatorSpec(metric="m", kind=kind, options=options))
    result = evaluator.evaluate(CASE, TargetResult(output=output))
    assert result.status == status
    assert result.score == (1 if status == "PASS" else 0)
    assert result.explanation


@pytest.mark.parametrize("kind", ["exact_match", "contains", "excludes", "json_schema"])
def test_not_applicable_is_not_pass(kind):
    evaluator = DeterministicEvaluator(EvaluatorSpec(metric="m", kind=kind))
    assert (
        evaluator.evaluate(EvalCase(id="b", input="x"), TargetResult(output="x")).status
        == "SKIPPED"
    )


@pytest.mark.parametrize(
    "kind,options",
    [
        ("regex", {"pattern": "["}),
        ("exact_match", {"case_sensitive": "false"}),
        ("numeric", {"expected": 0, "tolerance": -1}),
        ("numeric", {"expected": True}),
        ("json_schema", {"schema": {"type": "nonsense"}}),
        ("contains", {"typo": True}),
    ],
)
def test_invalid_evaluator_config(kind, options):
    with pytest.raises((ValueError, SchemaError)):
        DeterministicEvaluator(EvaluatorSpec(metric="m", kind=kind, options=options))


class FixtureProvider:
    def __init__(self, text):
        self.text, self.calls = text, []

    def complete(self, messages, *, json_output=False):
        self.calls.append((messages, json_output))
        return Completion(text=self.text, input_tokens=10, output_tokens=5)


def test_judge_and_target_boundary():
    provider = FixtureProvider(
        '{"score":0.8,"confidence":0.9,"reason":"correct","evidence":["Yes"]}'
    )
    spec = EvaluatorSpec(
        metric="correctness",
        kind="judge",
        version="rubric-v1",
        options={"rubric": "Does it answer the question?", "threshold": 0.9},
    )
    result = LLMJudge(spec, provider).evaluate(CASE, TargetResult(output="Yes"))
    assert result.status == "FAIL" and result.score == 0.8
    assert result.evaluator_version == "rubric-v1"
    assert provider.calls[0][1] is True
    assert json.loads(provider.calls[0][0][1]["content"])["answer"] == "Yes"
    assert LLMTarget(provider).execute(CASE).input_tokens == 10
    assert LocalTarget(lambda c: c.input.upper()).execute(CASE).output == "QUESTION"
    with pytest.raises(ValueError, match="no mock response"):
        MockTarget({}).execute(CASE)


@pytest.mark.parametrize(
    "text",
    [
        "not json",
        '{"score":2,"reason":"bad"}',
        '{"score":0.5,"reason":"bad","extra":1}',
        '{"score":"0.5","reason":"bad"}',
    ],
)
def test_invalid_judge_output(text):
    judge = LLMJudge(
        EvaluatorSpec(metric="m", kind="judge", options={"rubric": "test"}), FixtureProvider(text)
    )
    with pytest.raises(ValueError):
        judge.evaluate(CASE, TargetResult(output="Yes"))


def test_http_provider(monkeypatch):
    monkeypatch.setenv("TEST_KEY", "secret")

    def handler(request):
        assert request.headers["Authorization"] == "Bearer secret"
        assert request.url.path == "/v1/chat/completions"
        assert json.loads(request.content)["response_format"] == {"type": "json_object"}
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            },
        )

    provider = ChatProvider(
        model="test",
        api_key_env="TEST_KEY",
        transport=httpx.MockTransport(handler),
        input_cost_per_million=1,
        output_cost_per_million=2,
    )
    result = provider.complete([], json_output=True)
    assert result.cost == 0.00002


@pytest.mark.parametrize(
    "status,body",
    [
        (429, {"error": "secret"}),
        (200, {}),
        (200, {"choices": [{"message": {"content": "truncated"}, "finish_reason": "length"}]}),
    ],
)
def test_provider_errors_sanitized(monkeypatch, status, body):
    monkeypatch.setenv("TEST_KEY", "secret")
    provider = ChatProvider(
        model="test",
        api_key_env="TEST_KEY",
        transport=httpx.MockTransport(lambda r: httpx.Response(status, json=body)),
    )
    with pytest.raises(ProviderError) as exc:
        provider.complete([])
    assert "secret" not in str(exc.value)


def test_missing_key(monkeypatch):
    monkeypatch.delenv("TEST_KEY", raising=False)
    with pytest.raises(ProviderError, match="missing credential"):
        ChatProvider(model="test", api_key_env="TEST_KEY").complete([])


@pytest.mark.parametrize("failure", ["timeout", "transport"])
def test_http_timeout_and_transport_errors(monkeypatch, failure):
    monkeypatch.setenv("TEST_KEY", "secret")

    def handler(request):
        if failure == "timeout":
            raise httpx.ReadTimeout("secret", request=request)
        raise httpx.ConnectError("secret", request=request)

    provider = ChatProvider(
        model="test", api_key_env="TEST_KEY", transport=httpx.MockTransport(handler)
    )
    with pytest.raises(ProviderError) as exc:
        provider.complete([])
    assert "secret" not in str(exc.value)


def test_external_schema_reference_cannot_fetch():
    from jsonschema.exceptions import _WrappedReferencingError

    evaluator = DeterministicEvaluator(
        EvaluatorSpec(
            metric="schema",
            kind="json_schema",
            options={"schema": {"$ref": "https://example.invalid/schema.json"}},
        )
    )
    with pytest.raises(_WrappedReferencingError):
        evaluator.evaluate(CASE, TargetResult(output="{}"))


def test_case_schema_and_target_failure_skips():
    evaluator = DeterministicEvaluator(EvaluatorSpec(metric="schema", kind="json_schema"))
    case = CASE.model_copy(update={"expected_schema": {"type": "integer"}})
    assert evaluator.evaluate(case, TargetResult(output="1")).status == "PASS"
    assert (
        evaluator.evaluate(case, TargetResult(status="ERROR", error="failed")).status == "SKIPPED"
    )
    judge = LLMJudge(
        EvaluatorSpec(metric="judge", kind="judge", options={"rubric": "test"}),
        FixtureProvider("invalid"),
    )
    assert judge.evaluate(case, TargetResult(status="ERROR", error="failed")).status == "SKIPPED"


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"rubric": ""},
        {"rubric": "test", "threshold": 2},
        {"rubric": "test", "threshold": True},
        {"rubric": "test", "typo": 1},
    ],
)
def test_invalid_judge_config(options):
    with pytest.raises(ValueError):
        LLMJudge(
            EvaluatorSpec(metric="judge", kind="judge", options=options), FixtureProvider("{}")
        )
