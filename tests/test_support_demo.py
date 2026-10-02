import json
import re
from pathlib import Path

import httpx
import pytest

from evalforge.config import build_target, load_config
from evalforge.demos.support import SupportAssistant, answer_case, main, policy_prompt
from evalforge.demos.workflow import run_demo
from evalforge.models import EvalCase
from evalforge.providers import ChatProvider

EXAMPLES = Path(__file__).resolve().parents[1] / "examples/support"


def test_rule_application_never_reads_expected_answers():
    case = EvalCase(
        id="arbitrary",
        input="Please explain refunds.",
        reference_answer="wrong answer",
        expected_facts=["invented"],
    )
    assert "14 days" in answer_case(case)
    assert "30 days" in answer_case(case, refund_window=30)
    assert (
        SupportAssistant().answer("Where is the nearest train?")
        == "I do not know; please contact support."
    )
    assert "sign-in page" in SupportAssistant().answer("I forgot my password")
    with pytest.raises(ValueError):
        answer_case(EvalCase(id="x", input={}))
    with pytest.raises(ValueError):
        SupportAssistant(refund_window=0)


def test_prompt_files_capture_policy_evidence():
    config = load_config(EXAMPLES / "live.json")
    target, snapshot = build_target(config.baseline, base_dir=EXAMPLES)
    assert target.system_prompt == policy_prompt(14)
    assert snapshot["system_prompt_sha256"]
    assert snapshot["system_prompt"] == policy_prompt(14)


def test_offline_approved_baseline_workflow(tmp_path, capsys):
    assert run_demo(examples_dir=EXAMPLES, output_dir=tmp_path) == 0
    output = capsys.readouterr().out
    assert "Support demo verified" in output
    comparisons = [json.loads(p.read_text()) for p in tmp_path.glob("*/comparison.json")]
    assert sorted(c["status"] for c in comparisons) == ["FAIL", "PASS"]
    candidate = json.loads(
        next(
            p
            for p in tmp_path.glob("*/candidate.json")
            if json.loads(p.read_text())["target_name"] == "support-policy-v2"
        ).read_text()
    )
    failures = [
        r["case"]["id"]
        for r in candidate["cases"]
        if any(e["status"] == "FAIL" for e in r["evaluations"])
    ]
    assert failures == ["refund_window", "refund_late"]
    assert candidate["execution"]["provider_requests"] == 0


def test_live_workflow_without_credentials_has_no_side_effects(tmp_path, monkeypatch):
    monkeypatch.delenv("TEST_KEY", raising=False)
    output = tmp_path / "missing"
    assert (
        run_demo(
            examples_dir=EXAMPLES,
            output_dir=output,
            live=True,
            model="test-model",
            judge_model="test-judge",
            api_key_env="TEST_KEY",
        )
        == 2
    )
    assert not output.exists()


def test_provider_backed_application_with_controlled_transport(monkeypatch):
    monkeypatch.setenv("TEST_KEY", "test-key")

    def handler(request):
        payload = json.loads(request.content)
        assert payload["messages"][0]["content"] == policy_prompt(14)
        assert payload["messages"][1]["content"] == "Can I get a refund after 30 days?"
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"content": "The refund window is 14 days."},
                        "finish_reason": "stop",
                    }
                ]
            },
        )

    app = SupportAssistant(
        provider=ChatProvider(
            model="test", api_key_env="TEST_KEY", transport=httpx.MockTransport(handler)
        )
    )
    assert app.answer("Can I get a refund after 30 days?") == "The refund window is 14 days."


def test_complete_live_pipeline_with_controlled_http(tmp_path, monkeypatch, capsys):
    # These are simulated provider/judge responses; this test makes no live or paid calls.
    monkeypatch.setenv("TEST_KEY", "not-a-real-key")
    calls = []

    def handler(request):
        payload = json.loads(request.content)
        calls.append(payload)
        if "response_format" in payload:
            evidence = json.loads(payload["messages"][1]["content"])
            answer = evidence["answer"].casefold()
            correct = all(f.casefold() in answer for f in evidence["expected_facts"]) and not any(
                f.casefold() in answer for f in evidence["forbidden_facts"]
            )
            output = json.dumps(
                {"score": int(correct), "reason": "Controlled test verdict", "confidence": 1}
            )
        else:
            window = int(
                re.search(r"within (\d+) days", payload["messages"][0]["content"]).group(1)
            )
            output = SupportAssistant(refund_window=window).answer(
                payload["messages"][1]["content"]
            )
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": output}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 30, "completion_tokens": 20},
            },
        )

    real_client = httpx.Client
    monkeypatch.setattr(
        "evalforge.providers.httpx.Client",
        lambda **kwargs: real_client(**{**kwargs, "transport": httpx.MockTransport(handler)}),
    )
    assert (
        run_demo(
            examples_dir=EXAMPLES,
            output_dir=tmp_path,
            live=True,
            model="test-model",
            judge_model="test-judge",
            api_key_env="TEST_KEY",
        )
        == 0
    )
    assert len(calls) == 48  # Seed pair: 32, candidate-only comparison: 16.
    assert sum("response_format" in call for call in calls) == 24
    assert "Support demo verified" in capsys.readouterr().out


def test_standalone_application_entrypoint(capsys):
    assert main(["What is the refund window?"]) == 0
    assert "14 days" in capsys.readouterr().out


def test_local_target_parameters_are_recorded():
    config = load_config(EXAMPLES / "offline.json")
    target, snapshot = build_target(config.candidate)
    assert snapshot["parameters"] == {"refund_window": 30}
    assert "30 days" in target.execute(EvalCase(id="x", input="refund window")).output
