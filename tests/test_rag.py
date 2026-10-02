import json
from pathlib import Path

import httpx
import pytest

from evalforge.cli import main as cli
from evalforge.config import build_evaluators, build_target, load_config
from evalforge.evaluators import LLMJudge, RAGEvaluator
from evalforge.models import EvalCase, EvaluatorSpec, RetrievalTrace, RetrievedPassage, TargetResult
from evalforge.providers import ChatProvider, Completion
from evalforge.rag import BM25Retriever, RAGTarget, citations


@pytest.fixture
def corpus(tmp_path):
    root = tmp_path / "docs"
    root.mkdir()
    (root / "refund.md").write_text("# Refund policy\nRefunds are available within 14 days.")
    (root / "billing.md").write_text("# Billing\nInvoices can be downloaded from Billing settings.")
    return root


def test_retrieval_is_dynamic_versioned_and_does_not_fill_empty_matches(corpus):
    retriever = BM25Retriever(corpus)
    refund = retriever.retrieve("refund policy", top_k=1)
    assert refund.passages[0].document_id == "refund.md"
    assert retriever.retrieve("invoices", top_k=1).passages[0].document_id == "billing.md"
    assert not retriever.retrieve("xyzzy").passages
    assert BM25Retriever(corpus).corpus_hash == retriever.corpus_hash
    (corpus / "refund.md").write_text("# Refund policy\nRefunds within 30 days.")
    changed = BM25Retriever(corpus)
    assert changed.corpus_hash != retriever.corpus_hash
    assert changed.retrieve("refund").passages[0].chunk_id != refund.passages[0].chunk_id
    assert "14 days" in retriever.retrieve("refund").passages[0].text


def test_chunking_preserves_headings_and_is_bounded(corpus):
    (corpus / "refund.md").write_text("# Refund\none two three four five\n## Exceptions\nsix seven")
    retriever = BM25Retriever(corpus, chunk_words=2)
    assert all(len(c["text"].split()) <= 2 for c in retriever.chunks)
    assert any(c["heading"] == "Exceptions" for c in retriever.chunks)
    assert BM25Retriever(corpus, chunk_words=3).corpus_hash != retriever.corpus_hash
    for k in (0, True, 1.5):
        with pytest.raises(ValueError):
            retriever.retrieve("refund", top_k=k)
    for question in ("", " ", {}):
        with pytest.raises(ValueError):
            retriever.retrieve(question)
    for words in (0, True, 1.5):
        with pytest.raises(ValueError):
            BM25Retriever(corpus, chunk_words=words)


def test_empty_and_symlink_corpora_rejected(tmp_path):
    with pytest.raises(ValueError):
        BM25Retriever(tmp_path)
    with pytest.raises(ValueError):
        BM25Retriever(tmp_path / "missing")
    outside = tmp_path / "outside.md"
    outside.write_text("secret")
    root = tmp_path / "docs"
    root.mkdir()
    (root / "link.md").symlink_to(outside)
    with pytest.raises(ValueError):
        BM25Retriever(root)


def test_actual_provider_path_never_receives_evaluation_labels(corpus, monkeypatch):
    monkeypatch.setenv("TEST_KEY", "controlled-test-key")
    calls = []

    def handler(request):
        payload = json.loads(request.content)
        calls.append(payload)
        evidence = json.loads(payload["messages"][1]["content"])
        assert set(evidence) == {"question", "passages"}
        assert "HIDDEN_LABEL" not in request.content.decode()
        chunk = evidence["passages"][0]
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": f"Refunds within 14 days. [source:{chunk['chunk_id']}]"
                        },
                    }
                ],
                "usage": {"prompt_tokens": 30, "completion_tokens": 10},
            },
        )

    target = RAGTarget(
        BM25Retriever(corpus),
        ChatProvider(model="test", api_key_env="TEST_KEY", transport=httpx.MockTransport(handler)),
        top_k=1,
    )
    result = target.execute(
        EvalCase(
            id="HIDDEN_LABEL",
            input="refund policy",
            reference_answer="HIDDEN_LABEL",
            expected_facts=["HIDDEN_LABEL"],
            metadata={"relevant_document_ids": ["HIDDEN_LABEL"]},
        )
    )
    assert len(calls) == 1
    assert result.input_tokens == 30
    assert result.retrieval.citations == [result.retrieval.passages[0].chunk_id]


def traced_target(corpus, question="refund", top_k=3, output=None):
    trace = BM25Retriever(corpus).retrieve(question, top_k=top_k)
    answer = output if output is not None else f"14 days [source:{trace.passages[0].chunk_id}]"
    trace.citations = citations(answer)
    return TargetResult(output=answer, retrieval=trace)


def evaluator(kind, **options):
    return RAGEvaluator(EvaluatorSpec(metric=kind, kind=kind, options=options))


def test_document_metrics_denominators_and_empty_results(corpus):
    case = EvalCase(id="x", input="refund", metadata={"relevant_document_ids": ["refund.md"]})
    result = traced_target(corpus)
    assert evaluator("retrieval_recall", k=3).evaluate(case, result).score == 1
    assert evaluator("retrieval_precision", k=3).evaluate(case, result).score == pytest.approx(
        1 / 3
    )
    # No matching evidence is a scored retrieval miss, not an execution error.
    empty = traced_target(corpus, question="xyzzy", output="I do not know")
    case.input = "xyzzy"
    assert evaluator("retrieval_recall").evaluate(case, empty).score == 0
    assert evaluator("retrieval_precision").evaluate(case, empty).score == 0


def test_document_metric_deduplicates_chunks(corpus):
    case = EvalCase(id="x", input="refund", metadata={"relevant_document_ids": ["refund.md"]})
    result = traced_target(corpus)
    extra = result.retrieval.passages[0].model_copy(update={"chunk_id": "another", "rank": 2})
    result.retrieval.passages.append(extra)
    assert evaluator("retrieval_precision", k=2).evaluate(case, result).score == 0.5


def test_missing_invalid_and_mismatched_evidence(corpus):
    case = EvalCase(id="x", input="refund")
    result = traced_target(corpus)
    assert evaluator("retrieval_recall").evaluate(case, result).status == "SKIPPED"
    for labels in ([], ["refund.md", "refund.md"], [1], "refund.md", [""]):
        case.metadata = {"relevant_document_ids": labels}
        assert evaluator("retrieval_recall").evaluate(case, result).status == "ERROR"
    case.metadata = {"relevant_document_ids": ["refund.md"]}
    assert evaluator("retrieval_recall", k=4).evaluate(case, result).status == "ERROR"
    assert evaluator("retrieval_recall").evaluate(case, TargetResult(output="x")).status == "ERROR"
    case.input = "different question"
    assert evaluator("citation_validity").evaluate(case, result).status == "ERROR"
    case.input = "refund"
    result.retrieval.citations = ["invented"]
    assert evaluator("citation_validity").evaluate(case, result).status == "ERROR"
    assert (
        evaluator("retrieval_recall").evaluate(case, TargetResult(status="ERROR", error="x")).status
        == "SKIPPED"
    )


def test_citations_check_membership_not_truth(corpus):
    case = EvalCase(id="x", input="refund")
    good = traced_target(corpus)
    assert evaluator("citation_validity").evaluate(case, good).score == 1
    bad = traced_target(corpus, output="Invented [source:missing]")
    assert evaluator("citation_validity").evaluate(case, bad).score == 0
    absent = traced_target(corpus, output="No citation")
    assert evaluator("citation_validity").evaluate(case, absent).score == 0
    assert evaluator("citation_validity", require_citations=False).evaluate(case, absent).score == 1
    assert citations("[source:a] [source:a] [source:b]") == ["a", "b"]


@pytest.mark.parametrize(
    "kind,options",
    [
        ("retrieval_recall", {"k": 0}),
        ("retrieval_recall", {"k": True}),
        ("retrieval_recall", {"threshold": True}),
        ("retrieval_recall", {"threshold": 2}),
        ("retrieval_recall", {"unknown": 1}),
        ("citation_validity", {"require_citations": 1}),
        ("contains", {}),
    ],
)
def test_invalid_evaluator_options(kind, options):
    with pytest.raises(ValueError):
        RAGEvaluator(EvaluatorSpec(metric="x", kind=kind, options=options))


def test_faithfulness_judge_uses_context_not_reference(corpus):
    class Judge:
        def complete(self, messages, *, json_output=False):
            assert json_output
            payload = json.loads(messages[1]["content"])
            assert "reference_answer" not in payload
            assert "HIDDEN" not in messages[1]["content"]
            assert "14 days" in payload["retrieved_passages"][0]["text"]
            return Completion(text='{"score": 1, "reason": "supported", "evidence": ["14 days"]}')

    spec = EvaluatorSpec(
        metric="grounded",
        kind="faithfulness",
        options={"rubric": "Check every claim against retrieved context; unsupported claims fail."},
    )
    judge = LLMJudge(spec, Judge())
    case = EvalCase(id="x", input="refund", reference_answer="HIDDEN")
    assert judge.evaluate(case, traced_target(corpus)).status == "PASS"
    assert judge.evaluate(case, TargetResult(output="x")).status == "ERROR"


def test_trace_rejects_nonconsecutive_duplicate_or_excess_passages():
    p = RetrievedPassage(document_id="a", chunk_id="b", heading="A", text="body", rank=1, score=1)
    for passages in ([p, p], [p.model_copy(update={"rank": 2})]):
        with pytest.raises(ValueError):
            RetrievalTrace(query="q", corpus_hash="hash", top_k=1, passages=passages)


def test_rag_config_and_complete_cli_pipeline(tmp_path, monkeypatch):
    examples = Path(__file__).resolve().parents[1] / "examples/rag"
    config = load_config(examples / "live.json")
    target, snapshot = build_target(config.baseline, base_dir=examples)
    assert snapshot["corpus_hash"] == target.retriever.corpus_hash
    assert snapshot["system_prompt"]
    assert len(build_evaluators(config)) == 4
    monkeypatch.setenv("OPENAI_API_KEY", "controlled-test-key")
    real_client = httpx.Client

    def handler(request):
        payload = json.loads(request.content)
        if "response_format" in payload:
            answer = '{"score": 1, "reason": "Controlled verdict"}'
        else:
            evidence = json.loads(payload["messages"][1]["content"])
            # Simulated provider: answers from retrieved text, never test labels.
            passages = evidence["passages"]
            answer = " ".join(f"{p['text']} [source:{p['chunk_id']}]" for p in passages)
        return httpx.Response(
            200, json={"choices": [{"finish_reason": "stop", "message": {"content": answer}}]}
        )

    monkeypatch.setattr(
        "evalforge.providers.httpx.Client",
        lambda **kwargs: real_client(**{**kwargs, "transport": httpx.MockTransport(handler)}),
    )
    assert cli(["run", "--config", str(examples / "live.json"), "--output-dir", str(tmp_path)]) == 1
    folder = next(p for p in tmp_path.iterdir() if p.is_dir())
    comparison = json.loads((folder / "comparison.json").read_text())
    retrieval = next(g for g in comparison["gates"] if g["rule"]["metric"] == "recall")
    assert retrieval["baseline"] == 1
    assert retrieval["candidate"] == pytest.approx(2 / 3)
    assert "refund_window" in retrieval["case_ids"]
    assert "candidate retrieved sources" in (folder / "report.txt").read_text()


def test_rag_demo_retrieval_and_missing_model(corpus, capsys):
    from evalforge.demos.rag import main

    assert main(["refund", "--documents", str(corpus), "--retrieve-only"]) == 0
    assert "no AI answer generated" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main(["refund", "--documents", str(corpus)])
    assert main(["refund", "--documents", str(corpus), "--retrieve-only", "--top-k", "0"]) == 2
    assert main(["refund", "--documents", str(corpus / "missing"), "--retrieve-only"]) == 2


def test_rag_failure_keeps_retrieval_evidence(corpus, monkeypatch, capsys):
    from evalforge.demos.rag import main

    monkeypatch.delenv("MISSING_KEY", raising=False)
    assert (
        main(
            [
                "refund",
                "--documents",
                str(corpus),
                "--model",
                "test",
                "--api-key-env",
                "MISSING_KEY",
            ]
        )
        == 3
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ERROR"
    assert payload["retrieval"]["passages"][0]["document_id"] == "refund.md"


def test_rag_demo_ai_path_with_controlled_provider(corpus, monkeypatch, capsys):
    from evalforge.demos.rag import main

    class ControlledProvider:
        def __init__(self, **kwargs):
            pass

        def complete(self, messages):
            return Completion(text="Controlled answer")

    monkeypatch.setattr("evalforge.demos.rag.ChatProvider", ControlledProvider)
    assert main(["refund", "--documents", str(corpus), "--model", "test"]) == 0
    assert json.loads(capsys.readouterr().out)["output"] == "Controlled answer"


def test_metric_cutoff_is_passage_rank_not_unique_document_rank(corpus):
    case = EvalCase(id="x", input="refund", metadata={"relevant_document_ids": ["billing.md"]})
    target = traced_target(corpus)
    first = target.retrieval.passages[0]
    target.retrieval.passages.extend(
        [
            first.model_copy(update={"chunk_id": "another", "rank": 2}),
            first.model_copy(
                update={"chunk_id": "billing", "document_id": "billing.md", "rank": 3}
            ),
        ]
    )
    assert evaluator("retrieval_recall", k=2).evaluate(case, target).score == 0


@pytest.mark.parametrize(
    "backend", ["sqlite", pytest.param("postgres", marks=pytest.mark.postgres)]
)
def test_rag_evidence_sql_roundtrip(corpus, tmp_path, backend):
    import os
    from uuid import uuid4

    from evalforge.datasets import dataset_hash
    from evalforge.models import Dataset
    from evalforge.runner import run_experiment
    from evalforge.storage import SQLStore
    from evalforge.targets import LocalTarget

    url = (
        f"sqlite:///{tmp_path / 'rag.db'}"
        if backend == "sqlite"
        else os.environ.get("EVALFORGE_TEST_DATABASE_URL")
    )
    if not url:
        pytest.skip("EVALFORGE_TEST_DATABASE_URL not configured")
    case = EvalCase(id="x", input="refund", metadata={"relevant_document_ids": ["refund.md"]})
    dataset = Dataset(name=f"rag-{uuid4()}", version=dataset_hash([case]), cases=[case])
    target = LocalTarget(lambda _: traced_target(corpus))
    run = run_experiment(dataset, target, [evaluator("retrieval_recall")], target_name="rag")
    store = SQLStore(url)
    try:
        store.save_run(run)
        assert store.load_run(run.id) == run
        assert (
            store.load_run(run.id).cases[0].target.retrieval.passages[0].document_id == "refund.md"
        )
    finally:
        store.close()


def test_question_whitespace_and_multiline_trace_are_preserved(corpus):
    question = "  What is the refund policy?\nPlease explain.  "
    trace = BM25Retriever(corpus).retrieve(question)
    assert trace.query == question
    with pytest.raises(ValueError):
        RetrievalTrace(query=" ", corpus_hash="hash", top_k=1)
