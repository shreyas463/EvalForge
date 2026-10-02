"""Inspect retrieval locally, or ask a real model using retrieved documents."""

import argparse
import json
import sys

from evalforge.budgets import CURRENT_BUDGET, RunBudget
from evalforge.models import EvalCase, ExecutionLimits
from evalforge.providers import ChatProvider, ProviderError
from evalforge.rag import BM25Retriever, RAGTarget


def main(argv=None):
    parser = argparse.ArgumentParser(description="Document retrieval and real-model RAG demo")
    parser.add_argument("question")
    parser.add_argument("--documents", default="examples/rag/documents/baseline")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--retrieve-only", action="store_true", help="show passages; no AI answer")
    parser.add_argument("--model", help="model ID supported by your endpoint")
    parser.add_argument("--base-url", default="https://api.openai.com/v1")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    args = parser.parse_args(argv)
    if not args.retrieve_only and not args.model:
        parser.error("--model is required for AI answers; --retrieve-only makes no model calls")
    try:
        retriever = BM25Retriever(args.documents)
        if args.retrieve_only:
            print("Retrieval only — no AI answer generated.")
            print(retriever.retrieve(args.question, top_k=args.top_k).model_dump_json(indent=2))
        else:
            provider = ChatProvider(
                model=args.model,
                base_url=args.base_url,
                api_key_env=args.api_key_env,
                max_completion_tokens=400,
            )
            budget = RunBudget(ExecutionLimits(max_provider_requests=1, max_seconds=60))
            token = CURRENT_BUDGET.set(budget)
            try:
                result = RAGTarget(retriever, provider, top_k=args.top_k).execute(
                    EvalCase(id="interactive", input=args.question)
                )
            finally:
                CURRENT_BUDGET.reset(token)
            print(result.model_dump_json(indent=2))
            return 0 if result.status == "SUCCESS" else 3
        return 0
    except (ValueError, OSError, ProviderError) as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 3 if isinstance(exc, (ProviderError, OSError)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
