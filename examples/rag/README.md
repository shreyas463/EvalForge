# Real-model RAG evaluation

This application dynamically retrieves Markdown passages and sends them to your configured
chat model to generate an answer with `[source:CHUNK_ID]` citations. There is no predefined
answer table or fallback pretending to be AI. The documents are fictional support policies.

From the repository root, after installation:

```bash
python -m evalforge.demos.rag 'What is the refund policy?' --retrieve-only
```

This is search inspection only, with no answer generation or API calls.

You can inspect the same retrieval in the browser:

```bash
evalforge dashboard --config examples/rag/live.json --open
```

Select the RAG configuration, enter `refund` under **Check what the documents say**, and select
**Preview sources · no model calls**. The original collection finds the 14-day refund policy;
the changed collection has no matching refund passage. Other questions may retrieve unrelated
passages because search matches words rather than meaning. The view shows ranked source text,
file names and optional search details. It rebuilds both collections on each preview, so edited
Markdown appears on the next search. No credential is required, no answers are generated and
no evaluation is saved. Missing AI setup blocks **Run evaluation**, not source previews.

For AI answers, configure `OPENAI_API_KEY` locally (never commit a key), then:

```bash
python -m evalforge.demos.rag 'What is the refund policy?' --model YOUR_CHAT_MODEL
```

`YOUR_CHAT_MODEL` is a placeholder: replace it with the ID of a chat model supported by your
provider. `--base-url` and `--api-key-env` support other OpenAI-compatible endpoints, including
local model servers that support the adapter's Chat Completions options. The adapter requires
a nonempty credential variable even when a local server ignores authentication. Output includes
the actual generated answer, passages, citations and reported token usage. Calls may be paid.

To evaluate changes, edit `live.json`: replace both target model IDs and the judge model ID.
The judge must support JSON object mode. Then:

```bash
evalforge validate examples/rag/cases.jsonl
evalforge run --config examples/rag/live.json
```

The baseline retrieves from all three documents; the candidate is missing `refunds.md`.
Both generate actual model answers. The expected retrieval regression blocks the candidate:
exit 1 means quality failure, 3 means execution/evidence failure, 2 invalid configuration.
Six answers and six judge verdicts are normally requested, bounded by 30 total HTTP attempts,
400 output tokens per call and a cooperative 180-second budget. No hard dollar cap is claimed.

`report.txt` and all JSON evidence are saved under the repository's `.evalforge/` directory.
Failed cases show old/new answers and retrieved passages. Generation can vary: a model may
correctly abstain when the source is missing, but retrieval recall still falls. The demo does not
claim a guaranteed generated hallucination. No live model behavior has been validated yet.

## Metrics and limitations

- `retrieval_recall`: unique relevant documents in the first K passages / total labeled relevant
  documents. Labels live in `case.metadata.relevant_document_ids`, hidden from the app.
- `retrieval_precision`: unique relevant documents in the first K passages / K. Repeated-document
  passages and unfilled slots do not create extra hits. Labels must be exhaustive for meaningful
  precision. These are document-level metrics, not chunk-level relevance judgments.
- `citation_validity`: fraction of unique parsed `[source:...]` citations pointing to retrieved
  chunks; no citations fails by default. This checks membership, not semantic support.
- `faithfulness`: rubric-based model judgment using retrieved context, without reference answers.
  A grounded abstention can pass faithfulness while retrieval fails. The score is not calibrated.

Malformed labels and absent/mismatched retrieval evidence are explicit errors. Missing labels
skip retrieval metrics. The current BM25 tokenizer casefolds Unicode word tokens but does not
stem, expand synonyms or provide semantic search. It returns only positive-scoring passages;
zero lexical matches returns empty context. Corpus snapshots are rebuilt per target construction.

BM25 background: [Robertson and Zaragoza (2009)](https://doi.org/10.1561/1500000019).
Automated tests use clearly simulated provider responses; they do not prove live-model quality.
