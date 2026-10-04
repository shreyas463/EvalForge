# Real public-data retrieval benchmark

This example uses **200 human-written SQuAD 1.1 development questions** and **100 Wikipedia
paragraphs spanning 44 article topics**, downloaded from the official Stanford dataset repository.
It runs entirely offline after checkout: no key, model, judge or runtime download is required.

The data is a deterministic subset, not the official SQuAD leaderboard task. Each paragraph is
one Markdown document, and its question's labeled supporting document is recorded in JSONL.
Reference answers and supporting-document labels enter evaluators only, never the retrieval target.
The target returns a retrieval trace and explicitly says that no AI answer was generated.

## Run and see it

From the repository root, with EvalForge installed:

```bash
evalforge dashboard --config examples/squad/retrieval.json --config examples/squad/unchanged.json --open
```

1. Choose `retrieval.json` and select **Run retrieval evaluation · no AI key**.
2. The comparison is expected to be **Blocked** because the changed collection lacks ten source files.
3. Review the score, filter questions by topic or search text, and expand **Retrieved sources**.
4. Open **Save a reference version**. Select the original version, enter a reference name and your
   audit name, confirm that you reviewed it, and approve. This records your decision, not a model verdict.
5. Select that reference in **Reference version** above. The next run reuses its exact saved run
   rather than executing the original target again. Only compatible datasets and evaluator settings
   are offered; the server checks compatibility too.
6. **Job history and saved references** shows completed, stopped, interrupted and failed jobs,
   links to evidence and approval history. **Stop evaluation** terminates the active owned process.

To check a non-regressing pair, choose `unchanged.json`. Both targets retrieve the complete
collection. Passing means no regression against the configured reference, not perfect retrieval.

CLI equivalents:

```bash
evalforge validate examples/squad/cases.jsonl
evalforge run --config examples/squad/unchanged.json
evalforge run --config examples/squad/retrieval.json
```

The unchanged pair exits 0. The deliberately missing-source pair exits 1. Execution errors exit 3.
On the checked-in subset, measured source recall at five passages is **98% → 88.5%**. This is
retrieval coverage of the dataset's annotated source, not AI answer accuracy or a general benchmark
claim. Nineteen questions regress; four additional questions already miss their source in the
original retriever. Removing sources is deliberate fault injection; the underlying questions,
answers and paragraphs are real public data.

## Provenance, license and reproduction

See [ATTRIBUTION.md](ATTRIBUTION.md) and [manifest.json](manifest.json). Derived dataset content
is CC BY-SA 4.0. The manifest records the source revision, SHA-256, question IDs through the cases,
article links, removed files and hashes of the generated artifacts. EvalForge source code has a
separate license. The added README is explanatory project documentation, not generated dataset content.

Download the `source_url` listed in the manifest into a local file. Then reproduce in a **new** folder:

```bash
evalforge import-squad --source /path/to/dev-v1.1.json --destination /path/to/new-squad
```

The importer requires the pinned file checksum, refuses an existing destination, validates answer
spans, samples paragraphs by stable ID hash, and selects two questions per paragraph. It does not
sample based on retriever success. There is no download during tests or evaluation. To use a larger
subset, set `--paragraphs N --questions-per-paragraph N`; this changes the task and recorded scores.

## Use your own data, or prepare AI evaluation later

```bash
evalforge setup --dataset /path/to/cases.jsonl \
  --baseline-documents /path/to/original-markdown \
  --candidate-documents /path/to/changed-markdown \
  --output /path/to/new-config.json
```

Your JSONL cases need string questions and `metadata.relevant_document_ids` matching relative
Markdown filenames. Setup validates the data, corpora and evaluators and writes absolute file
references. It refuses to replace an existing configuration. Enable the generated file with
`evalforge dashboard --config /path/to/new-config.json --open`.

Add `--model MODEL_NAME --judge-model JUDGE_NAME` to prepare model-backed RAG with bounded
requests, time and output tokens. `--base-url` and `--api-key-env` choose the compatible provider.
Setup makes no model call. The browser's **Set up the AI connection** also edits non-secret model
settings for an already registered RAG file. Keys stay in local environment variables. Models must
support the configured chat options and JSON output for judging; live behavior is still unvalidated.

SQuAD annotates one supporting paragraph and answer spans, not all potentially relevant documents.
This example therefore evaluates annotated-source recall, without claiming exhaustive relevance
precision or semantic answer correctness. BM25 matches words and cannot understand synonyms.
