# EvalForge

**Continuous evaluation and regression gates for AI applications.**

EvalForge runs paired baseline and candidate targets against a versioned JSONL dataset, preserves case-level evidence, scores outputs, and blocks configured regressions. Category and critical-case gates catch failures that an overall average can hide.

**Status: V0/V1 core engine implemented.** This release includes the local CLI, provider-backed model/judge interfaces, JSON artifacts, PostgreSQL snapshot persistence, and a GitHub Actions workflow. The dashboard, RAG/agent evaluators, API/worker service, semantic similarity, human calibration, and production ingestion remain planned.

## Quick start

Requires Python 3.11 or newer. From a repository checkout:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install '.[dev,postgres]'
evalforge validate evals/support.jsonl
evalforge run --config examples/passing.json
evalforge run --config examples/regression.json
```

The passing fixture exits **0**. The deliberately incorrect refund answer exits **1**:

```text
EvalForge Quality Gate: FAIL
accuracy          1 → 0.958333   delta -0.0416667   FAIL
accuracy [refunds] 1 → 0.75      delta -0.25        FAIL
Critical failure: refund_01
Baseline:  "Refunds are available only within 14 days."
Candidate: "Refunds are available within 30 days."
```

These are explicit offline fixture responses for 24 fictional support-policy cases. They demonstrate the engine; they are not model-generated answers or a working support assistant.

Each invocation saves a new directory under `.evalforge/` containing `baseline.json`, `candidate.json`, `comparison.json`, `experiment.json`, `metrics.json`, and `report.txt`. Paths in configuration are relative to the config file. `--output-dir` overrides the artifact root.

## What works today

- Strict Pydantic domain/configuration schemas and JSONL validation with line-numbered errors, duplicate-ID/key detection, UTF-8 handling, and SHA-256 content versions.
- Mock targets, local Python callables, and raw model targets using a synchronous OpenAI-compatible Chat Completions HTTP adapter.
- Versioned exact match, required/forbidden substring, regex, JSON Schema (Draft 2020-12), and numeric-tolerance evaluators.
- Structured LLM judges with explicit rubrics, score/confidence validation, rationale, evidence, provider identity, and token/cost capture.
- Bounded concurrent experiment execution with independent target/evaluator errors; errors are never converted into quality scores.
- Weighted quality scores by metric and category, per-case evidence, average/p95 latency, reported token counts, configured price estimates, and target error rates.
- Paired comparisons requiring identical dataset content/version and evaluator configurations; mismatched applicability is rejected.
- Absolute/relative score regression limits, minimum scores/sample counts, category gates, critical-case gates, operational ceilings/increase limits, and warning/blocking severity.
- Atomic JSON artifact writes and transactional, immutable SQL snapshots in PostgreSQL JSONB (SQLite also supported for local use).
- CLI validation, execution, comparison of saved runs, and CI integration with pytest and coverage checks.

## Dataset format

One JSON object per line:

```json
{"id":"refund_01","input":"Can I get a refund after 30 days?","reference_answer":"Refunds are available only within 14 days.","category":"refunds","critical":true,"tags":["policy"]}
```

Supported fields: `id`, `input` (string or JSON object), `reference_answer`, `expected_facts`, `forbidden_facts`, `expected_schema`, `category`, `tags`, `critical`, `weight`, and `metadata`. Unknown fields and non-standard JSON constants are rejected. Empty lines are ignored; empty datasets are invalid. Content hashes include every case field, with cases sorted by ID. A supplied version label is recorded alongside the hash; changing content prevents comparison even if the label stays the same.

## Configure targets and gates

Use the complete examples in [`examples/passing.json`](examples/passing.json), [`examples/regression.json`](examples/regression.json), and [`examples/chat-judge.json`](examples/chat-judge.json).

Evaluator options:

| Kind | Options / case evidence |
| --- | --- |
| `exact_match` | `reference_answer`; optional `strip` and `case_sensitive` (defaults false/true) |
| `contains` | All `expected_facts` as literal substrings; optional `case_sensitive` |
| `excludes` | No `forbidden_facts` as literal substrings; optional `case_sensitive` |
| `regex` | Required `pattern`; `fullmatch` defaults true |
| `json_schema` | `options.schema` or the case's `expected_schema`; strict JSON parsing |
| `numeric` | Required numeric `expected`; absolute `tolerance` defaults 0 |
| `judge` | Required `rubric`; `threshold` defaults 0.5; top-level `judge_provider` required |

The substring checks verify text presence, not factual truth or semantics. Evaluators without required case evidence emit `SKIPPED`. Judge scores are model judgments, and self-reported confidence is not calibrated. JSON Schema supports local references; external references are not fetched.

Every evaluator requires a unique `metric`; optional `version` and `weight` default to `1`. Reserved aggregate/operational names are `overall`, `latency_ms`, `p95_latency_ms`, `cost`, and `error_rate`.

```json
{"metric":"accuracy","category":"refunds","max_regression":0.03,"min_samples":4,"critical":true,"severity":"block"}
```

`max_regression` is an absolute score decrease (0.03 = three percentage points). `max_relative_regression` is a fraction of baseline quality (0.05 = 5%). For lower-is-better operational metrics, use `max_value` and/or `max_increase_percent` (25 = 25%). A zero baseline permits no increase. Limits are inclusive. Critical gates reject **any candidate FAIL** for the selected metric(s) on critical cases, even when the baseline also fails. An overall critical gate examines individual evaluator decisions, not only their weighted mean.

Missing metrics, insufficient samples, unknown/error results, absent critical-case evidence, and partially missing cost data produce an error decision. Warnings do not block quality regressions, but execution/evidence errors still block. Runs with execution errors cannot pass by selecting an unrelated gate.

### Local application target

Provide an importable `module:function` receiving an `EvalCase` and returning a string or `TargetResult`:

```python
from evalforge.models import EvalCase

def respond(case: EvalCase) -> str:
    return my_application.answer(case.input)
```

```json
{"kind":"local","name":"application-v1","callable":"my_app.eval_target:respond"}
```

Install your application package or set `PYTHONPATH` so the module is importable. Local target configuration executes Python code and should come from trusted sources. The module file hash is recorded when available; it is not a complete dependency/environment fingerprint.

### Model targets and judges

In `examples/chat-judge.json`, replace `YOUR_CHAT_MODEL` and `YOUR_JUDGE_MODEL` with model IDs supported by your endpoint, then set the credential variable named by `api_key_env` (default `OPENAI_API_KEY`). You may set a different `base_url` and credential variable for each target and the judge.

```bash
evalforge run --config examples/chat-judge.json
```

The adapter uses `/chat/completions`, temperature, text responses, and JSON object mode for judges, following the [Chat Completions API contract](https://developers.openai.com/api/reference/resources/chat). Models/endpoints must support those options. There are no native Anthropic/Gemini, Responses API, streaming, tool-call, or multimodal adapters in this release. Provider and judge behavior is tested with controlled HTTP/structured fixtures; **live paid API calls have not been validated**. The chat example is a starting configuration, not a promised passing benchmark.

Token counts come from provider usage. Cost remains unknown unless both `input_cost_per_million` and `output_cost_per_million` are supplied and usage is returned. Prices are user-configured estimates; no pricing is hardcoded. Judge usage/cost is recorded separately from target cost. Keys are read from the environment and are not stored in run configuration. HTTP errors exclude response bodies and credential values.

## Save to PostgreSQL

Install the `postgres` extra and point at an existing database:

```bash
export EVALFORGE_DATABASE_URL='postgresql+psycopg://user:password@localhost:5432/evalforge'
evalforge run --config examples/passing.json
```

The CLI always writes JSON evidence and additionally stores the experiment and both runs when a database URL is supplied. SQLStore runs packaged Alembic migrations on first use; the database account needs schema migration permission. Existing V1 tables are adopted only after checking columns, types, primary keys, and foreign keys. Dataset `(name, version)` and run/experiment IDs are immutable; conflicting writes fail and transactions roll back. Exact repeat writes are idempotent. PostgreSQL is integration-tested. Concurrent insert retries, retention, and access control remain future work. `sqlite:///path/to/evalforge.db` is an optional local alternative.

Compare previously saved runs with new gates:

```bash
evalforge compare --config examples/regression.json \
  --baseline .evalforge/EXPERIMENT_DIRECTORY/baseline.json \
  --candidate .evalforge/EXPERIMENT_DIRECTORY/candidate.json
```

This command does not execute targets or judges. It validates the saved evidence and applies the config's gates to those runs; dataset/target fields remain required by the shared experiment configuration.

## Execution limits and provider retries

Add an experiment-wide budget (shared by baseline, candidate, judges, and retries):

```json
{"execution":{"concurrency":4,"max_provider_requests":100,"max_seconds":120,"max_observed_cost":0.50}}
```

All limits are optional; concurrency defaults to 1. Request reservations are thread-safe and count every built-in provider HTTP attempt, including failures and retries. Budget exhaustion preserves full case coverage with explicit errors and exits 3. Saved execution evidence records cumulative request/retry counts, elapsed time, configured limits, and observed provider cost across the invocation.

`max_observed_cost` is a **post-response estimate threshold, not a hard billing cap**. It requires configured prices; missing/unknown usage blocks further cost-limited work. In-flight calls can overshoot it, and providers may bill failed/timed-out requests whose usage is unknown. Only the built-in ChatProvider participates automatically; custom providers must use the budget interface. Use provider-side spending controls for a hard billing cap.

Provider configuration accepts `max_attempts` (1–10, default 1), `retry_base_seconds` (default 0.5), `retry_max_seconds` (default 10), and optional `max_completion_tokens`. Transient 408/429/500/502/503/504 and transport/timeouts can retry. Authentication, malformed output, and quota/billing errors do not retry. Valid `Retry-After` delays are honored; delays above the configured maximum or available deadline stop retries. Otherwise backoff uses exponential jitter. `timeout` bounds the completion/retry scheduling window and limits each HTTP phase to the remaining window; overdue responses are rejected. Time limits remain cooperative rather than process-level cancellation. See the [official retry guidance](https://developers.openai.com/api/docs/guides/rate-limits).

## Approved baselines and database versions

Approve a saved run once, then evaluate only candidates against it:

```bash
evalforge baseline approve --run .evalforge/EXPERIMENT_DIRECTORY/baseline.json \
  --name support-release --approved-by YOUR_NAME --note 'Reviewed policy cases'
evalforge run --config examples/regression.json --baseline-name support-release
# Or select the exact approved run:
evalforge run --config examples/regression.json --baseline-id APPROVED_RUN_ID
evalforge baseline show --name support-release
evalforge baseline history --name support-release
evalforge db status
evalforge db upgrade
```

These commands use `EVALFORGE_DATABASE_URL` or `--database-url`. Approval stores the run transactionally and appends an audit record; approving a replacement under the same name preserves previous approvals and runs. `--approved-by` is an audit label, not an authentication mechanism. Execution errors, unknown judgments, absence of scored evidence, and failed/unscored critical cases prevent approval. Candidate-only execution rejects dataset/evaluator mismatches before making target calls. It still requires the shared experiment configuration's baseline fields, but does not execute them.

Migrations initialize new databases and preserve existing V1 evidence. Partial or incompatible unversioned schemas and unknown revisions fail instead of being silently stamped. PostgreSQL migrations serialize with an advisory transaction lock. Destructive downgrades are unsupported; back up persistent databases before upgrades. `db status` only inspects the revision and exits 3 when it is not current.

## CLI exit codes

| Code | Meaning |
| --- | --- |
| 0 | Valid dataset, or quality gate passed (warnings allowed) |
| 1 | Blocking quality regression / critical failure |
| 2 | Invalid configuration, dataset content, or incompatible runs |
| 3 | Execution, missing-file, provider, evaluator, evidence, or storage error |

A run can complete with quality failures; `Run.status=ERROR` denotes execution failures. Comparison status determines the CI exit code.

## Tests and CI

```bash
pytest --cov=evalforge --cov-fail-under=90
ruff check src tests
ruff format --check src tests
```

To run the PostgreSQL integration test locally, set `EVALFORGE_TEST_DATABASE_URL` to a disposable PostgreSQL database. Without it, that test is explicitly skipped. The [GitHub Actions workflow](.github/workflows/evalforge.yml) runs Python 3.11–3.14, a PostgreSQL service, coverage/style checks, installed-package demos, and a negative regression test. Reports and JSON evidence are uploaded as artifacts and included in the job summary. The negative fixture is expected to fail with exit 1; any other exit fails the workflow. To gate your own application, replace the passing fixture command with your application's configured suite and retain its exit code.

## Architecture and boundaries

```text
JSONL + JSON config → Targets → Versioned evaluators → Saved runs
                                                        ↓
                          Weighted metrics → Paired comparison → Explicit gates
                                                        ↓
                                           CLI report + exit code + CI artifacts
```

The implementation is a `src/evalforge` Python package with separate models, datasets, targets/providers, evaluators, runner, scoring, regression, storage, configuration, reporting, and CLI modules. It owns the evaluation/regression engine rather than wrapping a third-party eval library.

Concurrency defaults to 1 and is configurable up to 32 cases; result order stays aligned with the dataset. Built-in providers support bounded transient retries; background workers remain deferred. Time budgets are cooperative: in-flight HTTP phases or local callables can finish after the deadline, and overdue results become errors. Local targets/evaluators cannot be forcibly terminated. Custom targets and evaluators must be thread-safe when concurrency exceeds 1. Latency is measured wall time around target execution, and p95 uses nearest rank. Operational averages are unweighted; quality averages use evaluator weights within each case, then case weights across cases. Skips are excluded and coverage is reported. No statistical significance claim is made by threshold gates. Artifacts contain raw inputs/outputs and should be handled as application data.

## Roadmap

[`TECHNICAL_DESIGN.md`](TECHNICAL_DESIGN.md) retains the complete long-term architecture and starts with an implementation ledger. Next stages can add API/workers, migrations, semantic/RAG evaluation, agent trajectories, dashboard, calibration, and production feedback. They have not been built in this release.

## License

See [LICENSE](LICENSE).
