# EvalForge — Complete Technical Design and Implementation Plan

> **Status:** Core, baseline reliability, initial RAG evaluation and local results dashboard implemented; broader platform remains planned.
> **Purpose:** Source of truth for the implemented core and the future architecture. The implementation ledger below takes precedence over conceptual examples in later sections.

---


# Implementation Ledger — Release 0.4.0

The requested V0/V1 vertical slice and the baseline/reliability/application-demo milestone are implemented as a Python library/CLI. The following 63 sections preserve the long-term product design; statements about services, advanced evaluators, or UI below describe **planned** functionality unless listed as implemented here.

## Implemented components

| Build phase | Implementation | Validation |
| --- | --- | --- |
| A: Domain | Pydantic eval cases, datasets, target/evaluator results, runs, experiments, regression rules and decisions | Unknown-field, finite-value, result-status, unique-ID/metric and result-matrix tests |
| B: Local engine | JSONL imports, content hashes, explicit mock fixtures, callable targets, deterministic checks, bounded concurrent runner, weighted scoring, paired comparison and terminal report | Golden evaluator cases, weighted/category regression tests, critical-case failures, missing-data/error tests |
| C: Providers | Provider protocol; OpenAI-compatible HTTP chat target; structured judge with rubric, version, provider identity, confidence/rationale/evidence and usage capture | Controlled HTTP/structured-response tests; no live paid model calls validated |
| D: Persistence | Atomic JSON artifacts; SQLAlchemy datasets/runs/experiments tables; PostgreSQL JSONB snapshots with immutable IDs/version labels and transactions | SQLite and real PostgreSQL round trips, idempotence, conflicts and rollback |
| F: CLI/CI | Strict JSON config; validate/run/compare commands; exit codes 0/1/2/3; GitHub Actions with PostgreSQL, Python 3.11–3.14, coverage/style checks and installed-package regression demos | CLI pass/fail/error/config tests and 24-case offline demos; hosted CI results are recorded in the PR |

Phase G now has an initial RAG slice, detailed below. Phase E (FastAPI/queue/workers), advanced phase H UI and phases I–J (agents/calibration), semantic similarity, embeddings, pairwise judging, production traces and adaptive generation are deferred.

## Reliability and application-demo extension

Release 0.2.0 adds approved baseline selection by name/run ID and immutable approval history; Alembic initial-schema/adoption and baseline revisions; bounded concurrency; retryable-provider classification with Retry-After handling; shared request/time/observed-cost limits; local target keyword parameters; and prompt-file contents/hashes.

The independent ForgeDesk demo (`src/evalforge/demos`, `examples/support`) implements an actual rule-based FAQ application and provider-backed prompt configurations. It evaluates eight authored cases, validates a seed pair, approves a saved baseline, runs only the candidate, and verifies the expected refund-category regression. The wrapper returns success only if the expected blocked regression is established. The offline application does not inspect case IDs, reference answers, or expected facts. The live version uses policy prompts, deterministic checks and a provider-backed judge; its complete 48-successful-call path is tested with controlled HTTP responses. No live paid validation is claimed because no credential was configured. Each live phase permits at most 40 provider attempts, including retries, with a 300-token output cap and cooperative 300-second deadline; this is not a dollar billing cap.

The GitHub Actions workflow runs the offline approved-baseline application demo in addition to the original fixture gates. Hosted dashboard, advanced RAG/agent features and API/worker services remain deferred.

## Initial RAG slice (0.3.0)

`rag.py` implements a frozen UTF-8 Markdown corpus snapshot, heading-aware word-bounded
chunks, content-derived chunk IDs, and in-memory BM25 search (positive log(1+RSJ odds),
k1=1.5, b=0.75, unique query tokens, deterministic ties). Corpus SHA-256 includes document
paths/content and chunk configuration. No embeddings, vector database or reranker exists yet.
The `rag` target reads only `case.input`, retrieves passages, and calls the configured chat
provider with source IDs. It has no offline answer fallback. Typed optional `TargetResult.retrieval`
captures query, corpus hash, retriever, requested K, ranked passage text/scores and parsed
citations. Provider failures retain retrieval evidence. Old non-RAG snapshots remain readable.

Document-level recall/precision use `metadata.relevant_document_ids` and the first K ranked
passages, deduplicating document hits. Recall divides by labeled relevant document count;
precision divides by K (missing slots count as misses). Missing labels skip; malformed labels,
missing/mismatched trace or insufficient configured retrieval depth error. Citation validity checks
source membership, not claim support. Faithfulness uses a structured provider judge with only
question, answer and retrieved evidence; reference answers are excluded. Judge scores remain
uncalibrated model judgments. Reports preserve retrieved evidence for failed cases.

`examples/rag` contains independent policy documents and three labeled cases, a complete
baseline corpus and a candidate missing the refund policy. The retrieval-only command makes
no model calls; the answer/evaluation modes require a provider. Controlled HTTP tests verify
label isolation and a missing-document regression. No live paid AI validation is claimed.

## Local results dashboard (0.3.0)

`evalforge dashboard --open` serves a packaged HTML/CSS/JavaScript results explorer through
Python's standard-library HTTP server, bound only to 127.0.0.1. The original 0.3.0 artifact viewer is extended
by the 0.4.0 local execution milestone below; the future FastAPI/control-plane/worker architecture remains planned. It lists validated
`experiment.json` snapshots under a configured artifact root, newest first, and shows gates,
baseline/candidate case answers, evaluator explanations and RAG passage/citation evidence.
The client uses text DOM rendering for untrusted inputs/outputs, never HTML interpolation.
Local Host/Origin validation, same-origin assets and CSP keep its scope local.
UUID-only experiment routes, symlink/path guards and a 16-MiB per-artifact read bound prevent
arbitrary file browsing. Invalid folders are counted/skipped. Listing scans all snapshots;
SQL querying, pagination, multiuser authentication and approvals in the UI are deferred.
Tests cover real loopback HTTP routes, invalid artifacts, path/origin guards and CLI dispatch.

## Browser execution and local jobs (0.4.0)

The dashboard now accepts repeated `--config PATH` flags to register trusted configurations.
With none registered it remains read-only. Setup validates datasets, target construction,
evaluator options, placeholder model IDs, local credential availability and model budgets.
Missing credentials are described by environment-variable name without returning secrets.
Registered local callables are trusted Python code; setup can import their modules.

`POST /api/jobs` accepts only a registered profile ID and a model-call confirmation boolean.
Host/Origin checks and a random session token protect submission. Model jobs require request,
time and output-token limits plus explicit confirmation; these are not monetary spending caps.
A SQLite transaction gives one execution dashboard ownership of each artifact root. A thread
supervises one separate CLI process at a time, while HTTP polling reports queued, running,
completed, error or interrupted status. A completed quality FAIL remains a completed job;
execution errors retain ERROR artifacts when available. Valid results move into normal UUID
artifact directories and the UI opens the comparison.

`_jobs/UUID/` stores atomic `job.json` state, a frozen `config.json` with absolute file references,
and `execution.log`. The source-config digest is captured at submission; file inputs and local
modules are still read at execution, with their actual versions recorded in experiment evidence.
Browser workers remove inherited `EVALFORGE_DATABASE_URL` and write artifacts only. Process
execution is bounded to min(configured max_seconds or 300, 3600) + 15 seconds. Normal server
shutdown terminates the owned process. Startup marks unfinished jobs interrupted without
retrying. An unexpected parent crash may leave a child process alive, so operators must check
processes and artifacts before retrying. There is no durable queue, cancellation endpoint,
per-question progress stream or job-history UI. These local jobs do not implement the planned
FastAPI/distributed worker architecture.

Tests exercise real CLI jobs and loopback submission, origin/token guards, busy rejection,
worker deadlines, execution-error artifacts and restart/ownership behavior. Live paid model
validation remains outstanding; controlled provider tests do not establish model quality.

## Model-free RAG source preview (0.4.0 follow-up)

Registered configurations with two valid RAG targets expose a **Preview sources** workflow.
`POST /api/retrieval` accepts only a registered profile ID and a nonempty question (maximum
2,000 characters), under the same loopback Host/Origin and session-token checks as job submission.
The existing 4-KiB UTF-8 JSON request-body limit also applies. It constructs BM25 retrievers from
both trusted configuration document paths and returns typed retrieval traces with collection names
and document counts. It does not construct a model provider, load evaluation cases/labels, invoke
a judge, submit a job or persist an experiment. Placeholder model IDs and missing keys do not
block previews. Invalid RAG files/target types are rejected without exposing arbitrary paths.

Each request rebuilds corpus snapshots; edits appear on the next search. The browser renders
passages as text, including zero-match guidance. Search scores are disclosed as lexical rankings,
not confidence or comparable cross-corpus quality scores. Stale responses are discarded when the
selected configuration changes or setup refreshes. This is an inspection tool, not an offline AI
answer generator or a quality-gate result. Tests use real Markdown corpora, assert providers and
evaluation labels are untouched, and cover edited/missing documents and real HTTP access guards.

## Concrete repository structure

```text
src/evalforge/
  models.py       validated domain contracts
  datasets.py     strict JSONL + canonical SHA-256 versioning
  providers.py    provider protocol + chat HTTP adapter
  targets.py      mock / local callable / raw model targets
  evaluators.py   deterministic evaluators + structured LLM judge
  runner.py       sequential execution + error separation
  scoring.py      quality and operational aggregation
  regression.py   paired compatibility + explicit gates
  storage.py      atomic JSON + immutable SQL snapshots
  config.py       validated JSON configuration + component construction
  report.py       terminal metrics and failure evidence
  cli.py          validate / run / compare / baseline / db and exit codes
  budgets.py      shared cooperative budgets and request reservations
  migrations/     versioned core/adoption and approval revisions
  demos/          rule-based/provider-backed support application and workflow
  __main__.py     python -m evalforge
 tests/           pytest unit, integration, CLI and PostgreSQL coverage
 evals/support.jsonl
 examples/passing.json
 examples/regression.json
 examples/chat-judge.json
 .github/workflows/evalforge.yml
```

Using one installable package instead of early backend/frontend/CLI services keeps the domain and engine reusable without unimplemented service scaffolding. Later API/workers can import the same modules.

## Dataset and evidence contracts

EvalCase currently supports `id`, string/object `input`, `reference_answer`, `expected_facts`, `forbidden_facts`, `expected_schema`, `category`, `tags`, `critical`, positive `weight`, and JSON `metadata`. Conversation/retrieval/tool-specific fields are deferred. JSONL parsing rejects duplicate keys/IDs, non-finite constants, extra fields and incorrectly typed flags. Empty lines are ignored; an empty suite is invalid.

The dataset fingerprint is SHA-256 over canonical JSON of **all case fields**, sorted by case ID with sorted object keys. Version labels can be supplied separately; comparisons require matching name, label and hash. Runs keep copies of every case and evaluator specification, target configuration, IDs, creation timestamp, raw target results and evaluator decisions. The runner passes copies to plugins so accidental mutation cannot change recorded dataset evidence. Local callable module hashes are captured when available; complete environment/dependency reproducibility remains future work.

Each case has exactly one result per configured metric. PASS/FAIL results must have a finite score in [0,1]; ERROR/SKIPPED/UNKNOWN results have no score. Target errors skip evaluation, evaluator exceptions are recorded independently, and arbitrary exception text is sanitized. A run with execution errors has ERROR status; ordinary quality failures remain COMPLETED. Comparison status drives CLI exit behavior.

## Evaluation and provider semantics

Implemented deterministic kinds: exact match (optional trimming/case folding), required/forbidden literal substrings, full/search regex, Draft 2020-12 JSON Schema, and absolute numeric tolerance. No expected evidence means SKIPPED, not PASS. JSON outputs use strict JSON parsing; external schema references are not fetched. Substring checks are lexical, not semantic correctness checks.

LLMJudge calls a Provider protocol with a rubric system message and a JSON evidence user message. It requests JSON object mode and validates a score, optional confidence, reason and optional evidence list. Malformed or unavailable judgments become evaluator errors. Judge prompt/spec versions, provider configuration and separate judge token/cost metadata are persisted. Rubric instructions identify case/output data as untrusted; this is not a guarantee against judge prompt injection or bias. Human calibration remains planned.

ChatProvider implements synchronous `/chat/completions`, configurable endpoint/model/environment credential name, timeout, temperature, output-token limit and bounded retry settings. It rejects incomplete/non-text/malformed completions and sanitizes transport/status errors. Endpoints/models must support these options and JSON object mode for judges. Native provider SDKs, streaming, Responses API, tool use and model-specific parameter negotiation are not implemented. Transient status/transport failures can retry with exponential jitter; valid Retry-After hints are respected and excessive hints/deadlines stop retries. Quota/billing/authentication errors and malformed completions do not retry. Mock targets only return explicitly supplied fixture outputs; they never silently copy the expected answer. Tests use controlled providers/transports; live paid API validation remains outstanding.

Execution is bounded by a configurable thread pool (1–32 cases), preserving input order and keeping at most that many futures in flight. Plugins must be thread-safe. One budget can span baseline/candidate and all built-in provider/judge/retry attempts. Request reservations enforce an exact invocation-wide attempt ceiling. Elapsed-time and observed-cost limits are cooperative: no arbitrary Python thread is killed; in-flight work drains, overdue/budgeted results become explicit errors, and unused cases retain error evidence. The observed-cost threshold is not a hard billing cap and may be overshot by in-flight calls or billed failures with unknown usage. Unknown cost blocks cost-limited continuation. Custom providers must integrate the budget context themselves. Run execution summaries record cumulative shared counters. New optional Run fields normalize defaults when comparing historical SQL payloads for idempotence without rewriting those payloads.

Latency is measured around target execution, including failures. Provider usage is recorded when returned; cost is unknown unless both per-million token prices are explicitly configured and usage is available. No current model pricing is hardcoded. Judge usage is separate from target operational metrics.

## Aggregation, comparison and gates

For overall quality, scored evaluator values are averaged with evaluator weights **inside each case**, then case values are averaged with case weights. Individual metrics use case weights. SKIPPED entries are excluded, UNKNOWN/ERROR entries make that case's selected score unavailable, and coverage/error/skip counts are reported. Operational means are unweighted; p95 latency uses nearest rank. Groups currently cover all cases and categories; tag/difficulty grouping is deferred.

Paired comparisons require identical case content and dataset name/version/hash plus identical evaluator specs, versions, options, weights and captured judge-provider configuration. Case ordering and object-key ordering do not affect compatibility. Changed applicability is rejected rather than dropping measurements from one side. Target/judge errors anywhere and UNKNOWN decisions prevent passing, even with an unrelated gate.

Quality rules support `min_score`, absolute `max_regression`, fractional `max_relative_regression`, category, positive `min_samples`, critical gate and block/warn severity. Operational rules use `max_value`/`max_increase_percent`. Thresholds are inclusive; score-delta equality has a 1e-12 rounding tolerance. Zero baseline permits no increase. Critical gates fail on any candidate FAIL among selected critical-case evaluator decisions even if the baseline failed too. A critical rule with no critical cases or no scored evidence yields ERROR. Missing metrics, insufficient samples, unavailable values and partial cost coverage yield ERROR. Warnings permit quality failure but cannot bypass evidence errors. Thresholds do not estimate statistical significance.

## Persistence and CLI contract

Each invocation creates a fresh UUID artifact directory: baseline/candidate/comparison/experiment JSON, aggregate metrics JSON and a text report. JSON writes use a temporary file, fsync and replace. Artifacts remain available if the optional SQL write fails; an invocation can leave partial artifacts after an interruption/storage failure. The CLI prints exit 3 for storage failures.

SQLStore uses `evalforge_datasets` with composite name/version key and content hash, `evalforge_runs` with dataset foreign keys, and `evalforge_experiments` with run foreign keys. Full validated payloads are stored as JSONB on PostgreSQL (JSON on SQLite), preserving evidence instead of prematurely normalizing an unfinished schema. Experiment plus both runs are inserted in one transaction. IDs/version labels cannot overwrite differing payloads; identical repeat writes are idempotent. Packaged Alembic revisions initialize tables and adopt validated existing V1 schemas; partial/mismatched schemas are rejected, and PostgreSQL migrations take an advisory transaction lock. A second revision adds append-only baseline approvals with named/latest and exact run-ID selection, approver/note/timestamp audit fields, and immutable historical runs. Approval rejects execution errors, unknown judgments, lack of scored evidence and failed/unscored critical cases. Normalized entity tables from section 31, query/report service, concurrent insert retries, retention and RBAC remain future work.

`validate` checks JSONL; `run` builds baseline/candidate targets and evaluators, executes and persists; `compare` loads saved runs and applies config gates without executing targets. Configuration is JSON, rather than the conceptual YAML in section 29. Config paths are relative to the config file; local callables must be importable `module:function`. The shared config still requires target/dataset fields for `compare`; they are not used to execute anything. `EVALFORGE_DATABASE_URL` or `--database-url` selects optional SQL storage.

CLI extensions: `baseline approve/show/history`, candidate-only `run --baseline-name/--baseline-id`, and `db status/upgrade`. Named baseline resolution uses the latest audit sequence; exact run-ID selection requires a prior approval. Compatibility is checked before executing the candidate. Approver strings are audit labels, not authentication. Destructive database downgrades are unsupported.

Exit codes: 0 pass/valid, 1 blocking regression, 2 invalid content/config/incompatible runs, 3 execution/provider/evaluator/evidence/missing-file/storage error. Missing files are operational errors. Secrets are read from environment variables; raw eval data is stored and requires appropriate handling.

## CI and release boundary

The workflow installs the built package, runs tests/style/coverage (minimum 90%), provides a PostgreSQL service, executes a passing 24-case fixture, and asserts the injected refund regression exits **exactly 1**. Any other result fails CI. JSON reports and failure evidence are uploaded and published in the job summary. This is an offline engine regression gate; consumers must configure their own application target/suite for an application release gate. Fixture policies are fictional and explicitly authored, not a live RAG/LLM demonstration.

V0/V1 core acceptance is covered: 20+ cases, two target configurations, deterministic and judge/provider boundaries, case/aggregate metrics, persistence, configured regressions and meaningful nonzero exit codes. Live model behavior and statistical quality validation are not claimed. The full portfolio definition in section 61 is **not complete**, and the broader service and advanced evaluation work remains planned.

---

# 1. Executive Summary

EvalForge is a platform for **continuous evaluation, regression testing, and quality gating of LLM applications, RAG systems, and AI agents**.

The core problem is that AI systems are probabilistic. A normal application can often be validated with deterministic unit tests. An AI application can produce many different valid outputs, and an output can sound convincing while being wrong, ungrounded, policy-violating, unnecessarily expensive, or produced through an incorrect agent trajectory.

EvalForge is designed to provide a repeatable engineering workflow for answering:

> **Did this AI system actually improve, what regressed, why did it regress, and should this version be allowed to ship?**

The platform will let developers:

1. create versioned evaluation datasets,
2. register or call different AI-system versions,
3. run the same cases across candidate and baseline versions,
4. capture outputs and traces,
5. score results using multiple evaluator types,
6. aggregate metrics by task and category,
7. detect regressions,
8. inspect individual failures,
9. enforce quality thresholds in CI/CD,
10. collect human labels to calibrate evaluators,
11. turn real production failures into future regression tests.

The first implementation will deliberately focus on one complete end-to-end path rather than trying to build the full long-term platform immediately.

---

# 2. Problem Statement

## 2.1 Why AI applications are difficult to test

Traditional software frequently has deterministic expectations.

Example:

```text
input: 2 + 2
expected output: 4
```

LLM applications are different.

For the prompt:

```text
Explain why the payment failed.
```

many different responses may all be valid. Exact-string comparison is therefore insufficient.

At the same time, fluency is not correctness. An LLM can return a polished response that:

- contains an invented fact,
- ignores provided context,
- cites the wrong document,
- violates a policy,
- fails to call a required tool,
- calls the wrong tool,
- uses correct tools in the wrong order,
- produces the correct final answer through an unsafe trajectory,
- costs substantially more than the previous version,
- becomes much slower,
- improves the average score while becoming much worse on one critical category.

The system still “runs,” but its quality may have degraded.

## 2.2 What causes regressions

AI teams frequently change:

- system prompts,
- user-prompt templates,
- model provider,
- model family,
- model version,
- temperature or sampling settings,
- tool descriptions,
- tool schemas,
- orchestration logic,
- retry behavior,
- retrieval method,
- embedding model,
- chunk size,
- chunk overlap,
- reranker,
- top-k settings,
- metadata filters,
- context-window strategy,
- safety policies,
- output schemas,
- memory logic,
- agent-planning strategy.

Any of these can change behavior.

## 2.3 The central engineering question

EvalForge exists to answer:

> **Can this candidate version be shipped without introducing unacceptable quality regressions?**

That requires more than one model-generated score. It requires repeatable datasets, traces, multiple evaluator types, baselines, category-aware comparisons, human calibration, and CI integration.

---

# 3. Product Goals

EvalForge should eventually provide all of the following.

## 3.1 Primary goals

### G1 — Reproducible evaluation

The same dataset and configuration should be runnable repeatedly against different system versions.

### G2 — Multiple evaluator types

Different failures require different evaluators.

EvalForge should support deterministic checks, semantic evaluators, model-based judges, RAG metrics, agent metrics, and operational metrics.

### G3 — Baseline vs. candidate comparison

Every important experiment should be comparable against a selected baseline.

### G4 — Regression detection

The system should detect where candidate quality decreases beyond configured tolerance.

### G5 — Debuggability

A score without an explanation is not enough.

A developer should be able to inspect:

- input,
- expected behavior,
- retrieved context,
- tool calls,
- model output,
- evaluator result,
- judge rationale,
- baseline output,
- candidate output,
- metric deltas.

### G6 — CI/CD integration

Evaluation should be runnable automatically for pull requests and releases.

### G7 — RAG-specific evaluation

EvalForge should separate retrieval failure from generation failure.

### G8 — Agent-specific evaluation

EvalForge should evaluate both final task success and the sequence of actions taken.

### G9 — Judge calibration

Automated evaluators should themselves be measurable against human labels.

### G10 — Continuous eval-set improvement

Real failures should be convertible into future regression tests.

---

# 4. Non-Goals

Defining non-goals prevents the project from becoming an unfocused “AI platform.”

EvalForge is **not initially intended to be**:

- a model-training platform,
- a fine-tuning platform,
- a general prompt IDE,
- a full production observability vendor,
- a chat application,
- a generic vector database,
- a model gateway,
- a replacement for application-level unit tests,
- a benchmark leaderboard for foundation models,
- a fully autonomous red-team platform,
- a replacement for human evaluation.

Some of these may integrate with EvalForge later, but they are not the initial product.

---

# 5. Target Users

## 5.1 AI application engineers

Need to validate prompt/model changes before shipping.

## 5.2 RAG engineers

Need to understand whether a failure came from retrieval or generation.

## 5.3 Agent engineers

Need to validate tool selection, tool arguments, trajectory quality, and task completion.

## 5.4 ML / applied-AI engineers

Need repeatable experiments and comparable metrics.

## 5.5 QA / reliability engineers

Need regression suites and release gates for probabilistic systems.

## 5.6 Small AI teams and startups

Need lightweight eval infrastructure without building an internal platform from scratch.

---

# 6. Core Use Cases

## UC1 — Prompt regression testing

Compare Prompt V1 and Prompt V2 against the same model and dataset.

## UC2 — Model migration

Compare two model providers or two model versions using the same application behavior.

## UC3 — RAG pipeline comparison

Compare retrieval configurations such as:

```text
Embedding A + top_k=5 + no reranker
vs.
Embedding B + top_k=10 + reranker
```

## UC4 — Agent workflow change

Determine whether a new planner or tool description improves task completion without increasing bad tool calls.

## UC5 — Pull-request quality gate

Run a reduced eval suite for every PR and fail CI when critical thresholds regress.

## UC6 — Release evaluation

Run a larger suite before deployment.

## UC7 — Human judge calibration

Compare evaluator decisions with human-reviewed labels.

## UC8 — Production failure replay

Turn a real failure into a reproducible offline test.

---

# 7. Core Product Principles

## 7.1 Prefer deterministic evaluation where possible

If a JSON schema can prove that an output is invalid, do not spend an LLM call asking a judge whether it is valid.

## 7.2 Use model judges only where judgment is genuinely needed

Examples:

- relevance,
- completeness,
- nuanced correctness,
- pairwise preference,
- instruction following.

## 7.3 Keep raw evidence

Store the response, trace, context, metrics, evaluator result, and configuration needed to understand a run.

## 7.4 Separate quality dimensions

Do not hide everything behind a single score.

A system may improve relevance while hurting groundedness.

## 7.5 Make regression rules explicit

A CI failure should be explainable.

## 7.6 Treat evaluators as systems that can fail

Judge quality must be measurable.

## 7.7 Optimize for paired comparisons

Whenever possible, compare baseline and candidate on exactly the same test cases.

---

# 8. Terminology

## Eval Case

One test example.

## Eval Dataset

A versioned collection of eval cases.

## Target

The AI system being evaluated.

Examples:

- raw LLM call,
- RAG endpoint,
- agent endpoint.

## Run

Execution of one target configuration across an eval dataset.

## Experiment

Logical grouping of one or more runs intended for comparison.

## Baseline

Approved reference run or configuration.

## Candidate

New version being evaluated.

## Evaluator

A component that scores one or more aspects of an output or trace.

## Metric

A numeric or categorical result emitted by an evaluator.

## Regression

A candidate metric that becomes worse than the allowed threshold relative to baseline.

## Trace

Structured record of execution such as retrieval, model calls, and tool calls.

## Quality Gate

Rule that determines whether a run should pass or fail CI.

---

# 9. End-to-End Workflow

The complete conceptual flow is:

```text
1. Developer defines Eval Dataset
                    │
                    ▼
2. Developer defines Target Configurations
   ├── baseline
   └── candidate
                    │
                    ▼
3. Experiment created
                    │
                    ▼
4. Runner executes each case
                    │
                    ▼
5. Trace collector records
   ├── prompt
   ├── retrieved context
   ├── model response
   ├── tool calls
   ├── tokens
   ├── latency
   └── errors
                    │
                    ▼
6. Evaluator pipeline runs
   ├── deterministic
   ├── semantic
   ├── LLM judge
   ├── RAG
   ├── agent
   └── operational
                    │
                    ▼
7. Metric aggregation
                    │
                    ▼
8. Baseline vs candidate comparison
                    │
                    ▼
9. Regression engine
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
10A. Dashboard              10B. CI Gate
     failure analysis            pass/fail
                    │
                    ▼
11. Human review where needed
                    │
                    ▼
12. Approved failures can become new eval cases
```

---

# 10. High-Level Architecture

```text
                          ┌──────────────────────┐
                          │      Web UI / CLI    │
                          └──────────┬───────────┘
                                     │
                                     ▼
                          ┌──────────────────────┐
                          │      FastAPI API     │
                          │     Control Plane    │
                          └──────────┬───────────┘
                                     │
             ┌───────────────────────┼──────────────────────┐
             │                       │                      │
             ▼                       ▼                      ▼
     ┌───────────────┐      ┌────────────────┐     ┌────────────────┐
     │ Dataset       │      │ Experiment     │     │ Configuration  │
     │ Service       │      │ Service        │     │ Registry       │
     └───────┬───────┘      └───────┬────────┘     └────────────────┘
             │                      │
             └───────────┬──────────┘
                         ▼
                ┌───────────────────┐
                │ Experiment       │
                │ Orchestrator     │
                └─────────┬─────────┘
                          │
                          ▼
                ┌───────────────────┐
                │ Job Queue /      │
                │ Worker Pool      │
                └─────────┬─────────┘
                          │
                          ▼
                ┌───────────────────┐
                │ Target Adapters   │
                │ LLM / RAG / Agent│
                └─────────┬─────────┘
                          │
                          ▼
                ┌───────────────────┐
                │ Trace Collector   │
                └─────────┬─────────┘
                          │
                          ▼
                ┌───────────────────┐
                │ Evaluator Pipeline│
                └─────────┬─────────┘
                          │
                          ▼
                ┌───────────────────┐
                │ Score Aggregator  │
                └─────────┬─────────┘
                          │
                          ▼
                ┌───────────────────┐
                │ Regression Engine │
                └──────┬───────┬────┘
                       │       │
                       ▼       ▼
                ┌──────────┐ ┌──────────────┐
                │Dashboard │ │GitHub Actions│
                └──────────┘ │Quality Gate  │
                             └──────────────┘

Persistence:
PostgreSQL stores datasets, configurations, experiments, cases, results,
metrics, labels, regression outcomes, and metadata.

Redis / queue infrastructure coordinates asynchronous work.
```

---

# 11. Major Components

# 11.1 Control Plane API

The API manages:

- datasets,
- dataset versions,
- target configurations,
- experiments,
- runs,
- evaluator configurations,
- quality gates,
- human labels,
- reports.

The API should not directly execute long-running eval jobs synchronously.

Responsibilities:

- validate requests,
- create database records,
- enqueue work,
- expose status,
- return results,
- enforce permissions later.

Proposed implementation:

- Python
- FastAPI
- Pydantic
- SQLAlchemy

---

# 11.2 Dataset Service

The dataset service manages eval cases and dataset versions.

Required capabilities:

- create dataset,
- add cases,
- edit metadata,
- create immutable version snapshots,
- tag cases by category,
- import JSONL,
- export JSONL,
- mark critical cases,
- link production incidents to cases.

Important design principle:

Once an experiment runs against dataset version X, version X should remain reproducible.

Editing the dataset should create a new logical version rather than silently changing historical experiments.

---

# 11.3 Configuration Registry

A target configuration describes what is being tested.

Example dimensions:

```text
model_provider
model_name
temperature
system_prompt_version
prompt_template_version
retriever_version
embedding_model
top_k
reranker
agent_version
tool_schema_version
application_commit_sha
```

This configuration must be stored with each run.

Without configuration capture, experiment results are not reproducible.

---

# 11.4 Experiment Orchestrator

The orchestrator coordinates experiment execution.

Responsibilities:

1. resolve dataset version,
2. resolve target configuration,
3. determine evaluator set,
4. create run records,
5. fan out case jobs,
6. monitor completion,
7. trigger evaluators,
8. aggregate metrics,
9. compare against baseline,
10. invoke regression rules,
11. finalize run status.

Initial implementation can be simpler than a fully distributed scheduler.

---

# 11.5 Worker Layer

Eval execution will be asynchronous because model calls may take seconds and an experiment may contain hundreds or thousands of cases.

Workers handle:

- target execution,
- judge calls,
- embedding calls,
- RAG metric computation,
- retryable provider failures.

Proposed early stack:

- Redis
- Celery or a lightweight queue abstraction

Important requirement:

Worker jobs should be idempotent where practical.

A retry should not create duplicate logical results.

---

# 11.6 Target Adapters

EvalForge should not assume every target is just a raw LLM API call.

The target abstraction should support:

### Raw model target

Input → model → output

### HTTP application target

Input → external endpoint → output

### RAG target

Input → retrieval → generation → output + retrieval trace

### Agent target

Input → planner/tool loop → final output + trajectory

A common internal result shape allows downstream evaluators to work consistently.

---

# 11.7 Trace Collector

A trace is the evidence needed to understand an execution.

For an LLM target:

- normalized input,
- system prompt identifier,
- model,
- response,
- token usage,
- latency,
- provider metadata.

For RAG:

- query,
- transformed query if any,
- retrieved chunk IDs,
- retrieval scores,
- reranked order,
- provided context,
- answer.

For agents:

- task input,
- reasoning-safe event metadata,
- tool names,
- tool arguments,
- tool outputs,
- retries,
- errors,
- final response.

EvalForge should store operational traces without relying on hidden model chain-of-thought.

---

# 11.8 Evaluator Pipeline

Each case can run through one or more evaluators.

The evaluator interface should conceptually accept:

```text
EvalCase
TargetResult
Optional BaselineResult
EvaluatorConfig
```

and return:

```text
metric_name
value
pass/fail/unknown
confidence
explanation
metadata
evaluator_version
```

Evaluators should be individually versioned.

If the judge prompt changes, historical results should still record which judge version produced them.

---

# 11.9 Score Aggregator

Individual results need aggregation across:

- entire dataset,
- category,
- tag,
- difficulty,
- evaluator,
- criticality.

Example:

```text
Overall groundedness: 0.92
Billing groundedness: 0.97
Refund-policy groundedness: 0.74
Critical cases passed: 19/20
```

This prevents strong performance in a large easy category from hiding failures in a small critical category.

---

# 11.10 Regression Engine

The regression engine compares candidate metrics against baseline metrics and configured rules.

It should support:

- absolute thresholds,
- relative degradation thresholds,
- category-specific thresholds,
- critical-case zero-tolerance rules,
- latency thresholds,
- cost thresholds,
- minimum sample requirements.

Example:

```yaml
gates:
  - metric: correctness
    min_score: 0.90

  - metric: groundedness
    max_regression: 0.03

  - metric: refund_policy_accuracy
    max_regression: 0.00
    critical: true

  - metric: p95_latency_ms
    max_increase_percent: 25

  - metric: estimated_cost_per_case
    max_increase_percent: 40
```

---

# 12. Evaluation Dataset Design

A strong eval system depends heavily on dataset quality.

## 12.1 Dataset contents

An eval case may contain:

- unique ID,
- input,
- conversation history,
- expected answer,
- expected facts,
- forbidden facts,
- reference documents,
- expected citations,
- expected tools,
- forbidden tools,
- expected structured schema,
- tags,
- category,
- priority,
- difficulty,
- critical flag,
- source,
- notes.

## 12.2 Example conceptual case

```json
{
  "id": "refund_014",
  "category": "billing",
  "tags": ["refund", "policy", "critical"],
  "input": "Can I get a refund 30 days after purchase?",
  "reference_answer": "Refunds are available only within 14 days.",
  "expected_facts": [
    "refund window is 14 days"
  ],
  "forbidden_facts": [
    "refund window is 30 days"
  ],
  "critical": true
}
```

## 12.3 Dataset sources

Cases can come from:

- handcrafted scenarios,
- known bugs,
- support transcripts after sanitization,
- production failures,
- synthetic generation,
- domain-expert examples,
- edge-case brainstorming,
- adversarial tests.

Synthetic generation should not replace human review for critical cases.

## 12.4 Dataset splitting

Useful logical groups:

- smoke suite,
- pull-request suite,
- release suite,
- critical-policy suite,
- adversarial suite,
- production-replay suite.

This lets teams balance evaluation cost and coverage.

---

# 13. Deterministic Evaluators

Deterministic checks should be preferred whenever the expected condition can be explicitly verified.

Examples:

## 13.1 Exact match

Useful for known short answers.

## 13.2 Contains / excludes

Check required or forbidden strings.

## 13.3 Regular expression

Validate formats.

Examples:

- order number,
- date,
- identifier,
- structured field.

## 13.4 JSON-schema validation

Verify machine-readable outputs.

## 13.5 Required fields

Confirm keys exist.

## 13.6 Numeric tolerance

Example:

```text
expected amount = 12.50
allowed tolerance = 0.01
```

## 13.7 Citation presence

Confirm required citations exist.

## 13.8 Tool-call checks

Validate:

- tool selected,
- tool not selected,
- argument values,
- number of calls.

Advantages:

- cheap,
- fast,
- reproducible,
- explainable.

---

# 14. Semantic Evaluators

Semantic evaluators are useful when wording differs but meaning should remain similar.

Potential uses:

- semantic similarity to reference answer,
- clustering failures,
- near-duplicate detection,
- retrieval relevance.

Important limitation:

High embedding similarity does not guarantee factual correctness.

Therefore semantic similarity should generally be one signal, not the sole correctness evaluator.

---

# 15. LLM-as-a-Judge

Some properties require flexible language understanding.

Planned judge dimensions:

- correctness,
- relevance,
- completeness,
- groundedness,
- instruction following,
- tone or policy adherence,
- pairwise preference.

## 15.1 Structured judge output

The judge should return structured fields rather than unstructured prose.

Conceptually:

```json
{
  "score": 0.8,
  "label": "pass",
  "confidence": 0.86,
  "reason": "The answer correctly states the 14-day policy.",
  "evidence": ["Refunds are available within 14 days."]
}
```

## 15.2 Judge prompt versioning

Every judge prompt must have a version identifier.

## 15.3 Bias controls

Potential judge failure modes include:

- verbosity preference,
- ordering bias,
- self-preference,
- stylistic bias,
- inconsistent grading.

Mitigations:

- concise rubrics,
- structured scoring criteria,
- randomized A/B ordering for pairwise tests,
- calibration against humans,
- repeated evaluation for important cases where justified,
- deterministic checks before judge calls.

## 15.4 Judge confidence

Judge confidence should be treated cautiously.

A self-reported confidence value is not automatically calibrated.

The more meaningful measure is empirical agreement with human labels.

---

# 16. Pairwise Evaluation

Instead of independently scoring two answers, a judge can compare:

```text
Response A
vs.
Response B
```

Possible outcomes:

- A better,
- B better,
- tie,
- invalid comparison.

Pairwise evaluation is useful when:

- comparing prompts,
- comparing models,
- comparing answer style,
- evaluating nuanced quality.

Ordering should be randomized to reduce position bias.

---

# 17. RAG Evaluation

RAG evaluation needs to distinguish retrieval quality from generation quality.

## 17.1 Retrieval metrics

Potential metrics:

- Recall@K,
- Precision@K,
- Hit Rate,
- Mean Reciprocal Rank,
- context relevance,
- relevant-chunk coverage.

## 17.2 Generation metrics

Potential metrics:

- answer correctness,
- groundedness,
- faithfulness to context,
- citation correctness,
- completeness.

## 17.3 Failure classification

EvalForge should classify failures conceptually as:

### Retrieval failure

Relevant evidence was not retrieved.

### Ranking failure

Relevant evidence was retrieved but ranked too low or excluded from final context.

### Generation failure

Correct evidence was present, but the model answered incorrectly.

### Citation failure

Answer may be correct but references the wrong source.

This classification is much more actionable than one generic “RAG score.”

## 17.4 RAG trace requirements

Capture:

- source document ID,
- chunk ID,
- retrieval score,
- rank,
- reranker score,
- final context inclusion,
- citations returned.

---

# 18. Agent Evaluation

Agent evaluation is one of the strongest long-term differentiators.

A final answer alone is not enough.

An agent may reach the right answer while:

- calling a forbidden tool,
- using the wrong account,
- making unnecessary calls,
- retrying excessively,
- providing incorrect arguments,
- performing actions in the wrong order.

## 18.1 Agent evaluation dimensions

### Task completion

Did the task succeed?

### Tool selection

Was the appropriate tool used?

### Tool arguments

Were arguments correct?

### Required steps

Were mandatory actions performed?

### Forbidden steps

Were disallowed actions avoided?

### Trajectory validity

Did the action sequence follow acceptable logic?

### Efficiency

Were there unnecessary steps?

### Recovery

Did the agent recover correctly from tool failure?

### Final answer

Did the user receive an accurate final response?

## 18.2 Agent trajectory model

Conceptually:

```text
User Task
  ↓
Agent Decision
  ↓
Tool Call 1
  ↓
Tool Result
  ↓
Agent Decision
  ↓
Tool Call 2
  ↓
Tool Result
  ↓
Final Response
```

The evaluator should operate on observable actions and outputs, not hidden chain-of-thought.

---

# 19. Operational Metrics

Quality must be evaluated together with operational behavior.

Metrics:

- end-to-end latency,
- p50 latency,
- p95 latency,
- p99 latency,
- prompt tokens,
- completion tokens,
- total tokens,
- estimated cost,
- provider errors,
- timeouts,
- retries.

A new model might improve answer quality 2% while increasing cost 300%.

EvalForge should make that tradeoff visible.

---

# 20. Scoring Model

The platform should avoid pretending that one universal score represents the entire system.

## 20.1 Per-case score

Each evaluator generates its own result.

Example:

```text
correctness = 1.0
groundedness = 0.7
policy = pass
latency_ms = 1800
cost_usd = 0.012
```

## 20.2 Category aggregation

Aggregate by:

- category,
- tag,
- priority,
- evaluator.

## 20.3 Optional composite score

A weighted composite may be supported for convenience.

Example:

```text
0.40 correctness
0.30 groundedness
0.20 completeness
0.10 instruction following
```

However, CI gating should still support individual critical metrics.

---

# 21. Regression Detection

Regression detection is the heart of the project.

## 21.1 Paired comparison

Baseline and candidate should run on the same case IDs.

For each case:

```text
delta = candidate_score - baseline_score
```

Then aggregate deltas.

## 21.2 Regression types

### Overall regression

Dataset-wide score falls.

### Category regression

A specific category falls.

### Critical-case regression

A critical example fails.

### Operational regression

Cost or latency exceeds tolerance.

### Behavioral regression

Tool usage or policy compliance worsens.

## 21.3 Gate examples

```text
FAIL if overall correctness < 90%

FAIL if groundedness falls more than 3 percentage points

FAIL if any critical policy test fails

FAIL if p95 latency increases by more than 25%

WARN if cost increases 20–40%

FAIL if cost increases >40%
```

## 21.4 Statistical support

Later versions can add:

- bootstrap confidence intervals,
- paired significance testing,
- minimum sample requirements,
- uncertainty estimates.

These should not be overused on tiny datasets.

---

# 22. Baseline Management

A baseline is an approved reference configuration.

Potential workflow:

```text
Candidate Run
    ↓
Passes review
    ↓
Marked Approved
    ↓
Promoted to Baseline
    ↓
Future runs compare against it
```

Store:

- baseline run ID,
- application commit SHA,
- model config,
- prompt version,
- dataset version,
- evaluator versions.

Baseline promotion should be explicit.

---

# 23. Experiment Versioning

Every experiment result should be reproducible from stored metadata.

Persist at minimum:

- experiment ID,
- run ID,
- dataset version,
- target version,
- evaluator versions,
- model configuration,
- application commit SHA,
- timestamp,
- environment,
- seed where relevant.

Historical results should not silently change if an evaluator is later updated.

---

# 24. Failure Explorer

A useful evaluation platform must make failures debuggable.

The failure detail page should show:

## Input

Original case input.

## Expected behavior

Reference answer, required facts, forbidden facts, expected tools.

## Baseline result

Output and trace.

## Candidate result

Output and trace.

## Retrieved context

Documents/chunks used.

## Tool trajectory

Tool names, arguments, results.

## Evaluator results

Per-metric scores.

## Judge explanation

Short explanation with evidence.

## Regression information

What changed from baseline.

## Human label

Optional reviewer decision.

---

# 25. Human Review and Judge Calibration

LLM judges are not ground truth.

EvalForge should support human labels.

## 25.1 Human label structure

Example:

```text
case_id
dimension
human_score
human_label
reviewer
notes
timestamp
```

## 25.2 Calibration metrics

Potential metrics:

- exact agreement,
- correlation,
- confusion matrix,
- false-positive rate,
- false-negative rate,
- Cohen's kappa for categorical labels where appropriate.

## 25.3 Calibration workflow

```text
Automated Judge
       │
       ▼
Sample Results
       │
       ▼
Human Review
       │
       ▼
Agreement Analysis
       │
       ▼
Judge Prompt / Rubric Update
       │
       ▼
New Evaluator Version
```

---

# 26. Adaptive Eval Generation

The eval suite should improve when real failures appear.

Proposed future workflow:

```text
Production Incident
        ↓
Trace captured
        ↓
Failure classified
        ↓
Candidate eval case generated
        ↓
Human review
        ↓
PII / sensitive data check
        ↓
Added to dataset
        ↓
Included in future regression runs
```

Important rule:

Automatically generated cases should normally require approval before entering critical suites.

---

# 27. Production Trace Ingestion

Later versions may ingest traces from production systems.

Use cases:

- identify recurring failures,
- replay known incidents,
- mine difficult cases,
- detect distribution shifts,
- build new eval sets.

Production data handling requires stronger privacy controls than synthetic eval data.

---

# 28. Proposed API Surface

The exact routes may evolve, but the system should conceptually expose APIs like:

## Datasets

```text
POST   /datasets
GET    /datasets
GET    /datasets/{id}
POST   /datasets/{id}/cases
POST   /datasets/{id}/versions
GET    /datasets/{id}/versions/{version}
```

## Targets

```text
POST   /targets
GET    /targets
GET    /targets/{id}
POST   /targets/{id}/versions
```

## Experiments

```text
POST   /experiments
GET    /experiments
GET    /experiments/{id}
POST   /experiments/{id}/run
```

## Runs

```text
GET    /runs/{id}
GET    /runs/{id}/cases
GET    /runs/{id}/metrics
GET    /runs/{id}/regressions
```

## Human labels

```text
POST   /cases/{result_id}/labels
GET    /cases/{result_id}/labels
```

## Baselines

```text
POST   /runs/{id}/promote-baseline
GET    /baselines
```

---

# 29. CLI Plan

The CLI should make EvalForge usable without the web UI.

Possible commands:

```bash
evalforge init
evalforge dataset validate evals.jsonl
evalforge run --config evalforge.yaml
evalforge compare <baseline-run> <candidate-run>
evalforge report <run-id>
evalforge baseline promote <run-id>
```

CI should use the same underlying APIs / engine as the CLI.

---

# 30. Configuration File

A repository-level config may look conceptually like:

```yaml
project: support-agent

dataset:
  path: evals/support.jsonl

target:
  type: http
  endpoint: http://localhost:8000/chat

evaluators:
  - name: policy_check
    type: deterministic

  - name: groundedness
    type: llm_judge
    rubric: groundedness_v1

  - name: semantic_similarity
    type: embedding

gates:
  - metric: correctness
    min_score: 0.90

  - metric: groundedness
    max_regression: 0.03

  - tag: critical
    max_failed_cases: 0

operations:
  max_p95_latency_increase_percent: 25
  max_cost_increase_percent: 40
```

The final syntax should be simple enough to review in code.

---

# 31. Database Design

PostgreSQL is the primary persistence layer.

Proposed tables:

## projects

- id
- name
- created_at

## datasets

- id
- project_id
- name
- description
- created_at

## dataset_versions

- id
- dataset_id
- version
- content_hash
- created_at

## eval_cases

- id
- dataset_version_id
- external_case_id
- input
- reference_output
- metadata
- critical

## targets

- id
- project_id
- name
- type

## target_versions

- id
- target_id
- config_json
- config_hash
- commit_sha

## evaluator_definitions

- id
- name
- type
- version
- config_json

## experiments

- id
- project_id
- dataset_version_id
- created_at

## runs

- id
- experiment_id
- target_version_id
- status
- started_at
- completed_at

## case_results

- id
- run_id
- eval_case_id
- output
- trace_json
- latency_ms
- token_usage
- estimated_cost

## evaluator_results

- id
- case_result_id
- evaluator_definition_id
- metric_name
- numeric_value
- categorical_value
- passed
- confidence
- explanation

## regression_results

- id
- candidate_run_id
- baseline_run_id
- metric_name
- scope
- baseline_value
- candidate_value
- delta
- gate_status

## human_labels

- id
- case_result_id
- dimension
- score
- label
- notes
- reviewer_id
- created_at

## baselines

- id
- project_id
- run_id
- active
- created_at

---

# 32. Storage Strategy

## PostgreSQL

Use for structured metadata and results.

## Object storage later

Large artifacts may eventually move to object storage:

- long traces,
- uploaded datasets,
- attachments,
- exported reports.

## Redis

Use for:

- queue coordination,
- short-lived job status,
- caching,
- distributed locks if needed.

Redis should not be the durable source of truth for experiment results.

---

# 33. Model Provider Abstraction

EvalForge should avoid binding core evaluation logic to one model provider.

A normalized model-call interface should expose:

```text
messages
model
temperature
structured_output_schema
timeout
metadata
```

Normalized result:

```text
text
structured_output
usage
latency
provider_request_id
error
```

A provider abstraction such as LiteLLM can reduce integration work, while EvalForge retains its own domain models.

---

# 34. Evaluator Extensibility

Evaluators should behave like plugins.

Conceptually:

```text
Evaluator
 ├── name
 ├── version
 ├── required_inputs
 ├── evaluate()
 └── output_schema
```

Example evaluator classes:

```text
ExactMatchEvaluator
RegexEvaluator
JsonSchemaEvaluator
EmbeddingSimilarityEvaluator
LLMJudgeEvaluator
PairwiseJudgeEvaluator
GroundednessEvaluator
RetrievalRecallEvaluator
ToolCallEvaluator
TrajectoryEvaluator
LatencyEvaluator
CostEvaluator
```

This design allows future custom evaluators without rewriting the experiment engine.

---

# 35. Observability

EvalForge itself should be observable.

Use OpenTelemetry-style traces for:

- API requests,
- experiment orchestration,
- model calls,
- judge calls,
- retrieval calls,
- evaluator execution,
- worker jobs.

Metrics:

- experiment duration,
- worker utilization,
- provider failure rate,
- judge-call count,
- queue depth,
- evaluation cost.

Logs should include stable IDs such as:

- experiment_id,
- run_id,
- case_id,
- evaluator_id.

---

# 36. Security and Privacy

Even as a portfolio project, the architecture should acknowledge realistic security needs.

## 36.1 Secrets

API keys must never be stored in source control.

Use environment variables or secret managers.

## 36.2 Sensitive eval data

Production-derived datasets may contain:

- PII,
- account data,
- internal documents,
- regulated information.

Future versions should support:

- redaction,
- dataset access controls,
- retention policies,
- encryption.

## 36.3 Prompt injection

RAG and agent evaluations should include prompt-injection cases.

## 36.4 Tool safety

Agent eval suites should test forbidden or high-risk tool operations.

## 36.5 Logging

Do not log secrets or raw credentials.

---

# 37. Web Dashboard

The dashboard should focus on evaluation workflows rather than chat.

## Page 1 — Projects

Shows available projects.

## Page 2 — Datasets

- versions,
- case count,
- categories,
- tags,
- critical-case count.

## Page 3 — Experiments

- target configuration,
- dataset version,
- status,
- overall metrics,
- baseline comparison.

## Page 4 — Compare Runs

Side-by-side:

```text
Metric            Baseline   Candidate   Delta
Correctness       0.88       0.92        +0.04
Groundedness      0.94       0.91        -0.03
Latency p95       2.0s       2.6s        +30%
```

## Page 5 — Failure Explorer

Individual case debugging.

## Page 6 — Judge Calibration

Human vs. automated evaluator agreement.

---

# 38. GitHub Actions Integration

A core project goal is making evals behave like automated quality tests.

Conceptual workflow:

```text
Pull Request
    ↓
Build application
    ↓
Run EvalForge PR suite
    ↓
Compare to baseline
    ↓
Apply quality gates
    ↓
Publish summary
    ↓
PASS / FAIL
```

Example output:

```text
EvalForge Quality Gate: FAILED

Dataset: support-pr-suite@v12
Baseline: run_842
Candidate: run_911

Metric                 Baseline   Candidate   Delta   Gate
Correctness             92%         94%       +2%    PASS
Groundedness            95%         94%       -1%    PASS
Refund-policy accuracy  96%         78%      -18%    FAIL
p95 latency             1.8s        2.1s      +17%   PASS

Critical failures:
- refund_014
- refund_021

Result: ❌ BLOCK MERGE
```

---

# 39. Pull Request vs. Release Suites

Running every possible eval on every commit may be expensive.

## PR suite

Small, high-signal suite.

Includes:

- critical cases,
- recent regressions,
- representative cases.

Goal:

Fast developer feedback.

## Release suite

Larger comprehensive suite.

Includes:

- broad category coverage,
- adversarial cases,
- long-running agent tasks,
- expensive judges.

Goal:

Release confidence.

---

# 40. Cost Control

Model-based evaluation can itself become expensive.

Strategies:

- deterministic checks first,
- run judges only when needed,
- smaller PR suites,
- caching identical judge requests,
- configurable evaluator selection,
- batch embedding requests,
- cost limits per experiment,
- early stop after severe gate failure where appropriate.

The dashboard should show evaluation cost separately from target application cost.

---

# 41. Reliability and Failure Handling

The runner must distinguish:

## Target failure

Application failed.

## Provider failure

Model API unavailable or rate limited.

## Evaluator failure

Judge/evaluator failed.

## Infrastructure failure

Worker or queue issue.

These should not be silently converted into low quality scores.

Possible result statuses:

```text
PASS
FAIL
ERROR
SKIPPED
UNKNOWN
```

Retries should apply only to retryable failures.

---

# 42. Testing Strategy for EvalForge Itself

EvalForge is a testing platform and therefore needs strong tests.

## Unit tests

Test:

- evaluators,
- scoring,
- threshold logic,
- config parsing,
- normalization.

## Integration tests

Test:

- API + DB,
- queue + worker,
- model adapter mocks,
- experiment lifecycle.

## Golden tests

Use fixed traces and expected evaluator outputs.

## Regression tests

Every fixed EvalForge bug becomes a test.

## End-to-end tests

Run:

```text
dataset → target → evaluators → comparison → CI result
```

with fake model providers to keep tests deterministic.

---

# 43. Proposed Repository Structure

The following is the future service-oriented structure. The implemented core uses the package layout in the implementation ledger:

```text
EvalForge/
│
├── README.md
├── TECHNICAL_DESIGN.md
├── LICENSE
├── pyproject.toml
├── docker-compose.yml
│
├── backend/
│   ├── app/
│   │   ├── api/
│   │   ├── core/
│   │   ├── db/
│   │   ├── datasets/
│   │   ├── experiments/
│   │   ├── runners/
│   │   ├── targets/
│   │   ├── evaluators/
│   │   ├── regression/
│   │   ├── traces/
│   │   └── workers/
│   └── tests/
│
├── frontend/
│   ├── app/
│   ├── components/
│   └── lib/
│
├── cli/
│
├── examples/
│   ├── support-bot/
│   ├── rag-demo/
│   └── agent-demo/
│
├── evals/
│   └── sample.jsonl
│
└── .github/
    └── workflows/
        └── evalforge.yml
```

---

# 44. Technology Choices and Rationale

## Python

Why:

- dominant ecosystem for LLM and ML tooling,
- strong async/API support,
- broad evaluation-library compatibility.

## FastAPI

Why:

- typed request/response models,
- good Python developer experience,
- async support,
- automatic API documentation.

## Pydantic

Why:

- strong schema validation,
- useful for structured LLM outputs,
- clear configuration models.

## PostgreSQL

Why:

- relational structure fits experiments and versioning,
- JSONB can store flexible metadata,
- reliable and mature.

## SQLAlchemy

Why:

- standard Python persistence abstraction,
- migrations can later use Alembic.

## Redis

Why:

- common worker coordination layer,
- fast transient state.

## Background worker queue

Why:

- evals are long-running,
- model calls should not block HTTP request lifecycles.

## LiteLLM-style abstraction

Why:

- reduces model-provider-specific code,
- allows cross-provider experiments.

## OpenTelemetry

Why:

- vendor-neutral observability standard,
- maps naturally to LLM/RAG/agent execution traces.

## Next.js + TypeScript + React

Why:

- strong ecosystem for interactive dashboards,
- easy deployment,
- typed UI code.

## Tailwind CSS

Why:

- fast dashboard UI development.

## Docker

Why:

- reproducible local environment,
- easy multi-service setup.

## GitHub Actions

Why:

- directly demonstrates the core “evals as CI tests” idea.

## pytest

Why:

- mature Python testing ecosystem.

---

# 45. Existing Eval Ecosystem Integration

EvalForge should learn from and interoperate with the ecosystem without becoming only a wrapper.

Potential integrations:

- Ragas for RAG-oriented metrics,
- DeepEval-style evaluator patterns,
- Inspect AI-style task evaluation,
- OpenTelemetry-compatible tracing,
- provider SDKs through an abstraction layer.

EvalForge's original engineering focus remains:

- experiment orchestration,
- dataset/version management,
- heterogeneous evaluator pipelines,
- regression logic,
- quality gates,
- failure analysis,
- judge calibration,
- production-failure feedback loops.

---

# 46. V0 — Design Validation

Before building the complete backend, validate the project with a minimal vertical slice.

Goal:

Prove the full concept.

Deliverable:

```text
small JSONL dataset
      ↓
two target configurations
      ↓
deterministic + judge evaluators
      ↓
baseline comparison
      ↓
terminal regression report
```

No dashboard is required yet.

---

# 47. V1 — Core Evaluation Engine

The first core workflow is implemented as described in the implementation ledger. This original milestone defines its goals:

## Features

- dataset schema,
- JSONL import,
- target interface,
- raw LLM target,
- deterministic evaluators,
- LLM judge evaluator,
- experiment runner,
- PostgreSQL persistence,
- baseline vs. candidate comparison,
- regression rules,
- CLI report.

## Acceptance criteria

A developer can:

1. define 20+ cases,
2. run two model/prompt configurations,
3. receive case-level metrics,
4. see aggregate metrics,
5. detect a configured regression,
6. receive a non-zero CLI exit code when a gate fails.

---

# 48. V2 — CI, RAG, and Dashboard

## Features

- GitHub Actions workflow,
- RAG target adapter,
- retrieval trace capture,
- Recall@K / Precision@K,
- groundedness,
- faithfulness,
- cost tracking,
- latency tracking,
- dashboard,
- compare-runs page,
- failure explorer.

## Acceptance criteria

A pull request can trigger a reduced eval suite and fail automatically when:

- a critical case fails,
- a category metric regresses too far,
- cost/latency exceeds configured tolerance.

---

# 49. V3 — Agent Evals and Calibration

## Features

- agent target adapter,
- tool-call traces,
- trajectory evaluator,
- task-completion evaluator,
- human review,
- judge-vs-human agreement dashboard,
- evaluator versioning,
- pairwise evaluation.

## Acceptance criteria

The platform can determine:

- whether an agent completed a task,
- whether correct tools were used,
- whether forbidden steps occurred,
- whether judge scores agree with reviewed human labels.

---

# 50. V4 — Production Feedback Loop

## Features

- production trace ingestion,
- incident replay,
- failure clustering,
- candidate eval generation,
- human approval workflow,
- adaptive eval-set growth.

Goal:

Turn real-world failures into permanent regression coverage.

---

# 51. Detailed Build Order

Recommended implementation sequence:

## Phase A — Domain model

1. define eval-case schema,
2. define target-result schema,
3. define evaluator-result schema,
4. define experiment/run schema,
5. define regression-rule schema.

## Phase B — Local engine

6. JSONL loader,
7. target interface,
8. mock target,
9. deterministic evaluators,
10. scoring and aggregation,
11. baseline comparison,
12. terminal report.

## Phase C — Real model support

13. provider abstraction,
14. structured model calls,
15. judge evaluator,
16. token/latency/cost capture.

## Phase D — Persistence

17. PostgreSQL schema,
18. experiment persistence,
19. run persistence,
20. dataset versioning.

## Phase E — API and workers

21. FastAPI service,
22. async worker,
23. job status,
24. retries and error handling.

## Phase F — CI

25. config file,
26. CLI exit codes,
27. GitHub Actions example,
28. CI summary report.

## Phase G — RAG

29. retrieval trace schema,
30. retrieval evaluators,
31. groundedness,
32. RAG failure classification.

## Phase H — UI

33. project page,
34. experiment list,
35. compare page,
36. failure explorer.

## Phase I — Agent support

37. trajectory schema,
38. tool-call evaluators,
39. task evaluator,
40. trajectory inspection UI.

## Phase J — Calibration

41. human labels,
42. agreement metrics,
43. evaluator-version comparison.

---

# 52. Example End-to-End Demo Scenario

A strong portfolio demo should use a realistic support RAG assistant.

## Baseline

```text
Model A
Prompt V1
Retriever V1
Top-k = 5
```

## Candidate

```text
Model B
Prompt V2
Retriever V2
Top-k = 8
```

## Dataset

100 cases:

- 30 billing,
- 25 account access,
- 20 technical troubleshooting,
- 15 refunds,
- 10 adversarial / prompt injection.

## Result

```text
Overall correctness       84% → 91%
Groundedness              88% → 94%
Retrieval Recall@5        79% → 90%
Refund accuracy           96% → 78%  ❌
Prompt-injection defense  90% → 95%
Latency                   1.8s → 2.4s
Cost/request              $0.012 → $0.018
```

EvalForge should block the candidate because refund accuracy is a critical metric even though overall quality improved.

This demo clearly communicates why category-aware regression testing matters.

---

# 53. Example CI Decision

```text
Experiment: exp_202
Baseline: run_842
Candidate: run_911
Dataset: support-release-v12

PASS:
- overall correctness +7%
- groundedness +6%
- retrieval recall +11%
- injection defense +5%

FAIL:
- refund accuracy -18%

WARN:
- latency +33%
- cost +50%

Final decision:
❌ QUALITY GATE FAILED
Reason:
Critical metric "refund accuracy" exceeded allowed regression threshold.
```

---

# 54. Dashboard Success Criteria

The dashboard is successful if a developer can answer these questions quickly:

1. What changed?
2. Which metrics improved?
3. Which metrics regressed?
4. Which categories are affected?
5. Which exact cases failed?
6. What did baseline answer?
7. What did candidate answer?
8. What context was retrieved?
9. What tools were called?
10. Why did the evaluator mark it as a failure?
11. Did a human agree?
12. What was the cost and latency tradeoff?

---

# 55. Project Success Metrics

Technical success can be judged by whether EvalForge can:

- reproduce an experiment,
- compare two system versions,
- detect a known injected regression,
- separate RAG retrieval failure from generation failure,
- fail CI correctly,
- produce case-level evidence,
- track cost and latency,
- measure judge-human agreement,
- replay a production-style failure.

---

# 56. Risks and Mitigations

## Risk — LLM judge unreliability

Mitigation:

- deterministic checks first,
- clear rubrics,
- human calibration,
- evaluator versioning.

## Risk — Eval overfitting

Teams may optimize only for the known eval set.

Mitigation:

- rotating holdout sets,
- production-derived cases,
- adversarial suites.

## Risk — High evaluation cost

Mitigation:

- tiered PR/release suites,
- caching,
- selective judges,
- deterministic checks.

## Risk — False CI failures

Mitigation:

- minimum sample sizes,
- explicit thresholds,
- warning vs. blocking gates,
- human override workflow later.

## Risk — Data leakage

Mitigation:

- access control,
- redaction,
- secret handling,
- sanitized production traces.

## Risk — Scope explosion

Mitigation:

Build V1 vertical slice before RAG, agents, and production monitoring.

---

# 57. Known Limitations

Even a mature EvalForge cannot prove that an AI system is always correct.

Limitations include:

- eval-set coverage,
- judge bias,
- stochastic model behavior,
- domain-specific correctness requirements,
- hidden real-world edge cases,
- changing model-provider behavior,
- imperfect human labels.

EvalForge should provide stronger evidence and regression protection, not claim mathematical certainty.

---

# 58. Future Extensions

Potential future directions:

- automatic adversarial case generation,
- multimodal evals,
- speech / image model evals,
- red-team suites,
- model-router evaluation,
- online A/B evaluation,
- drift detection,
- dataset coverage analysis,
- failure clustering,
- evaluator recommendation,
- custom organization policies,
- RBAC,
- hosted SaaS deployment,
- self-hosted enterprise mode,
- SDKs for Python and TypeScript,
- IDE integration,
- pull-request annotations,
- benchmark import/export.

---

# 59. What Makes EvalForge Technically Interesting

EvalForge should not be presented as:

> “I made a UI that asks one LLM to rate another LLM.”

The real engineering value comes from combining:

- versioned eval datasets,
- model/provider abstraction,
- execution traces,
- heterogeneous evaluator pipelines,
- RAG-specific diagnostics,
- agent trajectory evaluation,
- baseline/candidate paired comparison,
- category-level regression detection,
- cost and latency tradeoffs,
- CI/CD quality gates,
- human judge calibration,
- adaptive production feedback loops.

That makes the project an AI reliability and infrastructure system rather than a simple LLM wrapper.

---

# 60. Resume-Level Technical Narrative

Once implemented, the project should support a strong engineering story such as:

> Built EvalForge, a continuous evaluation and regression-testing platform for LLM, RAG, and agent applications. Designed versioned eval datasets, deterministic and model-based evaluators, RAG and tool-trajectory metrics, baseline/candidate regression gates, cost/latency tracking, and GitHub Actions integration to block quality regressions before deployment.

The exact resume bullet should use only features that are actually implemented and measured.

---

# 61. Final Definition of Done for the Portfolio Version

The portfolio version is complete when a reviewer can clone the repository and understand or run a demo showing:

1. a realistic eval dataset,
2. a baseline AI configuration,
3. a candidate configuration,
4. deterministic evaluation,
5. LLM-judge evaluation,
6. RAG evaluation,
7. at least one agent/tool-use evaluation,
8. case-level traces,
9. baseline/candidate metric comparison,
10. a deliberately introduced regression,
11. a GitHub Actions quality gate that catches it,
12. a dashboard showing the failed cases,
13. cost and latency comparison,
14. human-reviewed labels for a small calibration set,
15. documentation explaining the architecture and tradeoffs.

That is the standard the project should aim for.

---

# 62. Implementation Rule

Do **not** build every advanced feature at once.

The guiding sequence is:

> **Make one complete eval loop work first. Then make it richer.**

The first end-to-end loop is:

```text
Dataset
  ↓
Baseline + Candidate
  ↓
Execute
  ↓
Evaluate
  ↓
Compare
  ↓
Detect Regression
  ↓
Fail or Pass CI
```

Everything else should extend that core without breaking its simplicity.

---

# 63. Final Product Vision

EvalForge should eventually feel like:

> **unit tests + experiment tracking + observability + CI quality gates for probabilistic AI systems.**

A developer should be able to change a prompt, model, retriever, or agent workflow and receive an evidence-backed answer to:

> **What improved, what regressed, why, and is this safe to ship?**
