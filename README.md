# EvalForge

> **Continuous evaluation and regression testing for LLM applications, RAG systems, and AI agents.**

EvalForge is a developer-focused evaluation platform for testing AI systems the way traditional software is tested in CI/CD. It runs repeatable eval suites across prompts, models, retrieval pipelines, or agent versions; scores quality with multiple evaluator types; tracks cost and latency; compares candidate runs against baselines; and flags meaningful regressions before a change reaches production.

**Project status:** Design / architecture phase. No application code has been added yet.

## Why EvalForge?

AI applications are probabilistic. A model can produce a fluent answer that is wrong, ungrounded, unsafe, or inconsistent. A change that improves one task can silently hurt another. Traditional unit tests alone are not enough to answer:

> **Did this AI system actually get better, and what specifically got worse?**

EvalForge is designed to make AI quality **measurable, reproducible, debuggable, and enforceable in CI/CD**.

## Core Capabilities

EvalForge is planned around the following capabilities:

- versioned evaluation datasets
- multi-model and multi-prompt experiments
- deterministic checks
- semantic similarity evaluators
- LLM-as-a-judge evaluators
- pairwise model comparisons
- RAG retrieval and groundedness evaluation
- agent/tool-call/trajectory evaluation
- latency, token, cost, and error-rate tracking
- baseline vs. candidate regression detection
- category-level quality gates
- GitHub Actions integration
- failure inspection and trace analysis
- human review and judge calibration
- production-trace-to-eval workflows
- adaptive eval-set growth from real failures

## Simple Example

Suppose a support assistant is asked:

> "Can I cancel my Pro plan and get a refund after 30 days?"

The policy states that refunds are available only within 14 days.

A model responds:

> "Yes, refunds are available within 30 days."

The response sounds reasonable, but it is wrong. EvalForge could record:

- Correctness: **FAIL**
- Groundedness: **FAIL**
- Policy compliance: **FAIL**
- Hallucination detected: **YES**
- Retrieval quality: **PASS**
- Latency: **1.4 s**
- Cost: **$0.006**

If a new prompt, model, retriever, or agent workflow is introduced, EvalForge runs the same eval suite again and compares the result against a chosen baseline.

## Core Workflow

```text
Eval Dataset
    │
    ▼
Candidate AI System
(LLM / RAG / Agent)
    │
    ▼
Execution + Trace Capture
    │
    ▼
Evaluator Pipeline
 ┌───────────────┬───────────────┬───────────────┬──────────────┐
 │ Deterministic │ Semantic      │ LLM Judge     │ RAG / Agent  │
 │ checks        │ evaluators    │ evaluators    │ evaluators   │
 └───────────────┴───────────────┴───────────────┴──────────────┘
    │
    ▼
Score Aggregation
    │
    ▼
Baseline vs Candidate Comparison
    │
    ▼
Regression Engine
    │
 ┌──┴──────────────────────┐
 ▼                         ▼
Dashboard              CI Quality Gate
                        PASS / FAIL
```

## Example Regression Report

```text
Overall quality          86% → 91%   +5%
Groundedness             90% → 95%   +5%
Technical support        82% → 89%   +7%
Refund-policy accuracy   94% → 76%  -18%  ❌ REGRESSION
Average latency          1.8s → 2.4s +33%
Estimated cost/request   $0.012 → $0.018 +50%
```

The important behavior is not just computing one overall score. EvalForge should show **which capability regressed, by how much, on which test cases, and why**.

## Planned Evaluation Types

### LLM Output Evaluation

- correctness
- relevance
- completeness
- instruction following
- formatting / schema validity
- hallucination checks
- factual consistency
- policy compliance
- pairwise preference

### RAG Evaluation

- retrieval precision
- retrieval recall
- Recall@K / Precision@K
- context relevance
- answer faithfulness
- answer groundedness
- citation correctness
- retrieval failure vs. generation failure separation

### Agent Evaluation

- task completion
- correct tool selection
- tool-call arguments
- required / forbidden tool use
- trajectory validity
- unnecessary steps
- recovery from tool errors
- final-answer quality
- token / latency / cost efficiency

### Operational Evaluation

- latency
- token usage
- estimated model cost
- timeout rate
- provider error rate
- retry rate

## Proposed Technology Stack

### Evaluation / Backend
- Python
- FastAPI
- Pydantic
- SQLAlchemy
- PostgreSQL
- Redis
- background worker queue

### Model Abstraction
- LiteLLM-style provider abstraction
- OpenAI, Anthropic, Gemini, and compatible endpoints
- structured judge outputs
- embeddings for semantic evaluators

### Evaluation Ecosystem
EvalForge will own its core evaluator interface and experiment/regression engine, while allowing adapters or inspiration from tools such as:

- Ragas
- DeepEval
- Inspect AI
- custom evaluators

The goal is **not** to build a thin wrapper around an existing eval library.

### Observability
- OpenTelemetry
- prompt / model / retrieval / tool-call traces
- token, latency, error, and cost instrumentation

### Frontend
- Next.js
- TypeScript
- React
- Tailwind CSS

### Infrastructure
- Docker
- GitHub Actions
- pytest
- PostgreSQL
- optional cloud deployment later

## Project Scope

The first meaningful version will prove one complete workflow:

1. define a small eval dataset,
2. execute two versions of an LLM application,
3. score them with deterministic and model-based evaluators,
4. persist the experiment,
5. compare candidate vs. baseline,
6. surface failing cases,
7. fail a GitHub Actions quality gate when configured thresholds are exceeded.

RAG evaluation, agent trajectory evaluation, human calibration, production trace ingestion, and adaptive eval generation build on that foundation.

## Full Technical Design

The exhaustive architecture, data model, evaluator design, API plan, schemas, CI behavior, development milestones, security model, testing strategy, deployment plan, and implementation blueprint live here:

**[TECHNICAL_DESIGN.md](./TECHNICAL_DESIGN.md)**

That document is intended to be the source of truth for how EvalForge will be built.

## Long-Term Vision

```text
Build or modify AI feature
        ↓
Run EvalForge
        ↓
Compare against baseline
        ↓
Inspect regressions
        ↓
Pass quality thresholds
        ↓
Merge / deploy
        ↓
Capture real failures
        ↓
Convert failures into new eval cases
        ↓
Repeat
```

The long-term goal is for evaluation to become a normal part of the AI software-development lifecycle rather than a one-time benchmark.

## License

License to be added as the project develops.
