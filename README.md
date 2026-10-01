# EvalForge

> **Continuous evaluation and regression testing for LLM applications, RAG systems, and AI agents.**

EvalForge is a developer-focused platform for testing AI systems the way traditional software is tested in CI/CD. Instead of relying on “this output looks better,” teams can run a repeatable evaluation suite against different prompts, models, retrieval pipelines, or agent versions and measure whether quality actually improved or regressed.

**Project status:** Design / early development.

---

## Project Idea

Modern AI applications are difficult to test because their outputs are probabilistic. A normal unit test can tell you whether a function returned the expected value, but an LLM response can be phrased differently and still be correct — or sound convincing while being factually wrong.

EvalForge provides a structured way to evaluate those systems. A developer supplies an LLM, RAG application, or agent plus a dataset of representative test cases. EvalForge runs the same cases across one or more versions, scores the outputs using multiple evaluators, compares the results, and flags meaningful regressions before a change reaches production.

The goal is simple:

**make AI quality measurable, reproducible, and testable.**

---

## Problem Statement

AI teams regularly change:

- system prompts
- model providers or model versions
- retrieval strategies
- embedding models
- chunking settings
- tools available to an agent
- orchestration logic
- safety rules

A change can improve one part of the application while silently making another part worse.

For example, a new RAG pipeline might improve answer quality overall but reduce accuracy for billing questions. A new model might answer better but double cost and latency. An agent might reach the right final answer while taking an unsafe or incorrect sequence of tool calls.

Traditional unit tests do not capture these failures well.

EvalForge is designed to answer:

> **Did this AI system actually get better, and what specifically got worse?**

---

## Example

Imagine a support assistant receives:

> "Can I cancel my Pro plan and get a refund after 30 days?"

The source policy says refunds are available only within **14 days**.

Model A answers:

> "Yes, refunds are available within 30 days."

The answer sounds plausible, but it is incorrect.

EvalForge could record:

- Correctness: FAIL
- Groundedness: FAIL
- Policy compliance: FAIL
- Hallucination detected: YES
- Retrieval quality: PASS
- Latency: 1.4 s
- Cost: $0.006

If a new prompt or model changes the answer correctly, EvalForge can show that improvement. If another category becomes worse at the same time, the regression is surfaced separately.

---

## Core Workflow

```text
                 ┌──────────────────┐
                 │   Eval Dataset   │
                 └────────┬─────────┘
                          │
                          ▼
┌───────────────┐   ┌───────────────┐
│ Model / RAG / │──▶│ Eval Runner   │
│ Agent Version │   └───────┬───────┘
└───────────────┘           │
                            ▼
                   ┌─────────────────┐
                   │   Evaluators    │
                   │                 │
                   │ deterministic   │
                   │ LLM-as-a-judge  │
                   │ RAG metrics     │
                   │ agent metrics   │
                   │ cost / latency  │
                   └────────┬────────┘
                            │
                            ▼
                   ┌─────────────────┐
                   │ Experiment +    │
                   │ Regression Diff │
                   └────────┬────────┘
                            │
                  ┌─────────┴─────────┐
                  ▼                   ▼
          ┌──────────────┐    ┌──────────────┐
          │  Dashboard   │    │ GitHub CI    │
          └──────────────┘    │ Pass / Fail  │
                              └──────────────┘
```

---

## Planned Features

### 1. Evaluation Datasets

Create versioned datasets containing:

- input prompts
- expected behavior
- reference answers
- expected facts
- retrieved context
- required / forbidden tool calls
- metadata and categories
- edge cases and adversarial cases

This allows the same test suite to be run repeatedly as an AI system changes.

### 2. Multi-Model and Multi-Version Experiments

Run the same dataset against combinations such as:

- Model A + Prompt V1
- Model A + Prompt V2
- Model B + Prompt V2
- RAG V1 vs. RAG V2
- Agent workflow V1 vs. V2

EvalForge then produces a side-by-side comparison rather than relying on subjective inspection.

### 3. Multiple Evaluation Strategies

Not every AI output should be graded by another LLM.

EvalForge will combine several evaluation methods:

**Deterministic evaluators**
- exact / partial match
- regex checks
- JSON-schema validation
- required keyword / fact checks
- forbidden-content checks
- tool-call validation

**Semantic evaluators**
- embedding similarity
- reference-answer similarity

**LLM-as-a-judge**
- correctness
- relevance
- completeness
- groundedness
- instruction following
- pairwise preference

**RAG evaluators**
- context relevance
- answer faithfulness
- retrieval precision
- retrieval recall
- Recall@K / Precision@K

**Agent evaluators**
- correct tool selection
- tool-call arguments
- trajectory validity
- unnecessary steps
- task completion

**Operational metrics**
- latency
- token usage
- estimated cost
- error rate

### 4. Regression Detection

EvalForge will compare a candidate run against a baseline.

Example:

```text
Overall quality          86% → 91%   +5%
Groundedness             90% → 95%   +5%
Technical support        82% → 89%   +7%
Refund-policy accuracy   94% → 76%  -18%  ❌ REGRESSION
Average latency          1.8s → 2.4s +33%
```

The important part is not only computing an average score, but showing **where** the system became worse.

### 5. GitHub CI Integration

EvalForge is intended to work like an AI-quality test suite inside GitHub Actions.

A pull request that changes a prompt, model configuration, retrieval pipeline, or agent workflow can automatically trigger an eval run.

Example result:

```text
❌ EvalForge regression detected

Refund-policy accuracy
Baseline: 94%
Candidate: 76%
Change: -18%

Regression threshold: -5%

CI check failed.
```

This allows teams to catch model-quality regressions before merging code.

### 6. Failure Explorer

For failed examples, developers should be able to inspect:

```text
Input
  ↓
Retrieved Context
  ↓
Model Response
  ↓
Expected Behavior
  ↓
Evaluator Scores
  ↓
Judge Explanation
```

The goal is to make failures debuggable rather than only producing a numeric score.

### 7. Judge Calibration

LLM judges are also imperfect.

EvalForge will support human labels so that automated evaluators can be compared against human judgments.

Planned measurements include:

- judge vs. human agreement
- pairwise agreement
- false-positive / false-negative analysis
- judge consistency
- evaluator confidence

This addresses an important question in LLM evaluation:

> **Who evaluates the evaluator?**

### 8. Adaptive Eval Generation

A later feature will turn production failures into new regression tests.

```text
Production failure
      ↓
Failure trace
      ↓
Candidate eval case generated
      ↓
Human review
      ↓
Added to eval dataset
      ↓
Future versions tested against it
```

This allows the evaluation suite to become stronger as the application encounters new failures.

### 9. Experiment Dashboard

The dashboard will provide:

- experiment history
- baseline vs. candidate comparison
- metric breakdowns
- category-level regressions
- individual failed cases
- model / prompt metadata
- latency and cost comparison
- evaluator explanations

---

## Proposed Technology Stack

### Backend / Eval Engine

- **Python** — core evaluation engine
- **FastAPI** — API layer
- **Pydantic** — typed eval schemas and structured outputs
- **SQLAlchemy** — persistence layer
- **PostgreSQL** — experiments, datasets, runs, metrics, and labels
- **Redis** — caching and job coordination
- **Celery** or a lightweight worker queue — parallel evaluation jobs

### LLM / Model Layer

- **LiteLLM** — common interface across multiple model providers
- provider adapters for **OpenAI, Anthropic, and Gemini**
- structured outputs for judge responses
- embeddings for semantic evaluation

The architecture will remain provider-independent so the same eval suite can compare models from different vendors.

### Evaluation

EvalForge will implement its own core evaluator interface while integrating useful ideas or adapters from existing evaluation ecosystems such as:

- **Ragas** for RAG-oriented metrics
- **DeepEval** for reusable evaluation patterns
- **Inspect AI** / custom task-style evaluation where appropriate

The project is not intended to be only a wrapper around another eval library. The main engineering work is the experiment engine, evaluator orchestration, regression logic, judge calibration, tracing, CI integration, and debugging workflow.

### Observability

- **OpenTelemetry** for traces and spans
- trace capture for prompts, retrieval, model calls, and tool calls
- token, latency, and cost instrumentation

### Frontend

- **Next.js**
- **TypeScript**
- **React**
- **Tailwind CSS**

The frontend will focus on experiment comparison and failure analysis rather than being a generic chat UI.

### Infrastructure

- **Docker**
- **GitHub Actions**
- **pytest**
- **PostgreSQL**
- optional cloud deployment on **AWS / GCP / Azure**

---

## Proposed Architecture

```text
                        EvalForge
                            │
           ┌────────────────┼────────────────┐
           │                │                │
           ▼                ▼                ▼
      Dataset API      Experiment API    Dashboard
           │                │
           └────────┬───────┘
                    ▼
              Evaluation Runner
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
   LLM App       RAG App     AI Agent
        │           │           │
        └───────────┼───────────┘
                    ▼
             Trace Collection
                    │
                    ▼
            Evaluator Pipeline
       ┌────────────┼────────────┐
       ▼            ▼            ▼
 Deterministic   LLM Judge    RAG / Agent
    Checks                        Metrics
       └────────────┼────────────┘
                    ▼
              Score Aggregator
                    │
                    ▼
             Regression Engine
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
     PostgreSQL          GitHub Action
          │              Pass / Fail
          ▼
      Dashboard
```

---

## Suggested Development Roadmap

### V1 — Evaluation Engine

Build the smallest complete workflow:

- dataset format
- model adapter
- eval runner
- deterministic evaluators
- LLM-as-a-judge
- experiment storage
- baseline vs. candidate comparison
- CLI output

### V2 — RAG + CI

Add:

- RAG metrics
- retrieval tracing
- GitHub Actions integration
- regression thresholds
- cost / latency tracking
- web dashboard
- failure explorer

### V3 — Advanced Evals

Add:

- agent trajectory evaluation
- pairwise model comparison
- human annotations
- judge calibration
- adaptive eval generation from failures
- production trace ingestion

---

## What Makes EvalForge Different From a Simple LLM Judge

A basic eval project might ask another model:

> "Score this answer from 1 to 10."

EvalForge is intended to go substantially further.

It treats evaluation as an engineering system:

- versioned datasets
- repeatable experiments
- heterogeneous evaluators
- RAG and agent traces
- baseline comparisons
- category-level regressions
- judge calibration
- latency / cost tradeoffs
- CI/CD quality gates
- failure analysis
- continuously improving eval datasets

The goal is not just to **score LLM outputs**.

The goal is to build infrastructure that helps developers answer:

> **Can I safely ship this new version of my AI application?**

---

## Long-Term Vision

EvalForge should become a lightweight quality layer that can sit between AI development and deployment.

```text
Build AI feature
      ↓
Run EvalForge
      ↓
Compare against baseline
      ↓
Investigate failures
      ↓
Pass quality thresholds
      ↓
Merge / deploy
      ↓
Capture production failures
      ↓
Add new eval cases
      ↓
Repeat
```

As AI systems become more agentic and less deterministic, reliable evaluation becomes part of the software-development lifecycle rather than a one-time benchmark.

---

## Current Scope

The first goal is **not** to build every feature above at once.

The initial milestone is a working end-to-end system that can:

1. accept a small eval dataset,
2. run two LLM application versions,
3. score them with deterministic and model-based evaluators,
4. compare the results,
5. identify regressions,
6. fail a GitHub Actions check when configured thresholds are exceeded.

That gives EvalForge a useful core before expanding into RAG evaluation, agent trajectories, human calibration, and production monitoring.

---

## License

License to be added as the project develops.
