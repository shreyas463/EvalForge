# Tool workflow and recorded-trace evaluation — no AI key

This example executes real Python functions against a **fictional local catalog**. It does not
use an AI agent, place orders, send messages, or access an external service. Its purpose is to show
that EvalForge can inspect the steps an application took, rather than only its final answer.
The SQuAD example remains the real public-data retrieval benchmark.

## Run it

```bash
evalforge dashboard --config examples/agents/regression.json --config examples/agents/unchanged.json --config examples/agents/recorded.json --open
```

Choose `regression.json` and **Run evaluation**. Expand a question's **Tool calls** to see
arguments, results, success/failure and their actual sequence. The original completes four quote
requests correctly. The changed version introduces:

- Notebook: an incorrect quantity passed to the total calculator.
- Pen: a forbidden full-catalog dump.
- Folder: a quote finalized before its total is calculated.
- Marker: an unnecessary extra calculation.

The checks block these changes. `unchanged.json` passes. `recorded.json` evaluates captures of
these same locally executed tools without executing the tools again. These are clearly labeled
local demo captures, not production traces or AI-generated behavior.

CLI equivalents:

```bash
evalforge run --config examples/agents/unchanged.json
evalforge run --config examples/agents/regression.json
evalforge run --config examples/agents/recorded.json
```

Exit codes: 0 passes required gates, 1 blocks a quality regression, 2 rejects configuration/data,
3 reports execution or storage errors. All three examples run through tests in existing CI.

## Tool policy contract

`TargetResult.trajectory` holds ordered calls with unique IDs, tool names, arguments, JSON outputs,
SUCCESS/ERROR status and an error for failed calls. `task_completed` is optional and reported by
the application or capture source. It is **not** inferred from the answer by AI.

Each case can supply `metadata.agent_rules`:

```json
{
  "required_tools": ["lookup_product", "calculate_total"],
  "forbidden_tools": ["dump_catalog"],
  "ordered_tools": ["lookup_product", "calculate_total", "finalize_quote"],
  "expected_arguments": {"calculate_total": {"quantity": 2}},
  "max_calls": 3
}
```

Evaluator kinds and semantics:

| Kind | What it checks |
|---|---|
| `required_tools` | Every required tool has a successful call. |
| `forbidden_tools` | No forbidden tool was attempted, including failed calls. |
| `tool_order` | Successful calls contain the configured ordered subsequence. Extra calls are allowed; use a call limit and forbidden tools too. |
| `tool_arguments` | Every attempt to each specified tool matches the specified top-level argument values. Extra argument keys are allowed; nested values match exactly. Missing tools fail. |
| `tool_efficiency` | Total calls, including failures/retries, do not exceed `max_calls`. |
| `tool_recovery` | Each failed tool has a later successful call to that same tool. This does not prove side effects were rolled back. |
| `task_completion` | The externally reported completion flag is true. Missing flags yield UNKNOWN. |

Checks return binary scores. Missing policy fields skip the corresponding policy check;
malformed policy or missing trajectory evidence is an error. Policies belong to evaluators,
not the application. The demo reads case.input only. Task completion and tool compliance do not
establish semantic final-answer accuracy. No simulated reasoning trace is generated.

## Replay your saved application results

Export any saved EvalForge run:

```bash
evalforge trace-export --run .evalforge/RUN_FOLDER/candidate.json --output /path/to/new-traces.jsonl
```

Or adapt your external application's capture into this strict JSONL format, one record per case:

```json
{"case_id":"case-1","input":"original question","result":{"output":"captured answer","status":"SUCCESS","latency_ms":123,"trajectory":{"calls":[],"task_completed":true}}}
```

Use `{"kind":"recorded","name":"saved-version","records":"traces.jsonl"}` for a target
in an experiment config. Relative paths resolve against that configuration. Other evaluator and
regression settings work as usual; model-based evaluators still need a provider key.

Replay requires matching case IDs and exact typed inputs; missing/mismatched records produce
execution errors. Duplicate IDs, malformed results, duplicate JSON keys and oversized files
(over 16 MiB) are rejected. The raw capture SHA-256 is saved in target configuration. Recorded
usage/cost is supplied evidence; replay makes no new provider calls. Target latency measures
**replay overhead**; the original reported latency is retained in metadata.observed_latency_ms.
Capture provenance is not independently authenticated. This is file-based ingestion, not a
live production collector. No executable instructions in captured tool names/arguments run.

## Human review and disagreement reports

After running an example, replace RUN_FOLDER with the saved result directory:

```bash
evalforge review-export --run .evalforge/RUN_FOLDER/candidate.json \
  --metric tool_arguments --destination /path/to/new-review
```

The new folder contains `evidence.json` (inputs, results, trace and relevant policy) and
`labels.jsonl` (blank human scores). Automatic evaluator scores are omitted from the evidence
for independent review. Read the evidence, then fill each label's `human_score` from 0 to 1,
`reviewer` and optional `note`. Leave unreviewed scores null. These must be real reviews;
EvalForge does not fill them in or claim that test-authored labels are human validation.

```bash
evalforge review-report --run .evalforge/RUN_FOLDER/candidate.json \
  --labels /path/to/new-review/labels.jsonl --output /path/to/new-review-report.json
```

The report contains review coverage, missing/unreviewed labels, unscored automatic results,
paired status agreement, mean absolute score difference, disagreements and supplied reviewer
labels/notes. The human pass threshold defaults to 0.5 and can be changed with
`--human-pass-threshold`. Automatic PASS/FAIL uses the original evaluator decision. Scores
without paired automatic scores are excluded from agreement, not silently counted as matches.
Run IDs, case IDs, metrics, duplicate labels and score ranges are validated. Original run evidence
is never rewritten. Reviewer identity is self-reported; statistics are descriptive, not proof
of judge reliability. No real human calibration has been performed yet.
