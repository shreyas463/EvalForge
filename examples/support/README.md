# ForgeDesk support application demo

This small fictional application demonstrates the complete reviewed-baseline workflow. Its policy says refunds are available only within 14 days; the candidate changes the application policy to 30 days. It includes billing, cancellation, password, rate-limit and API-key policies.

## Offline application and regression gate

From the repository root, with EvalForge installed:

```bash
python -m evalforge.demos.support 'Can I get a refund after 30 days?'
python -m evalforge.demos.workflow
```

The first command runs the rule-based assistant. It uses the input question and its own policy; it never reads eval IDs or reference answers. This mode is intentionally a simple lexical FAQ implementation, not an LLM or RAG system.

The workflow:

1. runs the correct 14-day application version against itself;
2. saves raw run evidence and verifies the quality gate;
3. approves one saved run under `support-reviewed` with an audit record;
4. executes only the changed 30-day candidate against that stored baseline;
5. verifies a refund-category gate fails on the expected refund cases.

The workflow wrapper exits **0** when that expected regression is established. The underlying candidate evaluation exits **1**. Unexpected passes, execution failures, or failures without refund-regression evidence make the wrapper fail. CI runs the wrapper without credentials or paid calls.

Eight independently authored cases in `cases.jsonl` check required and forbidden literal facts. `offline.json` configures callable targets with explicit policy parameters. `prompt-v1.txt` and `prompt-v2.txt` contain the corresponding provider-backed application policies.

## Live model and judge validation

Configure the credential environment variable without putting its value in repository files or commands. Choose model IDs supported by your endpoint for text Chat Completions, temperature, `max_completion_tokens`, and JSON object mode for the judge.

```bash
python -m evalforge.demos.workflow --live \
  --model YOUR_CHAT_MODEL --judge-model YOUR_JUDGE_MODEL
```

For another compatible endpoint, add `--base-url URL --api-key-env YOUR_CREDENTIAL_VARIABLE_NAME`. `live.json` combines deterministic checks with a structured correctness judge. Prompt file contents and hashes, model/provider settings, judge specs, outputs, usage and scores are saved. If the initial baseline does not pass, it is not approved and the candidate phase does not run. If a live model does not exhibit the intended refund regression, the wrapper fails rather than claiming success.

**No live paid calls have been validated yet.** Controlled HTTP tests exercise all 48 successful target/judge calls in the two-phase live workflow. An actual run normally makes 32 calls to seed the baseline pair plus 16 candidate-only calls. Each phase shares a maximum of 40 provider attempts (including retries), so the complete workflow can issue at most 80 attempts. Output is capped at 300 completion tokens per attempt; the phase deadline is 300 seconds. Those request/output limits are not a dollar spending cap. No pricing is assumed. Use provider-side spending limits for financial controls, or configure observed-cost limits and both token prices before running.

For interactive provider-backed answers without evaluation:

```bash
python -m evalforge.demos.support 'What is the refund policy?' --model YOUR_CHAT_MODEL
```

This command caps completion output at 300 tokens, permits at most two attempts, and has a cooperative 60-second budget.

## Evidence

Artifacts are saved under `.evalforge/` by default (`--output-dir` changes the root). The workflow creates two normal experiment directories plus a `support-demo-UUID` directory containing generated configs and its SQLite approval registry. Each invocation is isolated; existing run evidence and approvals are preserved. No secret values are included in generated configs. Questions and model outputs are retained as application data.

Prompt wording is stochastic when used with live models. Literal checks may reject a valid paraphrase; the judge is a separate, imperfect quality signal. This demo proves engine/application integration and a known policy-regression path, not general support-assistant quality or judge calibration.
