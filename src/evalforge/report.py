"""Human-readable reports with gate decisions and case-level evidence."""

import json

from evalforge.models import Comparison, Run
from evalforge.scoring import aggregate


def _value(value):
    return "n/a" if value is None else f"{value:.6g}"


def render_report(baseline: Run, candidate: Run, comparison: Comparison) -> str:
    lines = [
        f"EvalForge Quality Gate: {comparison.status}",
        f"Dataset: {baseline.dataset_name}@{baseline.dataset_version}",
        f"Baseline: {baseline.id} ({baseline.target_name})",
        f"Candidate: {candidate.id} ({candidate.target_name})",
        "",
        "Metric | Baseline | Candidate | Delta | Samples | Gate",
    ]
    for gate in comparison.gates:
        label = gate.rule.metric + (f" [{gate.rule.category}]" if gate.rule.category else "")
        lines.append(
            f"{label} | {_value(gate.baseline)} | {_value(gate.candidate)} | "
            f"{_value(gate.delta)} | {gate.samples} | {gate.status} ({gate.rule.severity})"
        )
        lines.append(f"  {gate.explanation}")
        if gate.case_ids:
            lines.append(f"  affected cases: {', '.join(gate.case_ids)}")
    lines.extend(
        ["", f"Execution status: baseline={baseline.status}, candidate={candidate.status}"]
    )
    for row in baseline.cases:
        if row.target.status == "ERROR":
            lines.append(f"Baseline target error [{row.case.id}]: {row.target.error}")
        for evaluation in row.evaluations:
            if evaluation.status in {"ERROR", "UNKNOWN"}:
                lines.append(
                    f"Baseline evaluator error [{row.case.id}/{evaluation.metric}]: "
                    f"{json.dumps(evaluation.explanation)}"
                )
    if candidate.execution:
        lines.append(
            "Execution budget evidence: " + json.dumps(candidate.execution, sort_keys=True)
        )
    lines.extend(["", "Candidate failures / errors:"])
    before = {r.case.id: r for r in baseline.cases}
    for row in candidate.cases:
        failures = [e for e in row.evaluations if e.status in {"FAIL", "ERROR", "UNKNOWN"}]
        if row.target.status == "ERROR" or failures:
            lines.append(f"- {row.case.id} ({row.case.category}, critical={row.case.critical})")
            lines.append(f"  baseline output: {json.dumps(before[row.case.id].target.output)}")
            lines.append(f"  candidate output: {json.dumps(row.target.output)}")
            for label, target in [
                ("baseline", before[row.case.id].target),
                ("candidate", row.target),
            ]:
                if target.retrieval:
                    lines.append(f"  {label} retrieved sources:")
                    for passage in target.retrieval.passages:
                        lines.append(
                            f"    {passage.rank}. {passage.chunk_id} (score={passage.score:.4g})"
                        )
                        lines.append(f"       {json.dumps(passage.text)}")
                    lines.append(f"  {label} citations: {json.dumps(target.retrieval.citations)}")
                if target.trajectory:
                    lines.append(f"  {label} tool calls:")
                    for call in target.trajectory.calls:
                        lines.append(
                            f"    {call.id}. {call.tool} {call.status} "
                            f"arguments={json.dumps(call.arguments)} "
                            f"output={json.dumps(call.output)}"
                        )
            if row.target.error:
                lines.append(f"  target error: {row.target.error}")
            lines.extend(
                f"  {e.metric}: {e.status} — {json.dumps(e.explanation)}" for e in failures
            )
    lines.extend(["", "Aggregate quality by category:"])
    for category, metrics in aggregate(candidate).items():
        summary = metrics["overall"]
        lines.append(
            f"{category}: {_value(summary.value)} ({summary.samples}/{summary.total} scored cases)"
        )
    return "\n".join(lines) + "\n"
