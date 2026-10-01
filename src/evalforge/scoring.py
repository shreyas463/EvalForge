"""Weighted case-level quality and operational metrics with explicit coverage."""

import math

from evalforge.models import Model, Run


class MetricSummary(Model):
    value: float | None
    samples: int
    total: int
    errors: int = 0
    skipped: int = 0


def case_scores(run: Run, metric: str, category: str | None = None) -> dict[str, float]:
    weights = {e.metric: e.weight for e in run.evaluators}
    scores = {}
    for row in run.cases:
        if category is not None and row.case.category != category:
            continue
        selected = [e for e in row.evaluations if metric == "overall" or e.metric == metric]
        if not selected or any(e.status in {"ERROR", "UNKNOWN"} for e in selected):
            continue
        scored = [e for e in selected if e.score is not None]
        if scored:
            scores[row.case.id] = sum(e.score * weights[e.metric] for e in scored) / sum(
                weights[e.metric] for e in scored
            )
    return scores


def summarize(run: Run, metric: str = "overall", category: str | None = None) -> MetricSummary:
    rows = [r for r in run.cases if category is None or r.case.category == category]
    if metric in {"latency_ms", "p95_latency_ms", "cost", "error_rate"}:
        if metric == "error_rate":
            values = [float(r.target.status == "ERROR") for r in rows]
        elif metric == "cost":
            values = [r.target.cost for r in rows if r.target.cost is not None]
        else:
            values = [r.target.latency_ms for r in rows]
        value = None
        if values:
            value = (
                sorted(values)[math.ceil(0.95 * len(values)) - 1]
                if metric == "p95_latency_ms"
                else sum(values) / len(values)
            )
        return MetricSummary(value=value, samples=len(values), total=len(rows))
    scores = case_scores(run, metric, category)
    denominator = sum(r.case.weight for r in rows if r.case.id in scores)
    value = sum(scores[r.case.id] * r.case.weight for r in rows if r.case.id in scores)
    selected = [e for r in rows for e in r.evaluations if metric == "overall" or e.metric == metric]
    return MetricSummary(
        value=value / denominator if denominator else None,
        samples=len(scores),
        total=len(rows),
        errors=sum(e.status in {"ERROR", "UNKNOWN"} for e in selected),
        skipped=sum(e.status == "SKIPPED" for e in selected),
    )


def aggregate(run: Run) -> dict[str, dict[str, MetricSummary]]:
    metrics = [
        "overall",
        *(e.metric for e in run.evaluators),
        "latency_ms",
        "p95_latency_ms",
        "cost",
        "error_rate",
    ]
    groups = {"all": {m: summarize(run, m) for m in metrics}}
    for category in sorted({r.case.category for r in run.cases}):
        groups[f"category:{category}"] = {m: summarize(run, m, category) for m in metrics}
    return groups
