"""Paired comparisons and explicit, fail-closed quality gates."""

import json
import math

from evalforge.datasets import dataset_hash
from evalforge.models import Comparison, GateResult, RegressionRule, Run
from evalforge.scoring import case_scores, summarize


class ComparisonError(ValueError):
    pass


def _validate_pair(baseline: Run, candidate: Run):
    for run in (baseline, candidate):
        if dataset_hash([row.case for row in run.cases]) != run.dataset_hash:
            raise ComparisonError("run dataset hash does not match case evidence")
    if (baseline.dataset_name, baseline.dataset_version, baseline.dataset_hash) != (
        candidate.dataset_name,
        candidate.dataset_version,
        candidate.dataset_hash,
    ):
        raise ComparisonError(
            "baseline and candidate must use the same dataset version and content"
        )
    if sorted(
        json.dumps(e.model_dump(mode="json"), sort_keys=True) for e in baseline.evaluators
    ) != sorted(
        json.dumps(e.model_dump(mode="json"), sort_keys=True) for e in candidate.evaluators
    ):
        raise ComparisonError("baseline and candidate must use identical evaluator configurations")
    before = {r.case.id: r for r in baseline.cases}
    after = {r.case.id: r for r in candidate.cases}
    if before.keys() != after.keys():
        raise ComparisonError("baseline and candidate case IDs differ")
    # Applicability must not change; otherwise dropped measurements inflate candidate quality.
    for key in before:
        b = {e.metric: e.status == "SKIPPED" for e in before[key].evaluations}
        c = {e.metric: e.status == "SKIPPED" for e in after[key].evaluations}
        if b != c and before[key].target.status != "ERROR" and after[key].target.status != "ERROR":
            raise ComparisonError(f"evaluator applicability changed for case {key}")


def compare(baseline: Run, candidate: Run, rules: list[RegressionRule]) -> Comparison:
    _validate_pair(baseline, candidate)
    if not rules:
        raise ComparisonError("at least one explicit regression rule is required")
    gates = []
    for rule in rules:
        b = summarize(baseline, rule.metric, rule.category)
        c = summarize(candidate, rule.metric, rule.category)
        before_scores = case_scores(baseline, rule.metric, rule.category)
        after_scores = case_scores(candidate, rule.metric, rule.category)
        affected = sorted(
            k
            for k in before_scores.keys() & after_scores.keys()
            if after_scores[k] < before_scores[k]
        )
        reasons = []
        error = (
            b.value is None
            or c.value is None
            or b.samples < rule.min_samples
            or c.samples < rule.min_samples
            or b.errors
            or c.errors
        )
        if rule.metric == "cost" and (b.samples != b.total or c.samples != c.total):
            error = True
        delta = c.value - b.value if b.value is not None and c.value is not None else None
        if error:
            reasons.append("missing, incomplete, or insufficient metric evidence")
        else:
            if (
                rule.min_score is not None
                and c.value < rule.min_score
                and not math.isclose(c.value, rule.min_score, rel_tol=0, abs_tol=1e-12)
            ):
                reasons.append(f"candidate below minimum score {rule.min_score}")
            if (
                rule.max_regression is not None
                and b.value - c.value > rule.max_regression
                and not (
                    math.isclose(b.value - c.value, rule.max_regression, rel_tol=0, abs_tol=1e-12)
                )
            ):
                reasons.append(f"absolute regression exceeds {rule.max_regression}")
            if rule.max_relative_regression is not None:
                limit = abs(b.value) * rule.max_relative_regression
                if b.value - c.value > limit and not math.isclose(
                    b.value - c.value, limit, rel_tol=0, abs_tol=1e-12
                ):
                    reasons.append(f"relative regression exceeds {rule.max_relative_regression}")
            if rule.max_value is not None and c.value > rule.max_value:
                reasons.append(f"candidate exceeds maximum {rule.max_value}")
            if rule.max_increase_percent is not None:
                limit = abs(b.value) * rule.max_increase_percent / 100
                if delta > limit and not math.isclose(delta, limit, rel_tol=0, abs_tol=1e-12):
                    reasons.append(f"increase exceeds {rule.max_increase_percent}%")
        if rule.critical:
            selected = [
                r
                for r in candidate.cases
                if r.case.critical and (rule.category is None or r.case.category == rule.category)
            ]
            if not selected:
                error = True
                reasons.append("critical gate selected no critical cases")
            for row in selected:
                evaluations = [
                    e
                    for e in row.evaluations
                    if rule.metric == "overall" or e.metric == rule.metric
                ]
                if (
                    not evaluations
                    or not any(e.score is not None for e in evaluations)
                    or any(e.status in {"ERROR", "UNKNOWN"} for e in evaluations)
                ):
                    error = True
                    reasons.append(f"critical case {row.case.id} lacks evidence")
                if any(e.status == "FAIL" for e in evaluations):
                    affected.append(row.case.id)
                    reasons.append(f"critical case {row.case.id} failed")
        gates.append(
            GateResult(
                rule=rule,
                status="ERROR" if error else "FAIL" if reasons else "PASS",
                baseline=b.value,
                candidate=c.value,
                delta=delta,
                samples=min(b.samples, c.samples),
                case_ids=sorted(set(affected)),
                explanation="; ".join(reasons) or "all thresholds satisfied",
            )
        )
    uncertain = any(
        e.status in {"ERROR", "UNKNOWN"}
        for run in (baseline, candidate)
        for row in run.cases
        for e in row.evaluations
    )
    status = (
        "ERROR"
        if baseline.status == "ERROR"
        or candidate.status == "ERROR"
        or uncertain
        or any(g.status == "ERROR" for g in gates)
        else "FAIL"
        if any(g.status == "FAIL" and g.rule.severity == "block" for g in gates)
        else "PASS"
    )
    return Comparison(
        baseline_id=baseline.id, candidate_id=candidate.id, status=status, gates=gates
    )
