"""Sequential, evidence-preserving execution. Errors never become quality scores."""

import time

from evalforge.datasets import dataset_hash
from evalforge.evaluators import Evaluator
from evalforge.models import CaseResult, Dataset, EvaluatorResult, Run, TargetResult
from evalforge.providers import ProviderError
from evalforge.targets import Target

RESERVED_METRICS = {"overall", "latency_ms", "p95_latency_ms", "cost", "error_rate"}


def error_message(exc: Exception) -> str:
    # Arbitrary application/evaluator exception text can contain credentials or private payloads.
    return str(exc) if isinstance(exc, ProviderError) else f"{type(exc).__name__}: execution failed"


def run_experiment(
    dataset: Dataset,
    target: Target,
    evaluators: list[Evaluator],
    *,
    target_name: str,
    target_config: dict | None = None,
) -> Run:
    specs = [e.spec.model_copy(deep=True) for e in evaluators]
    metrics = [spec.metric for spec in specs]
    if not metrics or len(metrics) != len(set(metrics)) or RESERVED_METRICS.intersection(metrics):
        raise ValueError("evaluators must have unique, non-reserved metrics")
    rows = []
    for case in dataset.cases:
        start = time.perf_counter()
        try:
            result = target.execute(case.model_copy(deep=True))
            if not isinstance(result, TargetResult):
                raise TypeError("target must return TargetResult")
            result = TargetResult.model_validate(result.model_dump())
        except Exception as exc:
            result = TargetResult(status="ERROR", error=error_message(exc))
        result.latency_ms = (time.perf_counter() - start) * 1000
        scores = []
        for evaluator in evaluators:
            spec = evaluator.spec
            if result.status == "ERROR":
                score = EvaluatorResult(
                    metric=spec.metric,
                    evaluator_version=spec.version,
                    status="SKIPPED",
                    explanation="target execution failed",
                )
            else:
                try:
                    score = evaluator.evaluate(
                        case.model_copy(deep=True), result.model_copy(deep=True)
                    )
                    score = EvaluatorResult.model_validate(score.model_dump())
                    if score.metric != spec.metric or score.evaluator_version != spec.version:
                        raise ValueError("evaluator returned incorrect metric/version")
                except Exception as exc:
                    score = EvaluatorResult(
                        metric=spec.metric,
                        evaluator_version=spec.version,
                        status="ERROR",
                        explanation=error_message(exc),
                    )
            scores.append(score)
        rows.append(CaseResult(case=case.model_copy(deep=True), target=result, evaluations=scores))
    has_errors = any(
        row.target.status == "ERROR" or any(e.status == "ERROR" for e in row.evaluations)
        for row in rows
    )
    return Run(
        dataset_name=dataset.name,
        dataset_version=dataset.version,
        dataset_hash=dataset_hash(dataset.cases),
        target_name=target_name,
        target_config=target_config or {},
        evaluators=specs,
        cases=rows,
        status="ERROR" if has_errors else "COMPLETED",
    )
