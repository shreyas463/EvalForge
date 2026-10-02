"""Bounded concurrent, evidence-preserving execution. Errors never become quality scores."""

import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

from evalforge.budgets import CURRENT_BUDGET, BudgetExceeded, RunBudget
from evalforge.datasets import dataset_hash
from evalforge.evaluators import Evaluator
from evalforge.models import (
    CaseResult,
    Dataset,
    EvaluatorResult,
    ExecutionLimits,
    Run,
    TargetResult,
)
from evalforge.providers import ProviderError
from evalforge.targets import Target

RESERVED_METRICS = {"overall", "latency_ms", "p95_latency_ms", "cost", "error_rate"}


def error_message(exc: Exception) -> str:
    # Arbitrary application/evaluator exception text can contain credentials or private payloads.
    return (
        str(exc)
        if isinstance(exc, (ProviderError, BudgetExceeded))
        else f"{type(exc).__name__}: execution failed"
    )


def run_experiment(
    dataset: Dataset,
    target: Target,
    evaluators: list[Evaluator],
    *,
    target_name: str,
    target_config: dict | None = None,
    limits: ExecutionLimits | None = None,
    budget: RunBudget | None = None,
) -> Run:
    specs = [e.spec.model_copy(deep=True) for e in evaluators]
    metrics = [spec.metric for spec in specs]
    if not metrics or len(metrics) != len(set(metrics)) or RESERVED_METRICS.intersection(metrics):
        raise ValueError("evaluators must have unique, non-reserved metrics")
    limits = limits or (budget.limits if budget else ExecutionLimits())
    if budget and limits != budget.limits:
        raise ValueError("runner limits must match the shared budget")
    budget = budget or RunBudget(limits)

    def execute_case(case):
        start = time.perf_counter()
        result = None
        try:
            budget.check_start()
            result = target.execute(case.model_copy(deep=True))
            if not isinstance(result, TargetResult):
                raise TypeError("target must return TargetResult")
            result = TargetResult.model_validate(result.model_dump())
            budget.check_completion()
        except Exception as exc:
            output = result.output if isinstance(result, TargetResult) else None
            result = TargetResult(status="ERROR", error=error_message(exc), output=output)
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
                    budget.check_deadline()
                    score = evaluator.evaluate(
                        case.model_copy(deep=True), result.model_copy(deep=True)
                    )
                    budget.check_completion()
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
        return CaseResult(case=case.model_copy(deep=True), target=result, evaluations=scores)

    def worker(case):
        token = CURRENT_BUDGET.set(budget)
        try:
            return execute_case(case)
        finally:
            CURRENT_BUDGET.reset(token)

    rows = [None] * len(dataset.cases)
    # Keep at most concurrency futures in flight; do not queue the entire dataset.
    with ThreadPoolExecutor(max_workers=limits.concurrency, thread_name_prefix="evalforge") as pool:
        pending = {}
        next_case = 0
        while next_case < len(dataset.cases) or pending:
            while next_case < len(dataset.cases) and len(pending) < limits.concurrency:
                pending[pool.submit(worker, dataset.cases[next_case])] = next_case
                next_case += 1
            completed, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed:
                rows[pending.pop(future)] = future.result()

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
        execution=budget.summary(),
    )
