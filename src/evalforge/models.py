"""Validated, JSON-serializable contracts for the evaluation engine."""

from datetime import UTC, datetime
from typing import Annotated, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

Name = Annotated[str, Field(min_length=1, pattern=r"^\S(?:.*\S)?$")]
Score = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
NonNegative = Annotated[float, Field(ge=0, allow_inf_nan=False)]
Status = Literal["PASS", "FAIL", "ERROR", "SKIPPED", "UNKNOWN"]


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, allow_inf_nan=False)


class EvalCase(Model):
    id: Name
    input: str | dict[str, JsonValue]
    reference_answer: str | None = None
    expected_facts: list[str] = Field(default_factory=list)
    forbidden_facts: list[str] = Field(default_factory=list)
    expected_schema: dict[str, JsonValue] | None = None
    category: Name = "general"
    tags: list[Name] = Field(default_factory=list)
    critical: bool = False
    weight: Annotated[float, Field(gt=0, allow_inf_nan=False)] = 1
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class Dataset(Model):
    name: Name
    version: Name
    cases: list[EvalCase] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_cases(self):
        ids = [case.id for case in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate case IDs")
        return self


class TargetResult(Model):
    output: str | None = None
    status: Literal["SUCCESS", "ERROR"] = "SUCCESS"
    error: str | None = None
    latency_ms: NonNegative = 0
    input_tokens: Annotated[int, Field(ge=0)] | None = None
    output_tokens: Annotated[int, Field(ge=0)] | None = None
    cost: NonNegative | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def coherent_status(self):
        if self.status == "SUCCESS" and (self.output is None or self.error is not None):
            raise ValueError("SUCCESS requires output and no error")
        if self.status == "ERROR" and not self.error:
            raise ValueError("ERROR requires an error message")
        return self


class EvaluatorSpec(Model):
    metric: Name
    kind: Literal["exact_match", "contains", "excludes", "regex", "json_schema", "numeric", "judge"]
    version: Name = "1"
    options: dict[str, JsonValue] = Field(default_factory=dict)
    weight: Annotated[float, Field(gt=0, allow_inf_nan=False)] = 1
    provider_config: dict[str, JsonValue] = Field(default_factory=dict)


class EvaluatorResult(Model):
    metric: Name
    evaluator_version: Name
    status: Status
    score: Score | None = None
    confidence: Score | None = None
    explanation: str = ""
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def coherent_score(self):
        if (self.status in {"PASS", "FAIL"}) != (self.score is not None):
            raise ValueError("only PASS/FAIL results must have a score")
        return self


class CaseResult(Model):
    case: EvalCase
    target: TargetResult
    evaluations: list[EvaluatorResult]


class Run(Model):
    id: Name = Field(default_factory=lambda: str(uuid4()))
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    dataset_name: Name
    dataset_version: Name
    dataset_hash: Name
    target_name: Name
    target_config: dict[str, JsonValue] = Field(default_factory=dict)
    evaluators: list[EvaluatorSpec] = Field(min_length=1)
    cases: list[CaseResult] = Field(min_length=1)
    status: Literal["COMPLETED", "ERROR"]
    execution: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def complete_matrix(self):
        metrics = [e.metric for e in self.evaluators]
        ids = [c.case.id for c in self.cases]
        if len(set(metrics)) != len(metrics) or len(set(ids)) != len(ids):
            raise ValueError("duplicate metric or case IDs")
        versions = {e.metric: e.version for e in self.evaluators}
        for row in self.cases:
            if sorted(e.metric for e in row.evaluations) != sorted(metrics):
                raise ValueError("case evaluations must cover every metric exactly once")
            if any(e.evaluator_version != versions[e.metric] for e in row.evaluations):
                raise ValueError("evaluator version mismatch")
        has_errors = any(
            c.target.status == "ERROR" or any(e.status == "ERROR" for e in c.evaluations)
            for c in self.cases
        )
        if has_errors != (self.status == "ERROR"):
            raise ValueError("run status must reflect execution errors")
        return self


class RegressionRule(Model):
    metric: Name = "overall"
    category: Name | None = None
    min_score: Score | None = None
    max_regression: NonNegative | None = None
    max_relative_regression: NonNegative | None = None
    max_value: NonNegative | None = None
    max_increase_percent: NonNegative | None = None
    min_samples: Annotated[int, Field(ge=1)] = 1
    critical: bool = False
    severity: Literal["block", "warn"] = "block"

    @model_validator(mode="after")
    def has_gate(self):
        if not self.critical and all(
            getattr(self, name) is None
            for name in (
                "min_score",
                "max_regression",
                "max_relative_regression",
                "max_value",
                "max_increase_percent",
            )
        ):
            raise ValueError("rule must define a threshold or critical gate")
        operational = self.metric in {"latency_ms", "p95_latency_ms", "cost", "error_rate"}
        quality_fields = (self.min_score, self.max_regression, self.max_relative_regression)
        if operational and (any(v is not None for v in quality_fields) or self.critical):
            raise ValueError("operational metrics use max_value/max_increase_percent")
        if not operational and (
            self.max_value is not None or self.max_increase_percent is not None
        ):
            raise ValueError("quality metrics use min_score/max_regression")
        return self


class GateResult(Model):
    rule: RegressionRule
    status: Literal["PASS", "FAIL", "ERROR"]
    baseline: float | None = None
    candidate: float | None = None
    delta: float | None = None
    samples: int = 0
    case_ids: list[str] = Field(default_factory=list)
    explanation: str


class Comparison(Model):
    baseline_id: Name
    candidate_id: Name
    status: Literal["PASS", "FAIL", "ERROR"]
    gates: list[GateResult]


class Experiment(Model):
    id: Name = Field(default_factory=lambda: str(uuid4()))
    baseline: Run
    candidate: Run
    comparison: Comparison

    @model_validator(mode="after")
    def matching_runs(self):
        if (self.comparison.baseline_id, self.comparison.candidate_id) != (
            self.baseline.id,
            self.candidate.id,
        ):
            raise ValueError("comparison must refer to experiment runs")
        return self


class BaselineApproval(Model):
    id: Name = Field(default_factory=lambda: str(uuid4()))
    name: Name
    run_id: Name
    approved_by: Name
    approved_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    note: str = ""


class ExecutionLimits(Model):
    concurrency: Annotated[int, Field(ge=1, le=32)] = 1
    max_provider_requests: Annotated[int, Field(ge=1)] | None = None
    max_seconds: Annotated[float, Field(gt=0, allow_inf_nan=False)] | None = None
    max_observed_cost: NonNegative | None = None
