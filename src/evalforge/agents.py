"""Observable tool policy checks; no model or inference about unstated task success."""

import json
from typing import Annotated

from pydantic import Field, JsonValue, model_validator

from evalforge.models import EvaluatorResult, Model, Name

AGENT_KINDS = {
    "required_tools",
    "forbidden_tools",
    "tool_order",
    "tool_arguments",
    "tool_efficiency",
    "tool_recovery",
    "task_completion",
}


class AgentRules(Model):
    required_tools: list[Name] = Field(default_factory=list)
    forbidden_tools: list[Name] = Field(default_factory=list)
    ordered_tools: list[Name] = Field(default_factory=list)
    expected_arguments: dict[str, dict[str, JsonValue]] = Field(default_factory=dict)
    max_calls: Annotated[int, Field(ge=0)] | None = None

    @model_validator(mode="after")
    def consistent_tools(self):
        if set(self.required_tools + self.ordered_tools) & set(self.forbidden_tools):
            raise ValueError("required/ordered tools cannot be forbidden")
        if any(not tool.strip() for tool in self.expected_arguments):
            raise ValueError("argument rules need tool names")
        return self


class AgentEvaluator:
    def __init__(self, spec):
        if spec.kind not in AGENT_KINDS or spec.options:
            raise ValueError("agent checks use case.metadata.agent_rules without options")
        self.spec = spec

    def result(self, status, reason, **metadata):
        return EvaluatorResult(
            metric=self.spec.metric,
            evaluator_version=self.spec.version,
            status=status,
            score=float(status == "PASS") if status in {"PASS", "FAIL"} else None,
            explanation=reason,
            metadata=metadata,
        )

    def evaluate(self, case, target):
        if target.status != "SUCCESS":
            return self.result("SKIPPED", "target execution failed")
        trace = target.trajectory
        if trace is None:
            return self.result("ERROR", "missing tool trajectory evidence")
        if self.spec.kind == "task_completion":
            if trace.task_completed is None:
                return self.result("UNKNOWN", "no externally reported task completion")
            return self.result(
                "PASS" if trace.task_completed else "FAIL",
                "Externally reported completion flag; not semantic answer validation.",
            )
        if self.spec.kind == "tool_recovery":
            failures = [
                call.id
                for index, call in enumerate(trace.calls)
                if call.status == "ERROR"
                and not any(
                    later.tool == call.tool and later.status == "SUCCESS"
                    for later in trace.calls[index + 1 :]
                )
            ]
            return self.result(
                "FAIL" if failures else "PASS",
                "Every failed tool must have a later successful call to that tool.",
                unrecovered_call_ids=failures,
            )
        raw = case.metadata.get("agent_rules")
        if raw is None:
            return self.result("SKIPPED", "no tool policy supplied for this case")
        try:
            rules = AgentRules.model_validate(raw, strict=True)
        except ValueError:
            return self.result("ERROR", "invalid agent_rules policy")
        tools = [call.tool for call in trace.calls]
        successful = [call.tool for call in trace.calls if call.status == "SUCCESS"]
        kind = self.spec.kind
        if kind == "required_tools":
            if not rules.required_tools:
                return self.result("SKIPPED", "no required tools configured")
            violations = sorted(set(rules.required_tools) - set(successful))
            reason = "Required tools must complete successfully."
        elif kind == "forbidden_tools":
            if not rules.forbidden_tools:
                return self.result("SKIPPED", "no forbidden tools configured")
            violations = sorted(set(rules.forbidden_tools) & set(tools))
            reason = "Forbidden tools include failed attempts."
        elif kind == "tool_order":
            if not rules.ordered_tools:
                return self.result("SKIPPED", "no ordered tools configured")
            position = 0
            for tool in successful:
                if position < len(rules.ordered_tools) and tool == rules.ordered_tools[position]:
                    position += 1
            violations = rules.ordered_tools[position:]
            reason = "Successful calls must contain the required ordered subsequence."
        elif kind == "tool_arguments":
            if not rules.expected_arguments:
                return self.result("SKIPPED", "no expected arguments configured")
            violations = []
            for tool, expected in rules.expected_arguments.items():
                calls = [call for call in trace.calls if call.tool == tool]
                if not calls:
                    violations.append(f"missing tool: {tool}")
                for call in calls:
                    for key, value in expected.items():
                        if key not in call.arguments or json.dumps(
                            call.arguments[key], sort_keys=True
                        ) != json.dumps(value, sort_keys=True):
                            violations.append(f"{call.id}: {key}")
            reason = "Every attempt must match the specified top-level argument values."
        else:
            if rules.max_calls is None:
                return self.result("SKIPPED", "no call limit configured")
            violations = ["call limit exceeded"] if len(tools) > rules.max_calls else []
            reason = f"Observed {len(tools)} tool calls; limit {rules.max_calls}."
        return self.result("FAIL" if violations else "PASS", reason, violations=violations)
