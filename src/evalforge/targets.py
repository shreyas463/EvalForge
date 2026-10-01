"""Targets share one result contract; local callables enable application evaluation."""

import json
from collections.abc import Callable
from typing import Protocol

from evalforge.models import EvalCase, TargetResult
from evalforge.providers import Provider


class Target(Protocol):
    def execute(self, case: EvalCase) -> TargetResult: ...


class MockTarget:
    """Explicit fixture outputs; missing fixtures raise an error."""

    def __init__(self, responses: dict[str, str]):
        self.responses = responses

    def execute(self, case):
        if case.id not in self.responses:
            raise ValueError(f"no mock response for case {case.id}")
        return TargetResult(output=self.responses[case.id])


class LocalTarget:
    def __init__(self, function: Callable[[EvalCase], str | TargetResult]):
        self.function = function

    def execute(self, case):
        result = self.function(case)
        return result if isinstance(result, TargetResult) else TargetResult(output=result)


class LLMTarget:
    def __init__(self, provider: Provider, system_prompt: str = "You are a helpful assistant."):
        self.provider = provider
        self.system_prompt = system_prompt

    def execute(self, case):
        content = case.input if isinstance(case.input, str) else json.dumps(case.input)
        result = self.provider.complete(
            [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": content},
            ]
        )
        return TargetResult(
            output=result.text,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            cost=result.cost,
        )
