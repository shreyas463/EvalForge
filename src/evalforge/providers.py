"""Synchronous chat provider with bounded retries, request caps, and sanitized failures."""

import math
import os
import random
import time
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Protocol

import httpx
from pydantic import Field

from evalforge.budgets import CURRENT_BUDGET, BudgetExceeded
from evalforge.models import Model, NonNegative


class Completion(Model):
    text: str
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    cost: NonNegative | None = None
    attempts: int = Field(default=1, ge=1)


class Provider(Protocol):
    def complete(
        self, messages: list[dict[str, str]], *, json_output: bool = False
    ) -> Completion: ...


class ProviderError(RuntimeError):
    """Sanitized provider failure; never includes keys or response bodies."""


def retry_after_seconds(value):
    if value is None:
        return None
    try:
        seconds = float(value)
    except ValueError:
        try:
            when = parsedate_to_datetime(value)
            if when.tzinfo is None:
                return None
            seconds = (when - datetime.now(UTC)).total_seconds()
        except (ValueError, TypeError, OverflowError):
            return None
    return max(0.0, seconds) if math.isfinite(seconds) and seconds >= 0 else None


class ChatProvider:
    def __init__(
        self,
        *,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        api_key_env: str = "OPENAI_API_KEY",
        timeout: float = 60,
        temperature: float = 0,
        input_cost_per_million: float | None = None,
        output_cost_per_million: float | None = None,
        max_attempts: int = 1,
        retry_base_seconds: float = 0.5,
        retry_max_seconds: float = 10,
        max_completion_tokens: int | None = None,
        transport: httpx.BaseTransport | None = None,
    ):
        if timeout <= 0 or not math.isfinite(timeout):
            raise ValueError("timeout must be positive and finite")
        if not 1 <= max_attempts <= 10 or retry_base_seconds < 0 or retry_max_seconds < 0:
            raise ValueError("invalid retry limits")
        self.model, self.base_url, self.api_key_env = model, base_url.rstrip("/"), api_key_env
        self.timeout, self.temperature = timeout, temperature
        self.input_price, self.output_price = input_cost_per_million, output_cost_per_million
        self.max_attempts, self.retry_base, self.retry_max = (
            max_attempts,
            retry_base_seconds,
            retry_max_seconds,
        )
        self.max_completion_tokens, self.transport = max_completion_tokens, transport

    def complete(self, messages, *, json_output=False) -> Completion:
        key = os.environ.get(self.api_key_env)
        if not key:
            raise ProviderError(f"missing credential environment variable: {self.api_key_env}")
        budget = CURRENT_BUDGET.get()
        if (
            budget
            and budget.limits.max_observed_cost is not None
            and (self.input_price is None or self.output_price is None)
        ):
            raise BudgetExceeded("cost-limited provider requires configured input/output prices")
        payload = {"model": self.model, "messages": messages, "temperature": self.temperature}
        if json_output:
            payload["response_format"] = {"type": "json_object"}
        if self.max_completion_tokens is not None:
            payload["max_completion_tokens"] = self.max_completion_tokens
        # This bounds the whole completion including retry sleeps, not only individual attempts.
        deadline = time.monotonic() + self.timeout
        error = None
        for attempt in range(1, self.max_attempts + 1):
            remaining = deadline - time.monotonic()
            if budget:
                budget.check_deadline()
                experiment_remaining = budget.remaining_seconds()
                if experiment_remaining is not None:
                    remaining = min(remaining, experiment_remaining)
            if remaining <= 0:
                raise ProviderError("provider completion deadline exhausted")
            if budget:
                budget.reserve_request(retry=attempt > 1)
            hint = None
            retryable = False
            try:
                with httpx.Client(timeout=remaining, transport=self.transport) as client:
                    response = client.post(
                        f"{self.base_url}/chat/completions",
                        json=payload,
                        headers={"Authorization": f"Bearer {key}"},
                    )
                    response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                # Classify quota/billing codes without exposing the response body.
                try:
                    detail = exc.response.json().get("error", {})
                    terminal = isinstance(detail, dict) and any(
                        isinstance(detail.get(field), str)
                        and detail.get(field)
                        in {
                            "insufficient_quota",
                            "billing_hard_limit_reached",
                            "billing_not_active",
                        }
                        for field in ("code", "type")
                    )
                except (ValueError, AttributeError):
                    terminal = False
                retryable = status in {408, 429, 500, 502, 503, 504} and not terminal
                hint = retry_after_seconds(exc.response.headers.get("Retry-After"))
                error = f"provider HTTP {status}"
            except httpx.TimeoutException:
                retryable, error = True, "provider request timed out"
                if budget:
                    budget.observe_cost(None)
            except httpx.TransportError:
                retryable, error = True, "provider transport failed"
                if budget:
                    budget.observe_cost(None)
            except httpx.HTTPError:
                error = "provider transport failed"
            else:
                completion = self._parse_completion(response, attempt, budget)
                if time.monotonic() >= deadline:
                    raise ProviderError("provider completion deadline exhausted")
                return completion
            if not retryable or attempt == self.max_attempts:
                raise ProviderError(error) from None
            delay = (
                hint
                if hint is not None
                else random.uniform(0, min(self.retry_max, self.retry_base * 2 ** (attempt - 1)))
            )
            remaining = deadline - time.monotonic()
            if budget and budget.remaining_seconds() is not None:
                remaining = min(remaining, budget.remaining_seconds())
            # Never retry sooner than Retry-After; decline delays outside configured limits.
            if delay > self.retry_max or delay >= remaining:
                raise ProviderError(
                    f"{error}; retry delay exceeds available deadline/limit"
                ) from None
            time.sleep(delay)
        raise ProviderError(error or "provider failed")

    def _parse_completion(self, response, attempt, budget):
        accounted = False
        try:
            data = response.json()
            usage = data.get("usage") or {}
            inputs, outputs = usage.get("prompt_tokens"), usage.get("completion_tokens")
            cost = None
            if (
                inputs is not None
                and outputs is not None
                and self.input_price is not None
                and self.output_price is not None
            ):
                cost = (inputs * self.input_price + outputs * self.output_price) / 1_000_000
            # Account for usage even if the generated answer is truncated or unusable.
            usage_result = Completion(
                text="", input_tokens=inputs, output_tokens=outputs, cost=cost
            )
            if budget:
                budget.observe_cost(usage_result.cost)
                accounted = True
            choice = data["choices"][0]
            if choice.get("finish_reason") != "stop":
                raise ValueError("incomplete generation")
            return Completion(
                text=choice["message"]["content"],
                input_tokens=inputs,
                output_tokens=outputs,
                cost=cost,
                attempts=attempt,
            )
        except (ValueError, TypeError, KeyError, IndexError, AttributeError):
            if budget and not accounted:
                # A missing/invalid usage record cannot be treated as a free call.
                budget.observe_cost(None)
            raise ProviderError("provider returned invalid or incomplete completion") from None
