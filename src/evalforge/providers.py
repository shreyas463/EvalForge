"""Provider boundary and a synchronous OpenAI-compatible chat HTTP adapter.

No retries: callers retain one invocation per case, including failed requests.
"""

import os
from typing import Protocol

import httpx
from pydantic import Field

from evalforge.models import Model, NonNegative


class Completion(Model):
    text: str
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    cost: NonNegative | None = None


class Provider(Protocol):
    def complete(
        self, messages: list[dict[str, str]], *, json_output: bool = False
    ) -> Completion: ...


class ProviderError(RuntimeError):
    """Sanitized provider failure; never includes keys or response bodies."""


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
        transport: httpx.BaseTransport | None = None,
    ):
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env
        self.timeout = timeout
        self.temperature = temperature
        self.input_price = input_cost_per_million
        self.output_price = output_cost_per_million
        self.transport = transport

    def complete(self, messages, *, json_output=False) -> Completion:
        key = os.environ.get(self.api_key_env)
        if not key:
            raise ProviderError(f"missing credential environment variable: {self.api_key_env}")
        payload = {"model": self.model, "messages": messages, "temperature": self.temperature}
        if json_output:
            payload["response_format"] = {"type": "json_object"}
        try:
            with httpx.Client(timeout=self.timeout, transport=self.transport) as client:
                response = client.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers={"Authorization": f"Bearer {key}"},
                )
                response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise ProviderError(f"provider HTTP {exc.response.status_code}") from None
        except httpx.TimeoutException:
            raise ProviderError("provider request timed out") from None
        except httpx.HTTPError:
            raise ProviderError("provider transport failed") from None
        try:
            data = response.json()
            choice = data["choices"][0]
            if choice.get("finish_reason") != "stop":
                raise ValueError("incomplete generation")
            text = choice["message"]["content"]
            usage = data.get("usage") or {}
            cost = None
            inputs, outputs = usage.get("prompt_tokens"), usage.get("completion_tokens")
            if (
                inputs is not None
                and outputs is not None
                and self.input_price is not None
                and self.output_price is not None
            ):
                cost = (inputs * self.input_price + outputs * self.output_price) / 1_000_000
            return Completion(text=text, input_tokens=inputs, output_tokens=outputs, cost=cost)
        except (ValueError, TypeError, KeyError, IndexError):
            raise ProviderError("provider returned invalid or incomplete completion") from None
