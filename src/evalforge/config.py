"""Strict JSON experiment configuration and explicit component construction."""

import hashlib
import importlib
from functools import partial
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import urlsplit

from pydantic import Field, JsonValue, model_validator

from evalforge.agents import AGENT_KINDS, AgentEvaluator
from evalforge.datasets import parse_json
from evalforge.evaluators import DeterministicEvaluator, LLMJudge, RAGEvaluator
from evalforge.models import (
    EvaluatorSpec,
    ExecutionLimits,
    Model,
    Name,
    NonNegative,
    RegressionRule,
)
from evalforge.providers import ChatProvider
from evalforge.rag import SYSTEM_PROMPT, BM25Retriever, RAGTarget, RetrievalTarget
from evalforge.runner import RESERVED_METRICS
from evalforge.targets import LLMTarget, LocalTarget, MockTarget
from evalforge.traces import RecordedTarget


class ProviderConfig(Model):
    model: Name
    base_url: str = "https://api.openai.com/v1"
    api_key_env: Name = "OPENAI_API_KEY"
    timeout: Annotated[float, Field(gt=0, allow_inf_nan=False)] = 60
    temperature: Annotated[float, Field(ge=0, le=2, allow_inf_nan=False)] = 0
    max_attempts: Annotated[int, Field(ge=1, le=10)] = 1
    retry_base_seconds: NonNegative = 0.5
    retry_max_seconds: NonNegative = 10
    max_completion_tokens: Annotated[int, Field(ge=1)] | None = None
    input_cost_per_million: NonNegative | None = None
    output_cost_per_million: NonNegative | None = None

    @model_validator(mode="after")
    def valid_endpoint(self):
        url = urlsplit(self.base_url)
        if (
            url.scheme not in {"http", "https"}
            or not url.hostname
            or url.username
            or url.password
            or url.query
            or url.fragment
        ):
            raise ValueError(
                "base_url must be an HTTP(S) endpoint without credentials/query/fragment"
            )
        if (self.input_cost_per_million is None) != (self.output_cost_per_million is None):
            raise ValueError("provide both input and output prices, or neither")
        return self


class MockConfig(Model):
    kind: Literal["mock"]
    name: Name
    responses: dict[str, str]


class LocalConfig(Model):
    kind: Literal["local"]
    name: Name
    callable: Name
    parameters: dict[str, JsonValue] = Field(default_factory=dict)


class ChatConfig(Model):
    kind: Literal["chat"]
    name: Name
    provider: ProviderConfig
    system_prompt: str = "You are a helpful assistant."
    system_prompt_file: Name | None = None

    @model_validator(mode="after")
    def one_prompt(self):
        if self.system_prompt_file and self.system_prompt != "You are a helpful assistant.":
            raise ValueError("use either system_prompt or system_prompt_file")
        return self


class RecordedConfig(Model):
    kind: Literal["recorded"]
    name: Name
    records: Name


class RetrievalConfig(Model):
    kind: Literal["retrieval"]
    name: Name
    documents: Name
    top_k: Annotated[int, Field(ge=1, le=100)] = 5
    chunk_words: Annotated[int, Field(ge=1, le=2000)] = 180


class RAGConfig(Model):
    kind: Literal["rag"]
    name: Name
    documents: Name
    provider: ProviderConfig
    top_k: Annotated[int, Field(ge=1, le=100)] = 3
    chunk_words: Annotated[int, Field(ge=1, le=2000)] = 180


TargetConfig = Annotated[
    MockConfig | LocalConfig | ChatConfig | RAGConfig | RetrievalConfig | RecordedConfig,
    Field(discriminator="kind"),
]


class ExperimentConfig(Model):
    dataset: Name
    dataset_name: Name | None = None
    dataset_version: Name | None = None
    baseline: TargetConfig
    candidate: TargetConfig
    evaluators: list[EvaluatorSpec] = Field(min_length=1)
    judge_provider: ProviderConfig | None = None
    gates: list[RegressionRule] = Field(min_length=1)
    output_dir: Name = ".evalforge"
    execution: ExecutionLimits = Field(default_factory=ExecutionLimits)

    @model_validator(mode="after")
    def consistent_metrics(self):
        names = [e.metric for e in self.evaluators]
        if len(set(names)) != len(names) or set(names) & RESERVED_METRICS:
            raise ValueError("metrics must be unique and not reserved")
        if (
            any(e.kind in {"judge", "faithfulness"} for e in self.evaluators)
            and self.judge_provider is None
        ):
            raise ValueError("judge evaluator requires judge_provider")
        if any(g.metric not in set(names) | RESERVED_METRICS for g in self.gates):
            raise ValueError("gate references unknown metric")
        return self


def load_config(path: str | Path) -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        parse_json(Path(path).read_text(encoding="utf-8")), strict=True
    )


def build_target(config: TargetConfig, *, base_dir: Path | None = None):
    snapshot = config.model_dump(mode="json")
    if isinstance(config, MockConfig):
        return MockTarget(config.responses), snapshot
    if isinstance(config, RecordedConfig):
        target = RecordedTarget((base_dir or Path.cwd()) / config.records)
        snapshot["records_sha256"] = target.sha256
        snapshot["evidence_mode"] = "recorded"
        return target, snapshot
    if isinstance(config, (RAGConfig, RetrievalConfig)):
        retriever = BM25Retriever(
            (base_dir or Path.cwd()) / config.documents, chunk_words=config.chunk_words
        )
        snapshot["corpus_hash"] = retriever.corpus_hash
        snapshot["retriever"] = "bm25-v1"
        if isinstance(config, RetrievalConfig):
            return RetrievalTarget(retriever, top_k=config.top_k), snapshot
        snapshot["system_prompt"] = SYSTEM_PROMPT
        return RAGTarget(
            retriever, ChatProvider(**config.provider.model_dump()), top_k=config.top_k
        ), snapshot
    if isinstance(config, ChatConfig):
        prompt = config.system_prompt
        if config.system_prompt_file:
            path = (base_dir or Path.cwd()) / config.system_prompt_file
            prompt = path.read_text(encoding="utf-8")
            snapshot["system_prompt"] = prompt
            snapshot["system_prompt_sha256"] = hashlib.sha256(prompt.encode()).hexdigest()
        return LLMTarget(ChatProvider(**config.provider.model_dump()), prompt), snapshot
    module_name, separator, attribute = config.callable.partition(":")
    if not separator or not attribute:
        raise ValueError("local callable must be module:function")
    module = importlib.import_module(module_name)
    function = getattr(module, attribute)
    if not callable(function):
        raise ValueError("local target entry point must be callable")
    file = getattr(module, "__file__", None)
    if file:
        snapshot["module_hash"] = hashlib.sha256(Path(file).read_bytes()).hexdigest()
    return LocalTarget(partial(function, **config.parameters)), snapshot


def build_evaluators(config: ExperimentConfig):
    evaluators = []
    for original in config.evaluators:
        spec = original.model_copy(deep=True)
        if spec.kind in {"judge", "faithfulness"}:
            spec.provider_config = config.judge_provider.model_dump(mode="json")
            evaluators.append(LLMJudge(spec, ChatProvider(**config.judge_provider.model_dump())))
        elif spec.kind in AGENT_KINDS:
            evaluators.append(AgentEvaluator(spec))
        elif spec.kind in {"retrieval_recall", "retrieval_precision", "citation_validity"}:
            evaluators.append(RAGEvaluator(spec))
        else:
            evaluators.append(DeterministicEvaluator(spec))
    return evaluators
