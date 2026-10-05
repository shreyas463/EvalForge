"""Strict replay of previously captured application outputs, never executing recorded tools."""

import hashlib
import json
from pathlib import Path

from pydantic import JsonValue, TypeAdapter, model_validator

from evalforge.datasets import parse_json
from evalforge.models import Model, Name, NonNegative, TargetResult


class RecordedCase(Model):
    case_id: Name
    input: str | dict[str, JsonValue]
    result: TargetResult

    @model_validator(mode="after")
    def valid_observed_latency(self):
        if "observed_latency_ms" in self.result.metadata:
            TypeAdapter(NonNegative).validate_python(
                self.result.metadata["observed_latency_ms"], strict=True
            )
        return self


def export_traces(run_path, output):
    from evalforge.storage import read_run

    run = read_run(run_path)
    path = Path(output)
    if path.exists():
        raise ValueError("trace output already exists")
    records = []
    for row in run.cases:
        result = row.target.model_copy(deep=True)
        result.metadata["source_run_id"] = run.id
        result.metadata["source_target"] = run.target_name
        result.metadata["source_dataset_hash"] = run.dataset_hash
        records.append(RecordedCase(case_id=row.case.id, input=row.case.input, result=result))
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents replacing an existing capture.
    with path.open("x", encoding="utf-8") as destination:
        for record in records:
            destination.write(record.model_dump_json() + "\n")
    return {"source_run_id": run.id, "records": len(records), "output": str(path)}


class RecordedTarget:
    def __init__(self, path):
        with Path(path).open("rb") as source:
            raw = source.read(16 * 1024 * 1024 + 1)
        if len(raw) > 16 * 1024 * 1024:
            raise ValueError("recorded traces exceed 16 MiB")
        self.sha256 = hashlib.sha256(raw).hexdigest()
        self.records = {}
        for line_number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                record = RecordedCase.model_validate(parse_json(line), strict=True)
                if record.case_id in self.records:
                    raise ValueError("duplicate case ID")
            except ValueError:
                raise ValueError(f"invalid recorded trace at line {line_number}") from None
            self.records[record.case_id] = record
        if not self.records:
            raise ValueError("recorded trace file is empty")

    def execute(self, case):
        record = self.records.get(case.id)
        # Compare typed canonical JSON so bool/int mismatches are not accepted.
        if record is None or json.dumps(record.input, sort_keys=True) != json.dumps(
            case.input, sort_keys=True
        ):
            raise ValueError("missing or mismatched recorded input")
        result = record.result.model_copy(deep=True)
        result.metadata.setdefault("observed_latency_ms", result.latency_ms)
        result.metadata["evidence_mode"] = "recorded"
        return result
