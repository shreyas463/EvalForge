"""Strict JSONL imports and content-addressed dataset versions."""

import hashlib
import json
from pathlib import Path

from pydantic import ValidationError

from evalforge.models import Dataset, EvalCase


class DatasetError(ValueError):
    pass


def parse_json(text: str):
    """Reject non-standard constants and duplicate keys rather than silently changing evidence."""

    def constant(value):
        raise ValueError(f"non-finite JSON constant: {value}")

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(text, parse_constant=constant, object_pairs_hook=pairs)


def dataset_hash(cases: list[EvalCase]) -> str:
    payload = [c.model_dump(mode="json") for c in sorted(cases, key=lambda c: c.id)]
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def load_jsonl(path: str | Path, *, name: str | None = None, version: str | None = None) -> Dataset:
    path = Path(path)
    cases = []
    seen = set()
    try:
        with path.open(encoding="utf-8") as source:
            for line_no, line in enumerate(source, 1):
                if not line.strip():
                    continue
                try:
                    case = EvalCase.model_validate(parse_json(line), strict=True)
                    if case.id in seen:
                        raise ValueError(f"duplicate case ID: {case.id}")
                    seen.add(case.id)
                    cases.append(case)
                except (ValueError, ValidationError) as exc:
                    raise DatasetError(f"{path}:{line_no}: {exc}") from exc
    except UnicodeError as exc:
        raise DatasetError(f"{path}: dataset must be UTF-8") from exc
    if not cases:
        raise DatasetError(f"{path}: empty dataset")
    return Dataset(name=name or path.stem, version=version or dataset_hash(cases), cases=cases)
