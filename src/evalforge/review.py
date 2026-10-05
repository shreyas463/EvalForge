"""Human label export and disagreement reports over saved runs; never invoke a judge."""

from pathlib import Path

from evalforge.datasets import parse_json
from evalforge.models import Model, Name, Score
from evalforge.storage import read_run, write_json


class ReviewLabel(Model):
    run_id: Name
    case_id: Name
    metric: Name
    human_score: Score | None = None
    reviewer: str = ""
    note: str = ""


def export_review(run_path, destination, *, metric):
    run = read_run(run_path)
    if metric not in {spec.metric for spec in run.evaluators}:
        raise ValueError("unknown metric")
    folder = Path(destination)
    if folder.exists():
        raise ValueError("review destination already exists")
    labels, evidence = [], []
    for row in run.cases:
        labels.append(ReviewLabel(run_id=run.id, case_id=row.case.id, metric=metric))
        evidence.append(
            {
                "case_id": row.case.id,
                "input": row.case.input,
                "reference_answer": row.case.reference_answer,
                "expected_facts": row.case.expected_facts,
                "forbidden_facts": row.case.forbidden_facts,
                "agent_rules": row.case.metadata.get("agent_rules"),
                "target": row.target.model_dump(mode="json"),
            }
        )
    folder.mkdir(parents=True)
    (folder / "labels.jsonl").write_text(
        "".join(label.model_dump_json() + "\n" for label in labels), encoding="utf-8"
    )
    # Automatic scores are deliberately omitted to support independent review.
    write_json(folder / "evidence.json", {"run_id": run.id, "metric": metric, "cases": evidence})
    return {"run_id": run.id, "metric": metric, "cases": len(labels), "directory": str(folder)}


def calibration_report(run_path, labels_path, *, threshold=0.5):
    if isinstance(threshold, bool) or not 0 <= threshold <= 1:
        raise ValueError("human pass threshold must be within 0..1")
    run = read_run(run_path)
    rows = {row.case.id: row for row in run.cases}
    specs = {spec.metric: spec for spec in run.evaluators}
    seen, groups = set(), {}
    with Path(labels_path).open(encoding="utf-8") as source:
        for number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                label = ReviewLabel.model_validate(parse_json(line), strict=True)
            except ValueError:
                raise ValueError(f"invalid review label at line {number}") from None
            key = (label.case_id, label.metric)
            if (
                label.run_id != run.id
                or label.case_id not in rows
                or label.metric not in specs
                or key in seen
            ):
                raise ValueError(f"unknown, mismatched or duplicate review label at line {number}")
            seen.add(key)
            group = groups.setdefault(
                label.metric,
                {
                    "supplied": 0,
                    "reviewed": 0,
                    "unreviewed": 0,
                    "unscored": 0,
                    "paired": 0,
                    "agreements": 0,
                    "absolute_error": 0,
                    "disagreements": [],
                    "labels": [],
                },
            )
            group["supplied"] += 1
            if label.human_score is None:
                group["unreviewed"] += 1
                continue
            if not label.reviewer.strip():
                raise ValueError(f"scored review needs a reviewer at line {number}")
            group["reviewed"] += 1
            group["labels"].append(label.model_dump(mode="json"))
            evaluation = next(
                e for e in rows[label.case_id].evaluations if e.metric == label.metric
            )
            if evaluation.score is None:
                group["unscored"] += 1
                continue
            group["paired"] += 1
            group["absolute_error"] += abs(label.human_score - evaluation.score)
            agrees = (label.human_score >= threshold) == (evaluation.status == "PASS")
            group["agreements"] += int(agrees)
            if not agrees:
                group["disagreements"].append(
                    {
                        "case_id": label.case_id,
                        "human_score": label.human_score,
                        "automatic_score": evaluation.score,
                        "automatic_status": evaluation.status,
                        "reviewer": label.reviewer,
                        "note": label.note,
                    }
                )
    if not seen:
        raise ValueError("review file is empty")
    for group in groups.values():
        paired = group["paired"]
        group["status_agreement"] = group["agreements"] / paired if paired else None
        group["mean_absolute_error"] = group.pop("absolute_error") / paired if paired else None
        group["missing_labels"] = len(rows) - group["supplied"]
        group["review_coverage"] = group["reviewed"] / len(rows)
    return {
        "run_id": run.id,
        "dataset_hash": run.dataset_hash,
        "human_pass_threshold": threshold,
        "metrics": groups,
        "limitations": "Reviewer identity is self-reported. Agreement is descriptive, not proof "
        "of correctness or a statistically validated judge calibration.",
    }
