"""CLI: 0 pass, 1 blocking regression, 2 invalid input, 3 execution/storage error."""

import argparse
import os
import sys
from pathlib import Path
from uuid import uuid4

from pydantic import ValidationError

from evalforge.config import build_evaluators, build_target, load_config
from evalforge.datasets import DatasetError, load_jsonl
from evalforge.models import Experiment
from evalforge.regression import ComparisonError, compare
from evalforge.report import render_report
from evalforge.runner import run_experiment
from evalforge.scoring import aggregate
from evalforge.storage import SQLStore, StorageError, read_run, write_json

EXIT_CODES = {"PASS": 0, "FAIL": 1, "ERROR": 3}


def _persist(baseline, candidate, comparison, directory, database_url=None):
    experiment = Experiment(baseline=baseline, candidate=candidate, comparison=comparison)
    directory.mkdir(parents=True, exist_ok=False)
    for name, payload in [
        ("baseline", baseline),
        ("candidate", candidate),
        ("comparison", comparison),
        ("experiment", experiment),
    ]:
        write_json(directory / f"{name}.json", payload.model_dump(mode="json"))
    write_json(
        directory / "metrics.json",
        {
            label: {
                group: {metric: summary.model_dump() for metric, summary in metrics.items()}
                for group, metrics in aggregate(run).items()
            }
            for label, run in [("baseline", baseline), ("candidate", candidate)]
        },
    )
    report = render_report(baseline, candidate, comparison)
    (directory / "report.txt").write_text(report, encoding="utf-8")
    if database_url:
        store = SQLStore(database_url)
        try:
            store.save_experiment(experiment)
        finally:
            store.close()
    print(report, end="")
    print(f"Artifacts: {directory}")
    return EXIT_CODES[comparison.status]


def _execute(args):
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    base = config_path.parent
    if args.command == "compare":
        baseline, candidate = read_run(args.baseline), read_run(args.candidate)
    else:
        dataset = load_jsonl(
            base / config.dataset, name=config.dataset_name, version=config.dataset_version
        )
        evaluators = build_evaluators(config)
        # Build both before execution to catch configuration mistakes without spending model calls.
        baseline_target, baseline_config = build_target(config.baseline)
        candidate_target, candidate_config = build_target(config.candidate)
        baseline = run_experiment(
            dataset,
            baseline_target,
            evaluators,
            target_name=config.baseline.name,
            target_config=baseline_config,
        )
        candidate = run_experiment(
            dataset,
            candidate_target,
            evaluators,
            target_name=config.candidate.name,
            target_config=candidate_config,
        )
    comparison = compare(baseline, candidate, config.gates)
    directory = Path(args.output_dir).resolve() if args.output_dir else base / config.output_dir
    directory = directory / str(uuid4())
    return _persist(baseline, candidate, comparison, directory, args.database_url)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="evalforge", description="Evaluate and gate paired AI targets"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate", help="validate a JSONL dataset")
    validate.add_argument("dataset")
    for command in ("run", "compare"):
        p = sub.add_parser(
            command, help="run paired targets" if command == "run" else "compare saved runs"
        )
        p.add_argument("--config", required=True)
        p.add_argument("--output-dir")
        p.add_argument(
            "--database-url",
            help="SQLAlchemy URL; prefer EVALFORGE_DATABASE_URL environment variable",
        )
        if command == "compare":
            p.add_argument("--baseline", required=True)
            p.add_argument("--candidate", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            dataset = load_jsonl(args.dataset)
            print(f"Valid: {dataset.name}, {len(dataset.cases)} cases, version={dataset.version}")
            return 0

        args.database_url = args.database_url or os.environ.get("EVALFORGE_DATABASE_URL")
        return _execute(args)
    except (
        DatasetError,
        ComparisonError,
        ValidationError,
        ValueError,
        KeyError,
        ImportError,
        AttributeError,
        TypeError,
    ) as exc:
        # Validation errors may echo payloads; do not print entire model input.
        detail = (
            "; ".join(
                f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
                for error in exc.errors(include_input=False)[:5]
            )
            if isinstance(exc, ValidationError)
            else str(exc)
        )
        print(f"Invalid input: {detail}", file=sys.stderr)
        return 2
    except (OSError, StorageError) as exc:
        detail = str(exc) if isinstance(exc, StorageError) else type(exc).__name__
        print(f"Execution/storage error: {detail}", file=sys.stderr)
        return 3
    except Exception as exc:
        print(f"Execution error: {type(exc).__name__}", file=sys.stderr)
        return 3
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
