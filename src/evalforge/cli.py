"""CLI: 0 pass, 1 blocking regression, 2 invalid input, 3 execution/storage error."""

import argparse
import json
import os
import sys
from pathlib import Path
from uuid import uuid4

from pydantic import ValidationError
from sqlalchemy import create_engine

from evalforge.budgets import RunBudget
from evalforge.config import build_evaluators, build_target, load_config
from evalforge.datasets import DatasetError, dataset_hash, load_jsonl
from evalforge.migrations import HEAD, current_revision
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
        selected = args.baseline_name or args.baseline_id
        baseline = None
        if selected:
            if not args.database_url:
                raise ValueError("approved baseline selection requires a database URL")
            store = SQLStore(args.database_url)
            try:
                baseline = store.resolve_baseline(name=args.baseline_name, run_id=args.baseline_id)
            finally:
                store.close()
            if (baseline.dataset_name, baseline.dataset_version, baseline.dataset_hash) != (
                dataset.name,
                dataset.version,
                dataset_hash(dataset.cases),
            ):
                raise ComparisonError("approved baseline dataset differs from configured dataset")

            def canonical(specs):
                return sorted(json.dumps(e.model_dump(mode="json"), sort_keys=True) for e in specs)

            if canonical(baseline.evaluators) != canonical([e.spec for e in evaluators]):
                raise ComparisonError("approved baseline evaluator configuration differs")
        else:
            baseline_target, baseline_config = build_target(config.baseline, base_dir=base)
        candidate_target, candidate_config = build_target(config.candidate, base_dir=base)
        budget = RunBudget(config.execution)
        if baseline is None:
            baseline = run_experiment(
                dataset,
                baseline_target,
                evaluators,
                target_name=config.baseline.name,
                target_config=baseline_config,
                budget=budget,
            )
        candidate = run_experiment(
            dataset,
            candidate_target,
            evaluators,
            target_name=config.candidate.name,
            target_config=candidate_config,
            budget=budget,
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
    dashboard = sub.add_parser("dashboard", help="view saved experiments in a local browser")
    dashboard.add_argument(
        "--config",
        action="append",
        default=[],
        help="trusted config to enable for browser execution; repeat for multiple",
    )
    dashboard.add_argument("--artifact-root", default=".evalforge")
    dashboard.add_argument("--port", type=int, default=8765)
    dashboard.add_argument("--open", action="store_true", dest="open_browser")
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
        if command == "run":
            group = p.add_mutually_exclusive_group()
            group.add_argument("--baseline-name", help="use the latest approved named baseline")
            group.add_argument("--baseline-id", help="use an approved stored run ID")
        if command == "compare":
            p.add_argument("--baseline", required=True)
            p.add_argument("--candidate", required=True)
    baseline_parser = sub.add_parser("baseline", help="approve and inspect stored baselines")
    baseline_commands = baseline_parser.add_subparsers(dest="baseline_command", required=True)
    approve = baseline_commands.add_parser("approve")
    source = approve.add_mutually_exclusive_group(required=True)
    source.add_argument("--run", help="saved run JSON file")
    source.add_argument("--run-id", help="stored run ID")
    approve.add_argument("--name", required=True)
    approve.add_argument("--approved-by", required=True, help="audit identity, not authentication")
    approve.add_argument("--note", default="")
    show = baseline_commands.add_parser("show")
    show.add_argument("--name", required=True)
    history = baseline_commands.add_parser("history")
    history.add_argument("--name", required=True)
    for p in (approve, show, history):
        p.add_argument("--database-url")
    db = sub.add_parser("db", help="upgrade or inspect database migrations")
    db.add_argument("action", choices=["upgrade", "status"])
    db.add_argument("--database-url")
    args = parser.parse_args(argv)
    try:
        if args.command == "dashboard":
            from evalforge.dashboard import serve

            if not 0 <= args.port <= 65535:
                raise ValueError("port must be between 0 and 65535")
            return serve(
                args.artifact_root,
                port=args.port,
                open_browser=args.open_browser,
                configs=args.config,
            )
        if args.command == "validate":
            dataset = load_jsonl(args.dataset)
            print(f"Valid: {dataset.name}, {len(dataset.cases)} cases, version={dataset.version}")
            return 0

        args.database_url = args.database_url or os.environ.get("EVALFORGE_DATABASE_URL")
        if args.command in {"baseline", "db"}:
            if not args.database_url:
                raise ValueError("database URL required")
            if args.command == "db" and args.action == "status":
                engine = create_engine(args.database_url, hide_parameters=True)
                try:
                    revision = current_revision(engine)
                finally:
                    engine.dispose()
                print(f"Database revision: {revision or 'unversioned'} (expected {HEAD})")
                return 0 if revision == HEAD else 3
            store = SQLStore(args.database_url)
            try:
                if args.command == "db":
                    print(f"Database upgraded to {HEAD}")
                elif args.baseline_command == "approve":
                    run = read_run(args.run) if args.run else store.load_run(args.run_id)
                    approval = store.approve_baseline(
                        run, name=args.name, approved_by=args.approved_by, note=args.note
                    )
                    print(approval.model_dump_json(indent=2))
                elif args.baseline_command == "show":
                    print(store.resolve_baseline(name=args.name).model_dump_json(indent=2))
                else:
                    print(
                        json.dumps(
                            [a.model_dump(mode="json") for a in store.baseline_history(args.name)],
                            indent=2,
                        )
                    )
            finally:
                store.close()
            return 0
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
