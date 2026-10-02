"""Run the reviewed-baseline → changed application → blocked-regression demonstration."""

import argparse
import json
import os
from pathlib import Path
from uuid import uuid4

from evalforge.cli import main as evalforge
from evalforge.storage import write_json


def run_demo(
    *,
    examples_dir: Path,
    output_dir: Path,
    live=False,
    model=None,
    judge_model=None,
    base_url="https://api.openai.com/v1",
    api_key_env="OPENAI_API_KEY",
) -> int:
    if live and (not model or not judge_model or not os.environ.get(api_key_env)):
        print("Live demo requires --model, --judge-model, and a configured credential variable.")
        return 2
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    session = output_dir / ("support-demo-" + str(uuid4()))
    session.mkdir()
    template = "live.json" if live else "offline.json"
    config = json.loads((examples_dir / template).read_text(encoding="utf-8"))
    config["dataset"] = str((examples_dir / "cases.jsonl").resolve())
    if live:
        for target in ("baseline", "candidate"):
            config[target]["provider"].update(
                model=model, base_url=base_url, api_key_env=api_key_env
            )
            config[target]["system_prompt_file"] = str(
                (examples_dir / config[target]["system_prompt_file"]).resolve()
            )
        config["judge_provider"].update(
            model=judge_model, base_url=base_url, api_key_env=api_key_env
        )
    candidate_config = session / "candidate-config.json"
    write_json(candidate_config, config)
    seed = json.loads(json.dumps(config))
    seed["candidate"] = seed["baseline"].copy()
    seed_config = session / "baseline-config.json"
    write_json(seed_config, seed)
    database_url = f"sqlite:///{session / 'baseline.db'}"
    args = ["--output-dir", str(output_dir), "--database-url", database_url]
    before = set(output_dir.iterdir())
    seed_exit = evalforge(["run", "--config", str(seed_config), *args])
    if seed_exit != 0:
        print(f"Baseline validation did not pass (exit {seed_exit}); no baseline was approved.")
        return seed_exit
    created = [p for p in set(output_dir.iterdir()) - before if (p / "baseline.json").exists()]
    if len(created) != 1:
        raise RuntimeError("unable to locate baseline evidence")
    baseline_file = created[0] / "baseline.json"
    approval_exit = evalforge(
        [
            "baseline",
            "approve",
            "--run",
            str(baseline_file),
            "--name",
            "support-reviewed",
            "--approved-by",
            "support-demo",
            "--note",
            "Validated demo policy baseline",
            "--database-url",
            database_url,
        ]
    )
    if approval_exit != 0:
        return approval_exit
    before_candidate = set(output_dir.iterdir())
    candidate_exit = evalforge(
        ["run", "--config", str(candidate_config), "--baseline-name", "support-reviewed", *args]
    )
    if candidate_exit != 1:
        print(f"Expected a blocked policy regression (exit 1); observed {candidate_exit}.")
        return 3
    reports = [
        p / "comparison.json"
        for p in set(output_dir.iterdir()) - before_candidate
        if (p / "comparison.json").exists()
    ]
    if len(reports) != 1:
        raise RuntimeError("unable to locate candidate comparison evidence")
    gates = json.loads(reports[0].read_text(encoding="utf-8"))["gates"]
    if not any(
        g["status"] == "FAIL"
        and g["rule"]["category"] == "refunds"
        and set(g["case_ids"]) & {"refund_window", "refund_late"}
        for g in gates
    ):
        print("Candidate failed, but the expected refund-policy regression was not established.")
        return 3
    print("Support demo verified: approved 14-day baseline, blocked changed 30-day policy.")
    print(f"Demo configuration and registry: {session}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples-dir", type=Path, default=Path("examples/support"))
    parser.add_argument("--output-dir", type=Path, default=Path(".evalforge"))
    parser.add_argument(
        "--live", action="store_true", help="make paid model/judge calls; requires credentials"
    )
    parser.add_argument("--model")
    parser.add_argument("--judge-model")
    parser.add_argument("--base-url", default="https://api.openai.com/v1")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    return run_demo(**vars(parser.parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
