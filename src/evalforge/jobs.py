"""Bounded local evaluation jobs for operator-registered configuration files.

Each job runs the installed CLI in its own process. State survives dashboard restarts;
unfinished jobs are reported as interrupted, never silently restarted or rerun.
"""

import hashlib
import os
import sqlite3
import subprocess
import sys
import threading
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

from evalforge.config import ChatConfig, RAGConfig, build_evaluators, build_target, load_config
from evalforge.datasets import load_jsonl, parse_json
from evalforge.models import Experiment
from evalforge.storage import write_json


class JobBusy(RuntimeError):
    pass


def now():
    return datetime.now(UTC).isoformat()


class JobManager:
    def __init__(self, root, configs=()):
        self.root = Path(root).resolve()
        self.directory = self.root / "_jobs"
        if self.directory.is_symlink():
            raise ValueError("job directory must not be a symlink")
        self.configs = {str(index): Path(path).resolve() for index, path in enumerate(configs)}
        self.lock = threading.Lock()
        self.active = None
        self.process = None
        self.worker = None
        self.closed = False
        self.ownership = None
        if configs:
            self.directory.mkdir(parents=True, exist_ok=True)
            if self.directory.is_symlink() or (self.directory / "worker-lock.db").is_symlink():
                raise ValueError("job directory must not be a symlink")
            self.ownership = sqlite3.connect(
                self.directory / "worker-lock.db", timeout=0, check_same_thread=False
            )
            try:
                self.ownership.execute("BEGIN IMMEDIATE")
            except sqlite3.OperationalError:
                self.ownership.close()
                raise ValueError("another execution dashboard owns this artifact root") from None
        if configs and self.directory.exists():
            for path in self.directory.glob("*/job.json"):
                if path.is_symlink() or path.parent.is_symlink():
                    continue
                try:
                    job = parse_json(path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    continue
                if not isinstance(job, dict):
                    continue
                if job.get("status") in {"QUEUED", "RUNNING"}:
                    job.update(
                        status="INTERRUPTED",
                        finished_at=now(),
                        message=(
                            "Dashboard restarted before completion. Check saved results; "
                            "check for a surviving worker process before rerunning explicitly."
                        ),
                    )
                    write_json(path, job)

    def profiles(self):
        return [self.profile(identifier) for identifier in self.configs]

    def profile(self, identifier, *, config=None):
        if identifier not in self.configs:
            raise KeyError("unknown registered configuration")
        path = self.configs[identifier]
        profile = {
            "id": identifier,
            "name": path.name,
            "ready": False,
            "issues": [],
            "uses_model": False,
        }
        try:
            config = config or load_config(path)
            dataset = load_jsonl(path.parent / config.dataset)
            build_evaluators(config)
            profile["process_timeout_seconds"] = min(config.execution.max_seconds or 300, 3600) + 15
            providers = [
                target.provider
                for target in (config.baseline, config.candidate)
                if isinstance(target, (ChatConfig, RAGConfig))
            ]
            if config.judge_provider:
                providers.append(config.judge_provider)
            profile.update(
                name=f"{config.dataset_name or dataset.name} · {path.name}",
                questions=len(dataset.cases),
                uses_model=bool(providers),
                baseline=config.baseline.name,
                candidate=config.candidate.name,
                request_limit=config.execution.max_provider_requests,
            )
            for target in (config.baseline, config.candidate):
                build_target(target, base_dir=path.parent)
            for provider in providers:
                if provider.model.startswith("YOUR_"):
                    profile["issues"].append(
                        "Replace the placeholder model IDs in this configuration."
                    )
                if not os.environ.get(provider.api_key_env):
                    profile["issues"].append(
                        f"Set {provider.api_key_env} locally before starting the dashboard. "
                        "Never paste a key into this page."
                    )
            if providers and (
                config.execution.max_provider_requests is None
                or config.execution.max_seconds is None
                or any(provider.max_completion_tokens is None for provider in providers)
            ):
                profile["issues"].append(
                    "Browser model runs require request, time and output-token limits "
                    "in the configuration."
                )
            profile["issues"] = list(dict.fromkeys(profile["issues"]))
            profile["ready"] = not profile["issues"]
        except (ValueError, OSError, KeyError, TypeError, ImportError, AttributeError):
            profile["issues"].append(
                "Configuration, dataset or documents cannot be validated. Fix the "
                "registered files and refresh setup."
            )
        return profile

    def get(self, identifier):
        try:
            if str(UUID(identifier)) != identifier:
                raise ValueError()
        except ValueError:
            raise FileNotFoundError("job not found") from None
        path = self.directory / identifier / "job.json"
        if path.is_symlink() or path.parent.is_symlink():
            raise FileNotFoundError("job not found")
        return parse_json(path.read_text(encoding="utf-8"))

    def submit(self, profile_id, *, confirm_model_calls=False):
        if profile_id not in self.configs:
            raise KeyError("unknown registered configuration")
        config = load_config(self.configs[profile_id])
        profile = self.profile(profile_id, config=config)
        if not profile["ready"]:
            raise ValueError("; ".join(profile["issues"]))
        if profile["uses_model"] and not confirm_model_calls:
            raise ValueError("confirm model calls before starting this evaluation")
        with self.lock:
            if self.closed or self.active:
                raise JobBusy("An evaluation is already running. Wait for it to finish.")
            identifier = str(uuid4())
            job = {
                "id": identifier,
                "profile": profile["name"],
                "status": "QUEUED",
                "created_at": now(),
                "finished_at": None,
                "exit_code": None,
                "experiment_directory": None,
                "message": "Waiting to start.",
                "config_sha256": hashlib.sha256(self.configs[profile_id].read_bytes()).hexdigest(),
            }
            # Freeze validated configuration and resolve file references against its source.
            base = self.configs[profile_id].parent
            snapshot = config.model_dump(mode="json")
            snapshot["dataset"] = str((base / config.dataset).resolve())
            for name in ("baseline", "candidate"):
                target = snapshot[name]
                for field in ("documents", "system_prompt_file"):
                    if target.get(field):
                        target[field] = str((base / target[field]).resolve())
            frozen = self.directory / identifier / "config.json"
            write_json(frozen, snapshot)
            write_json(self.directory / identifier / "job.json", job)
            self.active = identifier
            self.worker = threading.Thread(
                target=self._run,
                args=(job, frozen, profile["process_timeout_seconds"]),
                daemon=True,
            )
            self.worker.start()
            return dict(job)

    def _run(self, job, config, timeout):
        directory = self.directory / job["id"]
        output = directory / "results"
        try:
            job.update(
                status="RUNNING",
                message="Evaluating baseline and candidate. No quality decision yet.",
            )
            write_json(directory / "job.json", job)
            environment = os.environ.copy()
            # Browser jobs are artifact-only; never silently write an inherited SQL database.
            environment.pop("EVALFORGE_DATABASE_URL", None)
            with (directory / "execution.log").open("wb") as log:
                with self.lock:
                    if self.closed:
                        raise RuntimeError("dashboard stopped")
                    process = subprocess.Popen(
                        [
                            sys.executable,
                            "-m",
                            "evalforge",
                            "run",
                            "--config",
                            str(config),
                            "--output-dir",
                            str(output),
                        ],
                        stdout=log,
                        stderr=log,
                        env=environment,
                    )
                    self.process = process
                try:
                    code = process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                    raise RuntimeError("worker process deadline exceeded") from None
            artifacts = list(output.glob("*/experiment.json")) if output.exists() else []
            if len(artifacts) == 1:
                experiment = Experiment.model_validate(
                    parse_json(artifacts[0].read_text(encoding="utf-8"))
                )
                if code in {0, 1, 3}:
                    destination = self.root / artifacts[0].parent.name
                    artifacts[0].parent.rename(destination)
                    job["experiment_directory"] = destination.name
                    job["comparison_status"] = experiment.comparison.status
            job.update(
                status="COMPLETED" if code in {0, 1} and job["experiment_directory"] else "ERROR",
                exit_code=code,
                message="Evaluation finished. Open the saved comparison."
                if code in {0, 1} and job["experiment_directory"]
                else (
                    "Evaluation could not finish reliably. Check the local "
                    "execution.log and configuration. Saved error evidence may be available."
                ),
            )
        except Exception:
            job.update(
                status="ERROR",
                message="Evaluation worker failed. Check the local job folder before rerunning.",
            )
        finally:
            with self.lock:
                if self.closed:
                    job.update(
                        status="INTERRUPTED",
                        message="Dashboard stopped. This job was not automatically retried.",
                    )
                job["finished_at"] = now()
                write_json(directory / "job.json", job)
                self.active = None
                self.process = None

    def close(self):
        with self.lock:
            self.closed = True
            process = self.process
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        if self.worker:
            self.worker.join(timeout=10)
        if self.ownership:
            self.ownership.close()
            self.ownership = None
