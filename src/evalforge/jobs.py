"""Bounded local evaluation jobs for operator-registered configuration files.

Each job runs the installed CLI in its own process. State survives dashboard restarts;
unfinished jobs are reported as interrupted, never silently restarted or rerun.
"""

import hashlib
import json
import os
import re
import sqlite3
import subprocess
import sys
import threading
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

from evalforge.config import (
    ChatConfig,
    ExperimentConfig,
    RAGConfig,
    RetrievalConfig,
    build_evaluators,
    build_target,
    load_config,
)
from evalforge.datasets import dataset_hash, load_jsonl, parse_json
from evalforge.models import BaselineApproval, Experiment
from evalforge.rag import BM25Retriever
from evalforge.storage import SQLStore, StorageError, write_json


class JobBusy(RuntimeError):
    pass


class WorkerTimeout(RuntimeError):
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
        self.cancelled = set()
        self.current_job = None
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
                    job = self.get(path.parent.name)
                except (OSError, ValueError):
                    continue
                if not isinstance(job, dict):
                    continue
                if job.get("status") in {"QUEUED", "RUNNING", "CANCELLING"}:
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
            "can_preview_retrieval": False,
        }
        try:
            config = config or load_config(path)
            dataset = load_jsonl(path.parent / config.dataset)
            evaluators = build_evaluators(config)
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
                dataset=config.dataset_name or dataset.name,
                dataset_version=config.dataset_version or dataset.version,
                dataset_hash=dataset_hash(dataset.cases),
                evaluators=[e.spec.model_dump(mode="json") for e in evaluators],
                model_settings=config.baseline.provider.model_dump(mode="json")
                if isinstance(config.baseline, RAGConfig)
                else None,
                judge_model=config.judge_provider.model if config.judge_provider else None,
                uses_model=bool(providers),
                baseline=config.baseline.name,
                candidate=config.candidate.name,
                request_limit=config.execution.max_provider_requests,
            )
            for target in (config.baseline, config.candidate):
                build_target(target, base_dir=path.parent)
            profile["can_preview_retrieval"] = all(
                isinstance(target, (RAGConfig, RetrievalConfig))
                for target in (config.baseline, config.candidate)
            )
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

    def preview_retrieval(self, profile_id, question):
        """Search trusted registered corpora; never construct/call a model or read labels."""
        if profile_id not in self.configs:
            raise KeyError("unknown registered configuration")
        if not isinstance(question, str) or not question.strip() or len(question) > 2000:
            raise ValueError("enter a question of 1 to 2000 characters")
        path = self.configs[profile_id]
        config = load_config(path)
        result = {"question": question.strip(), "mode": "retrieval_only"}
        for name in ("baseline", "candidate"):
            target = getattr(config, name)
            if not isinstance(target, (RAGConfig, RetrievalConfig)):
                raise ValueError("source previews require two RAG targets")
            retriever = BM25Retriever(
                path.parent / target.documents, chunk_words=target.chunk_words
            )
            trace = retriever.retrieve(question.strip(), top_k=target.top_k)
            result[name] = {
                "name": target.name,
                "documents": len({chunk["document_id"] for chunk in retriever.chunks}),
                "trace": trace.model_dump(mode="json"),
            }
        return result

    def get(self, identifier):
        try:
            if str(UUID(identifier)) != identifier:
                raise ValueError()
        except ValueError:
            raise FileNotFoundError("job not found") from None
        path = self.directory / identifier / "job.json"
        if path.is_symlink() or path.parent.is_symlink():
            raise FileNotFoundError("job not found")
        with path.open("rb") as source:
            data = source.read(65537)
        if len(data) > 65536:
            raise ValueError("job state exceeds size limit")
        return parse_json(data.decode("utf-8"))

    def history(self):
        jobs, invalid = [], 0
        if self.directory.exists():
            for directory in self.directory.iterdir():
                if not directory.is_dir():
                    continue
                try:
                    job = self.get(directory.name)
                    if (
                        not isinstance(job.get("created_at", ""), str)
                        or job.get("id") != directory.name
                        or job.get("status")
                        not in {
                            "QUEUED",
                            "RUNNING",
                            "CANCELLING",
                            "COMPLETED",
                            "ERROR",
                            "INTERRUPTED",
                            "CANCELLED",
                        }
                    ):
                        raise ValueError("invalid job state")
                    jobs.append(job)
                except (OSError, ValueError, AttributeError):
                    invalid += 1
        jobs.sort(key=lambda job: (job.get("created_at", ""), job["id"]), reverse=True)
        return {"jobs": jobs, "invalid_jobs": invalid}

    def cancel(self, identifier):
        self.get(identifier)  # Canonical UUID and file/path validation.
        with self.lock:
            if self.active != identifier:
                raise JobBusy("This job is no longer running. Refresh job history.")
            if self.process is not None and self.process.poll() is not None:
                raise JobBusy("The process has finished; wait for its saved result.")
            self.cancelled.add(identifier)
            self.current_job.update(status="CANCELLING", message="Stopping the evaluation process.")
            write_json(self.directory / identifier / "job.json", self.current_job)
            process = self.process
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        return self.get(identifier)

    def _baseline_store(self):
        folder = self.root / "_baselines"
        path = folder / "baselines.db"
        if folder.is_symlink() or path.is_symlink():
            raise StorageError("baseline store must not be a symlink")
        folder.mkdir(parents=True, exist_ok=True)
        return SQLStore(f"sqlite:///{path}")

    def baselines(self):
        if not (self.root / "_baselines" / "baselines.db").exists():
            return []
        store = self._baseline_store()
        try:
            items = []
            for approval in store.list_baselines():
                run = store.load_run(approval.run_id)
                items.append(
                    {
                        **approval.model_dump(mode="json"),
                        "dataset": run.dataset_name,
                        "target": run.target_name,
                        "dataset_hash": run.dataset_hash,
                        "dataset_version": run.dataset_version,
                        "evaluators": [e.model_dump(mode="json") for e in run.evaluators],
                        "history": [
                            a.model_dump(mode="json") for a in store.baseline_history(approval.name)
                        ],
                    }
                )
            return items
        finally:
            store.close()

    def approve(self, experiment, *, side, name, approved_by, note=""):
        if not self.configs:
            raise ValueError("enable configurations before approving local baselines")
        if side not in {"baseline", "candidate"}:
            raise ValueError("choose original or changed version")
        if len(name) > 100 or len(approved_by) > 100 or len(note) > 2000:
            raise ValueError("approval fields are too long")
        # Validate names before opening a database.
        BaselineApproval(
            name=name, run_id=getattr(experiment, side).id, approved_by=approved_by, note=note
        )
        store = self._baseline_store()
        try:
            return store.approve_baseline(
                getattr(experiment, side), name=name, approved_by=approved_by, note=note
            )
        finally:
            store.close()

    def configure(self, profile_id, settings):
        if profile_id not in self.configs:
            raise KeyError("unknown registered configuration")
        if set(settings) != {"model", "judge_model", "base_url", "api_key_env"} or any(
            not isinstance(value, str) or len(value) > 500 for value in settings.values()
        ):
            raise ValueError("provide model names, endpoint and credential-variable name only")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,99}", settings["api_key_env"]):
            raise ValueError("use an environment variable name, never a credential value")
        path = self.configs[profile_id]
        config = load_config(path)
        if not all(isinstance(target, RAGConfig) for target in (config.baseline, config.candidate)):
            raise ValueError("model settings require an existing RAG configuration")
        snapshot = config.model_dump(mode="json")
        for provider in [snapshot["baseline"]["provider"], snapshot["candidate"]["provider"]] + (
            [snapshot["judge_provider"]] if snapshot["judge_provider"] else []
        ):
            provider.update(
                model=settings["model"],
                base_url=settings["base_url"],
                api_key_env=settings["api_key_env"],
            )
        if snapshot["judge_provider"]:
            snapshot["judge_provider"]["model"] = settings["judge_model"] or settings["model"]
        updated = ExperimentConfig.model_validate(snapshot, strict=True)
        build_evaluators(updated)
        with self.lock:
            if self.active:
                raise JobBusy("Wait for the current job before changing model settings.")
            write_json(path, updated.model_dump(mode="json"))
        return self.profile(profile_id)

    def submit(self, profile_id, *, confirm_model_calls=False, baseline_name=None):
        if profile_id not in self.configs:
            raise KeyError("unknown registered configuration")
        config = load_config(self.configs[profile_id])
        profile = self.profile(profile_id, config=config)
        if not profile["ready"]:
            raise ValueError("; ".join(profile["issues"]))
        if profile["uses_model"] and not confirm_model_calls:
            raise ValueError("confirm model calls before starting this evaluation")
        baseline = None
        if baseline_name is not None:
            store = self._baseline_store()
            try:
                baseline = store.resolve_baseline(name=baseline_name)
                dataset = load_jsonl(
                    self.configs[profile_id].parent / config.dataset,
                    name=config.dataset_name,
                    version=config.dataset_version,
                )

                if (baseline.dataset_name, baseline.dataset_version, baseline.dataset_hash) != (
                    dataset.name,
                    dataset.version,
                    dataset_hash(dataset.cases),
                ):
                    raise ValueError("approved baseline uses a different dataset version")
                expected = sorted(
                    json.dumps(e.spec.model_dump(mode="json"), sort_keys=True)
                    for e in build_evaluators(config)
                )
                if (
                    sorted(
                        json.dumps(e.model_dump(mode="json"), sort_keys=True)
                        for e in baseline.evaluators
                    )
                    != expected
                ):
                    raise ValueError("approved baseline uses different evaluator settings")
            finally:
                store.close()
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
                "baseline_name": baseline_name,
                "baseline_run_id": baseline.id if baseline else None,
                "config_sha256": hashlib.sha256(self.configs[profile_id].read_bytes()).hexdigest(),
            }
            # Freeze validated configuration and resolve file references against its source.
            base = self.configs[profile_id].parent
            snapshot = config.model_dump(mode="json")
            snapshot["dataset"] = str((base / config.dataset).resolve())
            for name in ("baseline", "candidate"):
                target = snapshot[name]
                for field in ("documents", "system_prompt_file", "records"):
                    if target.get(field):
                        target[field] = str((base / target[field]).resolve())
            frozen = self.directory / identifier / "config.json"
            write_json(frozen, snapshot)
            write_json(self.directory / identifier / "job.json", job)
            self.active = identifier
            self.current_job = job
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
                    if self.closed or job["id"] in self.cancelled:
                        raise RuntimeError("dashboard stopped or job cancelled")
                    command = [
                        sys.executable,
                        "-m",
                        "evalforge",
                        "run",
                        "--config",
                        str(config),
                        "--output-dir",
                        str(output),
                    ]
                    if job["baseline_name"]:
                        command.extend(
                            [
                                "--baseline-id",
                                job["baseline_run_id"],
                                "--database-url",
                                f"sqlite:///{self.root / '_baselines' / 'baselines.db'}",
                            ]
                        )
                    process = subprocess.Popen(
                        command,
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
                    raise WorkerTimeout("worker process deadline exceeded") from None
            artifacts = list(output.glob("*/experiment.json")) if output.exists() else []
            if len(artifacts) == 1:
                experiment = Experiment.model_validate(
                    parse_json(artifacts[0].read_text(encoding="utf-8"))
                )
                if code in {0, 1, 3} and job["id"] not in self.cancelled:
                    destination = self.root / artifacts[0].parent.name
                    artifacts[0].parent.rename(destination)
                    job["experiment_directory"] = destination.name
                    job["comparison_status"] = experiment.comparison.status
            job.update(
                status="COMPLETED" if code in {0, 1} and job["experiment_directory"] else "ERROR",
                exit_code=code,
                failure_kind="configuration"
                if code == 2
                else "execution"
                if code not in {0, 1}
                else None,
                message="Evaluation finished. Open the saved comparison."
                if code in {0, 1} and job["experiment_directory"]
                else "Configuration or dataset is invalid. Refresh setup and check files."
                if code == 2
                else (
                    "Evaluation could not finish reliably. Check the local "
                    "execution.log and configuration. Saved error evidence may be available."
                ),
            )
        except WorkerTimeout:
            job.update(
                status="ERROR",
                failure_kind="timeout",
                message=(
                    "Time limit reached. Reduce the test size or increase the "
                    "configured budget before rerunning."
                ),
            )
        except Exception:
            job.update(
                status="ERROR",
                message="Evaluation worker failed. Check the local job folder before rerunning.",
            )
        finally:
            with self.lock:
                if job["id"] in self.cancelled:
                    job.update(
                        status="CANCELLED",
                        message="Evaluation cancelled. No quality decision was made.",
                    )
                    self.cancelled.discard(job["id"])
                elif self.closed:
                    job.update(
                        status="INTERRUPTED",
                        message="Dashboard stopped. This job was not automatically retried.",
                    )
                job["finished_at"] = now()
                try:
                    write_json(directory / "job.json", job)
                finally:
                    self.active = None
                    self.process = None
                    self.current_job = None

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
