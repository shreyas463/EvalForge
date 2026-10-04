"""Loopback dashboard for validated artifacts and registered local evaluation jobs."""

import json
import secrets
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from urllib.parse import urlsplit
from uuid import UUID

from evalforge.datasets import parse_json
from evalforge.jobs import JobBusy, JobManager
from evalforge.models import Experiment
from evalforge.storage import StorageError

MAX_ARTIFACT_BYTES = 16 * 1024 * 1024


class ArtifactReader:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()

    def load(self, identifier: str):
        try:
            if str(UUID(identifier)) != identifier:
                raise ValueError("noncanonical experiment directory")
        except ValueError:
            raise FileNotFoundError("experiment not found") from None
        path = self.root / identifier / "experiment.json"
        if (
            path.parent.is_symlink()
            or path.is_symlink()
            or not path.resolve().is_relative_to(self.root)
        ):
            raise FileNotFoundError("experiment not found")
        # Bound reads even if an artifact changes concurrently.
        with path.open("rb") as file:
            data = file.read(MAX_ARTIFACT_BYTES + 1)
        if len(data) > MAX_ARTIFACT_BYTES:
            raise ValueError("experiment exceeds dashboard size limit")
        return Experiment.model_validate(parse_json(data.decode("utf-8")))

    def list(self):
        summaries = []
        invalid = 0
        if not self.root.exists():
            return {"experiments": [], "invalid_artifacts": 0}
        for directory in self.root.iterdir():
            try:
                UUID(directory.name)
            except ValueError:
                continue
            if not directory.is_dir():
                continue
            try:
                experiment = self.load(directory.name)
            except (OSError, ValueError):
                invalid += 1
                continue
            summaries.append(
                {
                    "directory": directory.name,
                    "id": experiment.id,
                    "created_at": experiment.candidate.created_at.isoformat(),
                    "dataset": experiment.candidate.dataset_name,
                    "baseline": experiment.baseline.target_name,
                    "candidate": experiment.candidate.target_name,
                    "status": experiment.comparison.status,
                    "cases": len(experiment.candidate.cases),
                    "target_kind": experiment.candidate.target_config.get("kind", "unknown"),
                    "has_retrieval": any(
                        r.target.retrieval is not None for r in experiment.candidate.cases
                    ),
                }
            )
        summaries.sort(key=lambda item: (item["created_at"], item["directory"]), reverse=True)
        return {"experiments": summaries, "invalid_artifacts": invalid}


def make_server(root: str | Path, *, port: int = 8765, configs=()):
    reader = ArtifactReader(root)
    jobs = JobManager(root, configs)
    token = secrets.token_urlsafe(32)
    html = files("evalforge").joinpath("web/dashboard.html").read_bytes()
    script = files("evalforge").joinpath("web/dashboard.js").read_bytes()
    stylesheet = files("evalforge").joinpath("web/dashboard.css").read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def send(self, status, body, content_type="application/json"):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; frame-ancestors 'none'; object-src 'none'",
            )
            self.end_headers()
            self.wfile.write(body)

        def local_request(self):
            hosts = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
            host, origin = self.headers.get("Host"), self.headers.get("Origin")
            return host in hosts and (not origin or origin == f"http://{host}")

        def do_POST(self):
            if not self.local_request() or not secrets.compare_digest(
                self.headers.get("X-EvalForge-Token", ""), token
            ):
                self.send(403, b'{"error":"local session required"}')
                return
            if self.path not in {
                "/api/jobs",
                "/api/retrieval",
                "/api/cancel",
                "/api/approve",
                "/api/configure",
            }:
                self.send(404, b'{"error":"route not found"}')
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 4096 or self.headers.get("Content-Type") != "application/json":
                    raise ValueError("expected a small JSON request")
                payload = parse_json(self.rfile.read(length).decode("utf-8"))
                if self.path == "/api/approve":
                    if (
                        not isinstance(payload, dict)
                        or set(payload) != {"experiment_id", "side", "name", "approved_by", "note"}
                        or any(not isinstance(v, str) for v in payload.values())
                    ):
                        raise ValueError("provide an experiment and approval details")
                    experiment = reader.load(payload.pop("experiment_id"))
                    self.send(201, jobs.approve(experiment, **payload).model_dump_json().encode())
                    return
                if self.path == "/api/configure":
                    if (
                        not isinstance(payload, dict)
                        or set(payload) != {"profile_id", "settings"}
                        or not isinstance(payload["profile_id"], str)
                        or not isinstance(payload["settings"], dict)
                    ):
                        raise ValueError("select a registered model configuration")
                    self.send(
                        200,
                        json.dumps(
                            jobs.configure(payload["profile_id"], payload["settings"])
                        ).encode(),
                    )
                    return
                if self.path == "/api/cancel":
                    if (
                        not isinstance(payload, dict)
                        or set(payload) != {"job_id"}
                        or not isinstance(payload["job_id"], str)
                    ):
                        raise ValueError("select a running job")
                    self.send(200, json.dumps(jobs.cancel(payload["job_id"])).encode())
                    return
                if self.path == "/api/retrieval":
                    if (
                        not isinstance(payload, dict)
                        or set(payload) != {"profile_id", "question"}
                        or not isinstance(payload["profile_id"], str)
                    ):
                        raise ValueError("select a registered RAG configuration and question")
                    result = jobs.preview_retrieval(payload["profile_id"], payload["question"])
                    self.send(200, json.dumps(result, ensure_ascii=False, allow_nan=False).encode())
                    return
                if (
                    not isinstance(payload, dict)
                    or set(payload) - {"profile_id", "confirm_model_calls", "baseline_name"}
                    or not isinstance(payload.get("profile_id"), str)
                    or not isinstance(payload.get("confirm_model_calls", False), bool)
                    or (
                        payload.get("baseline_name") is not None
                        and not isinstance(payload["baseline_name"], str)
                    )
                ):
                    raise ValueError("select a registered configuration")
                job = jobs.submit(
                    payload["profile_id"],
                    confirm_model_calls=payload.get("confirm_model_calls", False),
                    baseline_name=payload.get("baseline_name"),
                )
                self.send(202, json.dumps(job).encode())
            except JobBusy as exc:
                self.send(409, json.dumps({"error": str(exc)}).encode())
            except FileNotFoundError:
                self.send(404, b'{"error":"job not found"}')
            except KeyError:
                self.send(404, b'{"error":"registered configuration not found"}')
            except StorageError:
                self.send(
                    422,
                    b'{"error":"local baseline unavailable; check approval and saved evidence"}',
                )
            except (ValueError, OSError):
                self.send(
                    422,
                    b'{"error":"configuration is not ready; refresh setup"}',
                )

        def do_GET(self):
            allowed_hosts = {
                f"127.0.0.1:{self.server.server_port}",
                f"localhost:{self.server.server_port}",
            }
            host = self.headers.get("Host")
            origin = self.headers.get("Origin")
            if host not in allowed_hosts or (origin and origin != f"http://{host}"):
                self.send(403, b'{"error":"local origin required"}')
                return
            path = urlsplit(self.path).path
            static = {
                "/": (html, "text/html; charset=utf-8"),
                "/dashboard.js": (script, "text/javascript; charset=utf-8"),
                "/dashboard.css": (stylesheet, "text/css; charset=utf-8"),
            }
            if path in static:
                body, content_type = static[path]
                self.send(200, body, content_type)
                return
            try:
                if path == "/api/setup":
                    payload = {
                        "profiles": jobs.profiles(),
                        "session_token": token,
                        "active_job": jobs.active,
                        "baselines_enabled": bool(jobs.configs),
                    }
                elif path == "/api/baselines":
                    payload = {"baselines": jobs.baselines()}
                elif path == "/api/jobs":
                    payload = jobs.history()
                elif path.startswith("/api/jobs/"):
                    payload = jobs.get(path.removeprefix("/api/jobs/"))
                elif path == "/api/experiments":
                    payload = reader.list()
                elif path.startswith("/api/experiments/"):
                    payload = reader.load(path.removeprefix("/api/experiments/")).model_dump(
                        mode="json"
                    )
                else:
                    raise FileNotFoundError()
                self.send(200, json.dumps(payload, ensure_ascii=False, allow_nan=False).encode())
            except FileNotFoundError:
                self.send(404, b'{"error":"experiment not found"}')
            except (OSError, ValueError, StorageError):
                self.send(422, b'{"error":"artifact cannot be read or validated"}')

    # No remote binding option: saved inputs/answers are local application data.
    class Server(ThreadingHTTPServer):
        def server_close(self):
            jobs.close()
            super().server_close()

    try:
        return Server(("127.0.0.1", port), Handler)
    except OSError:
        jobs.close()
        raise


def serve(root: str | Path, *, port: int = 8765, open_browser: bool = False, configs=()):
    server = make_server(root, port=port, configs=configs)
    url = f"http://127.0.0.1:{server.server_port}"
    print(f"EvalForge dashboard: {url}", flush=True)
    print(f"Reading saved results from: {Path(root).resolve()}", flush=True)
    print(
        "Local dashboard. Only explicitly registered configurations can run. Press Ctrl+C to stop.",
        flush=True,
    )
    try:
        if open_browser:
            webbrowser.open(url)
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0
