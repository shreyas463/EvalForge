"""Read-only loopback dashboard for validated saved experiment artifacts."""

import json
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from urllib.parse import urlsplit
from uuid import UUID

from evalforge.datasets import parse_json
from evalforge.models import Experiment

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


def make_server(root: str | Path, *, port: int = 8765):
    reader = ArtifactReader(root)
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
                if path == "/api/experiments":
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
            except (OSError, ValueError):
                self.send(422, b'{"error":"artifact cannot be read or validated"}')

    # No remote binding option: saved inputs/answers are local application data.
    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def serve(root: str | Path, *, port: int = 8765, open_browser: bool = False):
    server = make_server(root, port=port)
    url = f"http://127.0.0.1:{server.server_port}"
    print(f"EvalForge dashboard: {url}", flush=True)
    print(f"Reading saved results from: {Path(root).resolve()}", flush=True)
    print("Read-only local dashboard. Press Ctrl+C to stop.", flush=True)
    try:
        if open_browser:
            webbrowser.open(url)
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0
