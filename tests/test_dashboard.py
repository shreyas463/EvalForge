import json
import threading
from uuid import uuid4

import httpx
import pytest
from test_storage_cli import sample_experiment

from evalforge.cli import main
from evalforge.dashboard import ArtifactReader, make_server, serve
from evalforge.storage import write_json


@pytest.fixture
def artifacts(tmp_path):
    directory = tmp_path / str(uuid4())
    experiment = sample_experiment()
    write_json(directory / "experiment.json", experiment.model_dump(mode="json"))
    return tmp_path, directory.name, experiment


def test_reader_lists_saved_experiments_and_missing_root(artifacts):
    root, identifier, experiment = artifacts
    reader = ArtifactReader(root)
    assert reader.load(identifier) == experiment
    summary = reader.list()["experiments"][0]
    assert summary["directory"] == identifier
    assert summary["status"] == "FAIL"
    assert summary["cases"] == 1
    assert not summary["has_retrieval"]
    assert ArtifactReader(root / "missing").list() == {"experiments": [], "invalid_artifacts": 0}


def test_reader_skips_invalid_files_and_rejects_paths(artifacts):
    root, identifier, _ = artifacts
    invalid = root / str(uuid4())
    invalid.mkdir()
    (invalid / "experiment.json").write_text('{"not":"an experiment"}')
    (root / "support-demo-session").mkdir()
    (root / str(uuid4())).write_text("not a directory")
    reader = ArtifactReader(root)
    assert reader.list()["invalid_artifacts"] == 1
    assert len(reader.list()["experiments"]) == 1
    for value in ("../outside", "../../etc/passwd", identifier + "/experiment.json", "nope"):
        with pytest.raises(FileNotFoundError):
            reader.load(value)


def test_reader_rejects_symlinks_and_large_artifacts(artifacts, monkeypatch):
    root, identifier, _ = artifacts
    linked = root / str(uuid4())
    linked.symlink_to(root / identifier, target_is_directory=True)
    with pytest.raises(FileNotFoundError):
        ArtifactReader(root).load(linked.name)
    monkeypatch.setattr("evalforge.dashboard.MAX_ARTIFACT_BYTES", 10)
    with pytest.raises(ValueError, match="size"):
        ArtifactReader(root).load(identifier)


def test_http_routes_and_local_origin_guards(artifacts):
    root, identifier, experiment = artifacts
    server = make_server(root, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with httpx.Client(base_url=base, trust_env=False) as client:
            response = client.get("/")
            assert response.status_code == 200
            assert "Experiment results" in response.text
            assert "default-src 'self'" in response.headers["content-security-policy"]
            for path in ("/dashboard.js", "/dashboard.css"):
                assert client.get(path).status_code == 200
            assert (
                client.get("/api/experiments").json()["experiments"][0]["directory"] == identifier
            )
            assert client.get(f"/api/experiments/{identifier}").json()["id"] == experiment.id
            assert client.get("/api/experiments/nope").status_code == 404
            assert client.get("/unknown").status_code == 404
            assert (
                client.get("/api/experiments", headers={"Host": "attacker.example"}).status_code
                == 403
            )
            assert (
                client.get(
                    "/api/experiments", headers={"Origin": "https://attacker.example"}
                ).status_code
                == 403
            )
            assert client.get("/api/experiments", headers={"Origin": base}).status_code == 200
            assert client.post("/api/experiments").status_code == 501
            (root / identifier / "experiment.json").write_text("broken")
            assert client.get(f"/api/experiments/{identifier}").status_code == 422
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_dashboard_cli_dispatch_and_invalid_port(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "evalforge.dashboard.serve", lambda root, **kwargs: calls.append((root, kwargs)) or 0
    )
    assert main(["dashboard", "--artifact-root", "runs", "--port", "9999", "--open"]) == 0
    assert calls == [("runs", {"port": 9999, "open_browser": True})]
    assert main(["dashboard", "--port", "-1"]) == 2


def test_dashboard_server_cleanup_and_optional_browser(monkeypatch, tmp_path):
    events = []

    class Server:
        server_port = 8765

        def serve_forever(self):
            raise KeyboardInterrupt()

        def server_close(self):
            events.append("closed")

    monkeypatch.setattr("evalforge.dashboard.make_server", lambda *args, **kwargs: Server())
    monkeypatch.setattr("evalforge.dashboard.webbrowser.open", lambda url: events.append(url))
    assert serve(tmp_path, open_browser=True) == 0
    assert events == ["http://127.0.0.1:8765", "closed"]


def test_untrusted_answers_remain_json_not_html(artifacts):
    root, identifier, experiment = artifacts
    payload = experiment.model_dump(mode="json")
    payload["candidate"]["cases"][0]["target"]["output"] = '<script>alert("x")</script>'
    write_json(root / identifier / "experiment.json", payload)
    loaded = ArtifactReader(root).load(identifier).model_dump(mode="json")
    assert json.loads(json.dumps(loaded))["candidate"]["cases"][0]["target"]["output"].startswith(
        "<script>"
    )
