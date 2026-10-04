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
            assert client.post("/api/experiments").status_code == 403
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
    assert calls == [("runs", {"port": 9999, "open_browser": True, "configs": []})]
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


def test_registered_browser_job_runs_real_cli(tmp_path):
    from pathlib import Path

    from evalforge.jobs import JobManager

    examples = Path(__file__).resolve().parents[1] / "examples/support/offline.json"
    manager = JobManager(tmp_path, [examples])
    try:
        assert manager.profiles()[0]["ready"]
        job = manager.submit("0")
        manager.worker.join(timeout=20)
        saved = manager.get(job["id"])
        assert saved["status"] == "COMPLETED"
        assert saved["exit_code"] == 1  # A completed quality failure is not a worker error.
        frozen = json.loads((tmp_path / "_jobs" / job["id"] / "config.json").read_text())
        assert Path(frozen["dataset"]).is_absolute()
        for key, value in json.loads(examples.read_text())["execution"].items():
            assert frozen["execution"][key] == value
        experiment = ArtifactReader(tmp_path).load(saved["experiment_directory"])
        assert experiment.comparison.status == "FAIL"
        assert len(experiment.candidate.cases) == 8
    finally:
        manager.close()


def test_setup_model_requirements_no_secret_exposure(tmp_path, monkeypatch):
    from pathlib import Path

    from evalforge.jobs import JobManager

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    manager = JobManager(tmp_path, [Path(__file__).resolve().parents[1] / "examples/rag/live.json"])
    try:
        profile = manager.profiles()[0]
        assert not profile["ready"]
        assert profile["uses_model"]
        assert any("placeholder" in issue for issue in profile["issues"])
        assert any("OPENAI_API_KEY" in issue for issue in profile["issues"])
        with pytest.raises(ValueError):
            manager.submit("0")
        assert not list((tmp_path / "_jobs").glob("*/job.json"))
    finally:
        manager.close()


def test_worker_ownership_and_restart_recovery(tmp_path):
    from pathlib import Path

    from evalforge.jobs import JobManager

    config = Path(__file__).resolve().parents[1] / "examples/support/offline.json"
    manager = JobManager(tmp_path, [config])
    try:
        with pytest.raises(ValueError, match="owns"):
            JobManager(tmp_path, [config])
        job_id = str(uuid4())
        write_json(tmp_path / "_jobs" / job_id / "job.json", {"id": job_id, "status": "RUNNING"})
        viewer = JobManager(tmp_path)
        assert viewer.get(job_id)["status"] == "RUNNING"  # Reading never interrupts another worker.
        viewer.close()
    finally:
        manager.close()
    restarted = JobManager(tmp_path, [config])
    try:
        assert restarted.get(job_id)["status"] == "INTERRUPTED"
        with pytest.raises(FileNotFoundError):
            restarted.get("../outside")
        with pytest.raises(KeyError):
            restarted.submit("unregistered/path.json")
    finally:
        restarted.close()


def test_http_job_submission_requires_session_token(tmp_path):
    from pathlib import Path

    config = Path(__file__).resolve().parents[1] / "examples/support/offline.json"
    server = make_server(tmp_path, port=0, configs=[config])
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with httpx.Client(base_url=base, trust_env=False) as client:
            setup = client.get("/api/setup").json()
            headers = {"X-EvalForge-Token": setup["session_token"]}
            assert client.post("/api/jobs", json={"profile_id": "0"}).status_code == 403
            assert (
                client.post(
                    "/api/jobs",
                    json={"profile_id": "0"},
                    headers={**headers, "Origin": "https://attacker.example"},
                ).status_code
                == 403
            )
            assert (
                client.post(
                    "/api/jobs", json={"profile_id": "unknown"}, headers=headers
                ).status_code
                == 404
            )
            assert (
                client.post("/api/jobs", json={"config": "/tmp/file"}, headers=headers).status_code
                == 422
            )
            response = client.post("/api/jobs", json={"profile_id": "0"}, headers=headers)
            assert response.status_code == 202
            identifier = response.json()["id"]
            assert client.get(f"/api/jobs/{identifier}").json()["status"] in {
                "QUEUED",
                "RUNNING",
                "COMPLETED",
            }
            assert client.get("/api/jobs/not-a-uuid").status_code == 404
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_busy_job_timeout_and_failed_worker_release_slot(tmp_path, monkeypatch):
    import subprocess
    from pathlib import Path

    from evalforge.jobs import JobBusy, JobManager

    config = Path(__file__).resolve().parents[1] / "examples/support/offline.json"
    entered, release = threading.Event(), threading.Event()
    killed = []

    class Process:
        def __init__(self, *args, **kwargs):
            self.timed_out = False

        def wait(self, timeout=None):
            if not self.timed_out:
                entered.set()
                release.wait(5)
                self.timed_out = True
                raise subprocess.TimeoutExpired("controlled test", timeout)
            return -9

        def kill(self):
            killed.append(True)

        def poll(self):
            return -9 if self.timed_out else None

    monkeypatch.setattr("evalforge.jobs.subprocess.Popen", Process)
    manager = JobManager(tmp_path, [config])
    try:
        job = manager.submit("0")
        assert entered.wait(5)
        with pytest.raises(JobBusy):
            manager.submit("0")
        release.set()
        manager.worker.join(5)
        assert killed == [True]
        assert manager.get(job["id"])["status"] == "ERROR"
        assert manager.active is None
    finally:
        manager.close()


def test_worker_keeps_execution_error_evidence(tmp_path):
    from pathlib import Path

    from evalforge.jobs import JobManager

    original = Path(__file__).resolve().parents[1] / "examples/support/offline.json"
    config = json.loads(original.read_text())
    config["dataset"] = str(original.parent / "cases.jsonl")
    for name in ("baseline", "candidate"):
        config[name] = {"kind": "mock", "name": name, "responses": {}}
    path = tmp_path / "errors.json"
    write_json(path, config)
    manager = JobManager(tmp_path, [path])
    try:
        job = manager.submit("0")
        manager.worker.join(20)
        saved = manager.get(job["id"])
        assert saved["status"] == "ERROR"
        assert saved["exit_code"] == 3
        assert (
            ArtifactReader(tmp_path).load(saved["experiment_directory"]).comparison.status
            == "ERROR"
        )
    finally:
        manager.close()


def test_model_job_requires_confirmation_and_budgets(tmp_path, monkeypatch):
    from pathlib import Path

    from evalforge.jobs import JobManager

    source = Path(__file__).resolve().parents[1] / "examples/rag/live.json"
    config = json.loads(source.read_text())
    config["dataset"] = str(source.parent / config["dataset"])
    for name in ("baseline", "candidate"):
        config[name]["documents"] = str(source.parent / config[name]["documents"])
        config[name]["provider"]["model"] = "controlled-test-model"
    config["judge_provider"]["model"] = "controlled-test-model"
    monkeypatch.setenv("OPENAI_API_KEY", "controlled-secret-never-returned")
    path = tmp_path / "model.json"
    write_json(path, config)
    manager = JobManager(tmp_path / "artifacts", [path])
    try:
        assert manager.profile("0")["ready"]
        assert "controlled-secret" not in json.dumps(manager.profiles())
        with pytest.raises(ValueError, match="confirm model calls"):
            manager.submit("0")
        config["execution"]["max_provider_requests"] = None
        write_json(path, config)
        assert not manager.profile("0")["ready"]
        with pytest.raises(ValueError, match="limits"):
            manager.submit("0", confirm_model_calls=True)
        assert manager.active is None
    finally:
        manager.close()


def test_rag_preview_searches_real_corpora_without_models_or_labels(tmp_path, monkeypatch):
    from pathlib import Path

    from evalforge.jobs import JobManager

    source = Path(__file__).resolve().parents[1] / "examples/rag/live.json"
    manager = JobManager(tmp_path, [source])
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    try:
        profile = manager.profile("0")
        assert profile["can_preview_retrieval"] and not profile["ready"]

        def forbidden(*args, **kwargs):
            pytest.fail(
                "a retrieval preview must not build model targets or read evaluation labels"
            )

        monkeypatch.setattr("evalforge.jobs.build_target", forbidden)
        monkeypatch.setattr("evalforge.jobs.load_jsonl", forbidden)
        result = manager.preview_retrieval("0", "  refund  ")
        assert result["mode"] == "retrieval_only"
        assert result["question"] == "refund"
        assert result["baseline"]["documents"] == 3
        assert result["candidate"]["documents"] == 2
        passages = result["baseline"]["trace"]["passages"]
        assert passages[0]["document_id"] == "refunds.md"
        assert "14 days" in passages[0]["text"]
        assert result["candidate"]["trace"]["passages"] == []
        assert result["baseline"]["trace"]["citations"] == []
        assert manager.active is None
        assert not list((tmp_path / "_jobs").glob("*/job.json"))
        assert not ArtifactReader(tmp_path).list()["experiments"]
        for invalid in ("", "   ", None, {}, "x" * 2001):
            with pytest.raises(ValueError, match="question"):
                manager.preview_retrieval("0", invalid)
        with pytest.raises(KeyError):
            manager.preview_retrieval("../outside.json", "refund")
    finally:
        manager.close()


def test_rag_preview_rebuilds_documents_and_keeps_empty_matches(tmp_path):
    from pathlib import Path

    from evalforge.jobs import JobManager

    source = Path(__file__).resolve().parents[1] / "examples/rag/live.json"
    config = json.loads(source.read_text())
    documents = tmp_path / "docs"
    documents.mkdir()
    document = documents / "policy.md"
    document.write_text("# Refund policy\nRefunds within 14 days.")
    for name in ("baseline", "candidate"):
        config[name]["documents"] = str(documents)
    path = tmp_path / "rag.json"
    write_json(path, config)
    manager = JobManager(tmp_path / "results", [path])
    try:
        first = manager.preview_retrieval("0", "refunds")["baseline"]["trace"]
        document.write_text("# Refund policy\nRefunds within 7 days.")
        updated = manager.preview_retrieval("0", "refunds")["baseline"]["trace"]
        assert first["corpus_hash"] != updated["corpus_hash"]
        assert "7 days" in updated["passages"][0]["text"]
        assert manager.preview_retrieval("0", "zzzznomatch")["baseline"]["trace"]["passages"] == []
        document.unlink()
        with pytest.raises(ValueError, match="nonempty"):
            manager.preview_retrieval("0", "refunds")
    finally:
        manager.close()


def test_http_retrieval_preview_checks_session_profile_and_question(tmp_path):
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    server = make_server(
        tmp_path,
        port=0,
        configs=[root / "examples/rag/live.json", root / "examples/support/offline.json"],
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with httpx.Client(base_url=base, trust_env=False) as client:
            setup = client.get("/api/setup").json()
            headers = {"X-EvalForge-Token": setup["session_token"]}
            payload = {"profile_id": "0", "question": "refund"}
            assert client.post("/api/retrieval", json=payload).status_code == 403
            assert (
                client.post(
                    "/api/retrieval",
                    json=payload,
                    headers={**headers, "Origin": "https://attacker.example"},
                ).status_code
                == 403
            )
            result = client.post("/api/retrieval", json=payload, headers=headers)
            assert result.status_code == 200
            assert result.json()["baseline"]["trace"]["passages"][0]["document_id"] == "refunds.md"
            for invalid in (
                {"profile_id": "0"},
                {**payload, "path": "/tmp"},
                {**payload, "question": False},
                {**payload, "profile_id": "1"},
            ):
                assert (
                    client.post("/api/retrieval", json=invalid, headers=headers).status_code == 422
                )
            assert (
                client.post(
                    "/api/retrieval", json={**payload, "profile_id": "unknown"}, headers=headers
                ).status_code
                == 404
            )
            assert not client.get("/api/experiments").json()["experiments"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
