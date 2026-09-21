"""Phase 2 read-only job inspection tools."""

from __future__ import annotations

import pytest

from olive_mcp_server.mcp_server import _TOOL_IMPORTS, call_tool
from olive_mcp_server.tools import studio_jobs


def test_job_tools_registered():
    for name in (
        "list_optimization_jobs",
        "get_optimization_job",
        "get_optimization_results",
    ):
        assert name in _TOOL_IMPORTS


def test_list_jobs_studio_unavailable(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("OLIVE_STUDIO_API_URL", raising=False)
    result = studio_jobs.list_optimization_jobs()
    assert result.get("error") == "studio_unavailable"


def test_list_jobs_ok(monkeypatch: pytest.MonkeyPatch):
    def fake_request(method, path, **_kw):
        assert method == "GET"
        assert path == "/api/olive/jobs"
        return {
            "ok": True,
            "count": 2,
            "jobs": [
                {"id": "a", "status": "completed", "exitCode": 0},
                {"id": "b", "status": "running", "exitCode": None},
            ],
        }

    monkeypatch.setattr(studio_jobs, "studio_request", fake_request)
    result = studio_jobs.list_optimization_jobs(limit=1)
    assert result["count"] == 1
    assert result["total"] == 2
    assert result["jobs"][0]["id"] == "a"
    assert result["side_effect"] is False


def test_get_job_not_found(monkeypatch: pytest.MonkeyPatch):
    def fake_request(method, path, **_kw):
        """Return a job-not-found error response for a simulated request."""
        return {"error": "Job not found"}

    monkeypatch.setattr(studio_jobs, "studio_request", fake_request)
    result = studio_jobs.get_optimization_job("missing")
    assert result["error"] == "job_not_found"


def test_get_job_ok(monkeypatch: pytest.MonkeyPatch):
    def fake_request(method, path, **_kw):
        assert path.endswith("/jid-1")
        return {
            "id": "jid-1",
            "status": "completed",
            "exitCode": 0,
            "logs": ["line1", "line2"],
            "logsTruncated": False,
            "latestMetrics": {"gpu": 1},
            "finishedAt": 123,
        }

    monkeypatch.setattr(studio_jobs, "studio_request", fake_request)
    result = studio_jobs.get_optimization_job("jid-1")
    assert result["id"] == "jid-1"
    assert result["status"] == "completed"
    assert result["terminal"] is True
    assert result["log_count"] == 2
    assert result["side_effect"] is False


def test_get_results_metadata_only(monkeypatch: pytest.MonkeyPatch):
    def fake_request(method, path, **_kw):
        """
        Provide a completed optimization job fixture for request tests.

        Returns:
                dict: A completed job with logs and no latest metrics.
        """
        return {
            "id": "jid-2",
            "status": "completed",
            "exitCode": 0,
            "logs": [
                "writing model to D:\\out\\model.onnx",
                "done",
            ],
            "latestMetrics": None,
        }

    monkeypatch.setattr(studio_jobs, "studio_request", fake_request)
    monkeypatch.delenv("OLIVE_MCP_ALLOW_ABSOLUTE_ARTIFACT_PATHS", raising=False)
    result = studio_jobs.get_optimization_results("jid-2", log_tail=1)
    assert result["log_tail"] == ["done"]
    assert result["artifact_path_refs"] == ["model.onnx"]
    assert result["artifact_paths_absolute"] is False
    assert "Metadata only" in result["note"]
    assert result["side_effect"] is False


def test_results_strips_absolute_paths_to_basenames(monkeypatch: pytest.MonkeyPatch):
    def fake_request(method, path, **kw):
        return {
            "id": "jid-3",
            "status": "completed",
            "logs": ["saved C:\\Users\\alice\\out\\model.onnx", "done"],
        }

    monkeypatch.setattr(studio_jobs, "studio_request", fake_request)
    result = studio_jobs.get_optimization_results("jid-3", log_tail=5)
    assert result["artifact_path_refs"] == ["model.onnx"]


def test_get_results_redacts_absolute_paths_by_default(monkeypatch: pytest.MonkeyPatch):
    def fake_request(method, path, **_kw):
        return {
            "id": "jid-redact",
            "status": "completed",
            "exitCode": 0,
            "logs": [
                "saved /home/alice/models/out.onnx",
                "also C:\\Users\\bob\\cache\\weights.safetensors",
                "relative models/foo.onnx kept",
            ],
            "latestMetrics": None,
        }

    monkeypatch.setattr(studio_jobs, "studio_request", fake_request)
    monkeypatch.delenv("OLIVE_MCP_ALLOW_ABSOLUTE_ARTIFACT_PATHS", raising=False)
    result = studio_jobs.get_optimization_results("jid-redact", log_tail=3)
    assert result["artifact_path_refs"] == ["out.onnx", "weights.safetensors", "models/foo.onnx"]
    assert all("alice" not in p and "bob" not in p for p in result["artifact_path_refs"])
    assert all("alice" not in line and "bob" not in line for line in result["log_tail"])
    assert "out.onnx" in result["log_tail"][0]
    assert result["artifact_paths_absolute"] is False


def test_get_results_absolute_paths_require_env_opt_in(monkeypatch: pytest.MonkeyPatch):
    def fake_request(method, path, **_kw):
        return {
            "id": "jid-abs",
            "status": "completed",
            "exitCode": 0,
            "logs": ["wrote /home/alice/out/model.onnx"],
            "latestMetrics": None,
        }

    monkeypatch.setattr(studio_jobs, "studio_request", fake_request)
    monkeypatch.delenv("OLIVE_MCP_ALLOW_ABSOLUTE_ARTIFACT_PATHS", raising=False)
    denied = studio_jobs.get_optimization_results(
        "jid-abs",
        log_tail=1,
        include_absolute_artifact_paths=True,
    )
    assert denied["artifact_path_refs"] == ["model.onnx"]
    assert denied["artifact_paths_absolute"] is False
    assert "OLIVE_MCP_ALLOW_ABSOLUTE_ARTIFACT_PATHS" in denied["note"]

    monkeypatch.setenv("OLIVE_MCP_ALLOW_ABSOLUTE_ARTIFACT_PATHS", "1")
    allowed = studio_jobs.get_optimization_results(
        "jid-abs",
        log_tail=1,
        include_absolute_artifact_paths=True,
    )
    assert allowed["artifact_path_refs"] == ["/home/alice/out/model.onnx"]
    assert allowed["artifact_paths_absolute"] is True
    assert "/home/alice/out/model.onnx" in allowed["log_tail"][0]


def test_get_job_empty_id():
    result = studio_jobs.get_optimization_job("  ")
    assert result["error"] == "invalid_job_id"


def test_call_tool_list_jobs_unavailable(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("OLIVE_STUDIO_API_URL", raising=False)
    result = call_tool("list_optimization_jobs", {})
    assert result.get("error") == "studio_unavailable"


def test_validate_job_ok(monkeypatch: pytest.MonkeyPatch):
    def fake_request(method, path, **kw):
        assert method == "POST"
        assert path == "/api/olive/jobs/validate"
        return {
            "ok": True,
            "valid": True,
            "fingerprint": "abc",
            "provider": "CPUExecutionProvider",
            "errors": [],
            "warnings": [],
            "cudaVersion": "auto",
            "recipe_summary": {"pass_count": 0},
        }

    monkeypatch.setattr(studio_jobs, "studio_request", fake_request)
    result = studio_jobs.validate_optimization_job(recipe={"passes": {}})
    assert result["valid"] is True
    assert result["fingerprint"] == "abc"
    assert result["side_effect"] is False


def test_submit_job_ok(monkeypatch: pytest.MonkeyPatch):
    def fake_request(method, path, **kw):
        assert path == "/api/olive/jobs/submit"
        return {
            "ok": True,
            "job_id": "jid-9",
            "state": "setting_up",
            "fingerprint": "fp",
            "reused": False,
            "submitted_at": "t",
        }

    monkeypatch.setattr(studio_jobs, "studio_request", fake_request)
    result = studio_jobs.submit_optimization_job(recipe={"passes": {}}, idempotency_key="k")
    assert result["ok"] is True
    assert result["job_id"] == "jid-9"
    assert result["side_effect"] is True


def test_cancel_job_ok(monkeypatch: pytest.MonkeyPatch):
    def fake_request(method, path, **kw):
        assert path == "/api/olive/agent/cancel"
        assert kw.get("body", {}).get("client") == "mcp"
        return {"ok": True, "status": "cancelled"}

    monkeypatch.setattr(studio_jobs, "studio_request", fake_request)
    result = studio_jobs.cancel_optimization_job("jid-9")
    assert result["ok"] is True
    assert result["status"] == "cancelled"


def test_cancel_job_reports_refusal(monkeypatch: pytest.MonkeyPatch):
    def fake_request(method, path, **kw):
        """Return a refusal response indicating that job cancellation is disabled."""
        return {"ok": False, "error": "forbidden", "reason": "Job cancellation is disabled"}

    monkeypatch.setattr(studio_jobs, "studio_request", fake_request)
    result = studio_jobs.cancel_optimization_job("jid-9")
    assert result["ok"] is False
    assert result["error"] == "forbidden"


def test_job_id_rejects_path_segments():
    result = studio_jobs.get_optimization_job("../etc/passwd")
    assert result["error"] == "invalid_job_id"
    result2 = studio_jobs.cancel_optimization_job("a/b")
    assert result2["error"] == "invalid_job_id"


def test_phase3_tools_registered():
    for name in (
        "validate_optimization_job",
        "submit_optimization_job",
        "cancel_optimization_job",
    ):
        assert name in _TOOL_IMPORTS
