"""Unit tests for execute_and_observe (agent_execute.py).

Covers: successful completion, failed job early abort, timeout expiry,
submission denied (policy), studio unavailable, invalid recipe, and
JSON round-trip serialization for all outputs.

Requirements: 13.1, 13.6, 13.7
"""

from __future__ import annotations

import json
from typing import Any

import pytest

import olive_mcp_server.tools.agent_execute as agent_execute
from olive_mcp_server.tools.agent_execute import execute_and_observe

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _json_round_trip(result: dict[str, Any]) -> None:
    """Assert JSON serialization round-trip produces value-equal dict."""
    assert json.loads(json.dumps(result)) == result


# ---------------------------------------------------------------------------
# Test: Successful completion
# ---------------------------------------------------------------------------


def test_successful_completion(monkeypatch: pytest.MonkeyPatch):
    """Submit succeeds, polls twice, job completes with metrics and logs."""
    call_count = {"n": 0}
    submit_kwargs: dict[str, Any] = {}

    def fake_studio_request(method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        if method == "POST" and "/jobs/submit" in path:
            submit_kwargs.update(kwargs)
            return {"job_id": "job-001"}
        # GET status polls
        call_count["n"] += 1
        if call_count["n"] == 1:
            return {"status": "running", "logs": ["step 1"]}
        return {
            "status": "completed",
            "logs": ["step 1", "done"],
            "latestMetrics": {"latency_ms": 45},
            "exitCode": 0,
        }

    monkeypatch.setattr(agent_execute, "studio_request", fake_studio_request)
    monkeypatch.setattr(agent_execute, "_sleep", lambda _: None)

    result = execute_and_observe(recipe={"model_path": "test.onnx"}, timeout=60)

    assert result["status"] == "completed"
    assert result["job_id"] == "job-001"
    assert result["side_effect"] is True
    assert result["timed_out"] is False
    assert result["metrics"] == {"latency_ms": 45}
    assert result["logs"] == ["step 1", "done"]
    assert result["exit_code"] == 0
    assert isinstance(result["elapsed_ms"], int)
    assert isinstance(result["artifact_path_refs"], list)
    # The /jobs/submit call must pass the submission timeout (120s for env setup).
    assert submit_kwargs.get("timeout") == 120.0
    _json_round_trip(result)


# ---------------------------------------------------------------------------
# Test: Failed job early abort
# ---------------------------------------------------------------------------


def test_failed_job_early_abort(monkeypatch: pytest.MonkeyPatch):
    """Job fails on first poll — stops immediately."""
    poll_count = {"n": 0}

    def fake_studio_request(method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        if method == "POST" and "/jobs/submit" in path:
            return {"job_id": "job-002"}
        poll_count["n"] += 1
        return {"status": "failed", "logs": ["error line"], "exitCode": 1}

    monkeypatch.setattr(agent_execute, "studio_request", fake_studio_request)
    monkeypatch.setattr(agent_execute, "_sleep", lambda _: None)

    result = execute_and_observe(recipe={"model_path": "bad.onnx"}, timeout=60)

    assert result["status"] == "failed"
    assert result["side_effect"] is True
    assert result["exit_code"] == 1
    assert result["logs"] == ["error line"]
    assert poll_count["n"] == 1  # stopped after 1 poll
    _json_round_trip(result)


def test_failed_job_records_bounded_failure_context(monkeypatch: pytest.MonkeyPatch):
    """Failed terminal results derive a useful failure string for session state."""
    recorded: dict[str, Any] = {}

    def fake_studio_request(method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        if method == "POST" and "/jobs/submit" in path:
            return {"job_id": "job-002b"}
        return {
            "status": "failed",
            "logs": [
                "setup complete",
                "RuntimeError: calibration exploded",
                "See /tmp/output/error.log",
            ],
            "exitCode": 17,
        }

    def fake_ensure_session(session_id: str | None) -> tuple[str, dict[str, Any]]:
        assert session_id == "session-1"
        return "session-1", {"sessionId": "session-1", "attemptCount": 0}

    def fake_record_attempt(session_id: str, **kwargs: Any) -> dict[str, Any]:
        recorded["session_id"] = session_id
        recorded.update(kwargs)
        return {"sessionId": session_id, "attemptCount": 1}

    monkeypatch.setattr(agent_execute, "studio_request", fake_studio_request)
    monkeypatch.setattr(agent_execute, "_sleep", lambda _: None)
    monkeypatch.setenv("OLIVE_STUDIO_API_URL", "http://127.0.0.1:3000")
    monkeypatch.setattr("olive_mcp_server.tools.studio_loopback._ensure_session", fake_ensure_session)
    monkeypatch.setattr("olive_mcp_server.tools.studio_loopback._record_attempt", fake_record_attempt)

    result = execute_and_observe(
        recipe={"model_path": "bad.onnx"},
        timeout=60,
        session_id="session-1",
    )

    assert result["status"] == "failed"
    assert result["session_id"] == "session-1"
    assert recorded["session_id"] == "session-1"
    assert recorded["success"] is False
    assert recorded["failure"] == (
        "status=failed; exit_code=17; "
        "log_tail=setup complete | RuntimeError: calibration exploded | See /tmp/output/error.log"
    )


def test_session_update_error_preserves_execution_result(monkeypatch: pytest.MonkeyPatch):
    """Bookkeeping failures stay secondary to the execution outcome."""

    def fake_studio_request(method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        if method == "POST" and "/jobs/submit" in path:
            return {"job_id": "job-bookkeeping"}
        return {"status": "completed", "logs": ["done"], "exitCode": 0}

    def fake_ensure_session(session_id: str | None) -> tuple[str, dict[str, Any]]:
        assert session_id == "session-2"
        return "session-2", {"sessionId": "session-2", "attemptCount": 0}

    monkeypatch.setattr(agent_execute, "studio_request", fake_studio_request)
    monkeypatch.setattr(agent_execute, "_sleep", lambda _: None)
    monkeypatch.setenv("OLIVE_STUDIO_API_URL", "http://127.0.0.1:3000")
    monkeypatch.setattr("olive_mcp_server.tools.studio_loopback._ensure_session", fake_ensure_session)
    monkeypatch.setattr(
        "olive_mcp_server.tools.studio_loopback._record_attempt",
        lambda *args, **kwargs: {
            "error": "studio_unavailable",
            "message": "Olive Studio bridge timed out.",
            "detail": "timeout_seconds=5.0",
        },
    )

    result = execute_and_observe(
        recipe={"model_path": "test.onnx"},
        timeout=60,
        session_id="session-2",
    )

    assert result["status"] == "completed"
    assert result["job_id"] == "job-bookkeeping"
    assert result["side_effect"] is True
    assert result["session_id"] == "session-2"
    assert result["session_update_error"] == {
        "error": "studio_unavailable",
        "message": "Olive Studio bridge timed out.",
        "detail": "timeout_seconds=5.0",
    }
    _json_round_trip(result)


# ---------------------------------------------------------------------------
# Test: Timeout expiry
# ---------------------------------------------------------------------------


def test_timeout_expiry(monkeypatch: pytest.MonkeyPatch):
    """Job never reaches terminal state — timeout fires."""
    # Simulate time: starts at 0, increments 100 each call to exceed any timeout
    time_values = iter([0, 0, 100, 200, 300])

    def fake_monotonic() -> float:
        return next(time_values, 9999)

    def fake_studio_request(method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        if method == "POST" and "/jobs/submit" in path:
            return {"job_id": "job-003"}
        return {"status": "running"}

    monkeypatch.setattr(agent_execute, "studio_request", fake_studio_request)
    monkeypatch.setattr(agent_execute, "_sleep", lambda _: None)
    monkeypatch.setattr(agent_execute, "_monotonic", fake_monotonic)

    result = execute_and_observe(recipe={"model_path": "slow.onnx"}, timeout=10)

    assert result["timed_out"] is True
    assert result["side_effect"] is True
    assert result["status"] == "running"
    _json_round_trip(result)


def test_derive_failure_text_captures_timeout_and_log_tail():
    """Timeout results include status, timeout, exit code, and bounded log tail."""
    failure = agent_execute._derive_failure_text(
        {
            "status": "running",
            "timed_out": True,
            "exit_code": 124,
            "logs": [f"line-{index}" for index in range(8)],
        }
    )

    assert failure == (
        "status=running; timed_out=true; exit_code=124; log_tail=line-3 | line-4 | line-5 | line-6 | line-7"
    )


# ---------------------------------------------------------------------------
# Test: Submission denied (policy)
# ---------------------------------------------------------------------------


def test_submission_denied_policy(monkeypatch: pytest.MonkeyPatch):
    """Studio returns forbidden — maps to submission_denied, no side_effect."""

    def fake_studio_request(method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        return {"error": "forbidden", "message": "Agent policy forbids submission"}

    monkeypatch.setattr(agent_execute, "studio_request", fake_studio_request)

    result = execute_and_observe(recipe={"model_path": "test.onnx"})

    assert result["error"] == "submission_denied"
    assert "side_effect" not in result
    _json_round_trip(result)


# ---------------------------------------------------------------------------
# Test: Studio unavailable
# ---------------------------------------------------------------------------


def test_studio_unavailable(monkeypatch: pytest.MonkeyPatch):
    """Studio returns studio_unavailable — pass through, no side_effect."""

    def fake_studio_request(method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        return {"error": "studio_unavailable", "message": "not reachable"}

    monkeypatch.setattr(agent_execute, "studio_request", fake_studio_request)

    result = execute_and_observe(recipe={"model_path": "test.onnx"})

    assert result["error"] == "studio_unavailable"
    assert "side_effect" not in result
    _json_round_trip(result)


# ---------------------------------------------------------------------------
# Test: Invalid recipe
# ---------------------------------------------------------------------------


def test_invalid_recipe(monkeypatch: pytest.MonkeyPatch):
    """Studio returns validation_error — maps to invalid_recipe, no side_effect."""

    def fake_studio_request(method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        return {"error": "validation_error", "message": "missing model_path"}

    monkeypatch.setattr(agent_execute, "studio_request", fake_studio_request)

    result = execute_and_observe(recipe={})

    assert result["error"] == "invalid_recipe"
    assert "side_effect" not in result
    _json_round_trip(result)


# ---------------------------------------------------------------------------
# Test: JSON round-trip for success cases (extra verification)
# ---------------------------------------------------------------------------


def test_json_round_trip_comprehensive(monkeypatch: pytest.MonkeyPatch):
    """All success-path outputs survive JSON serialization round-trip."""

    def fake_studio_request(method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        if method == "POST" and "/jobs/submit" in path:
            return {"job_id": "job-rt"}
        return {
            "status": "completed",
            "logs": ["saved output/model.onnx", "done"],
            "latestMetrics": {"latency_ms": 12.5, "model_size_mb": 100.0},
            "exitCode": 0,
        }

    monkeypatch.setattr(agent_execute, "studio_request", fake_studio_request)
    monkeypatch.setattr(agent_execute, "_sleep", lambda _: None)

    result = execute_and_observe(recipe={"model_path": "m.onnx"}, timeout=60)

    # Core round-trip
    _json_round_trip(result)

    # Also verify artifact refs detected from logs
    assert "model.onnx" in result["artifact_path_refs"]


# ---------------------------------------------------------------------------
# Test: Artifact path extraction from logs
# ---------------------------------------------------------------------------


def test_artifact_path_extraction(monkeypatch: pytest.MonkeyPatch):
    """Artifact basenames are extracted from log lines."""

    def fake_studio_request(method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        if method == "POST" and "/jobs/submit" in path:
            return {"job_id": "job-art"}
        return {
            "status": "completed",
            "logs": [
                "Saved model to C:\\Users\\dev\\output\\optimized.onnx",
                "Wrote config /tmp/out/config.json",
                "No artifact here",
            ],
            "exitCode": 0,
        }

    monkeypatch.setattr(agent_execute, "studio_request", fake_studio_request)
    monkeypatch.setattr(agent_execute.time, "sleep", lambda _: None)

    result = execute_and_observe(recipe={"model_path": "x.onnx"}, timeout=60)

    assert "optimized.onnx" in result["artifact_path_refs"]
    assert "config.json" in result["artifact_path_refs"]
    _json_round_trip(result)


# ---------------------------------------------------------------------------
# Test: Poll error during status fetch (partial result)
# ---------------------------------------------------------------------------


def test_poll_error_returns_partial(monkeypatch: pytest.MonkeyPatch):
    """If status poll fails mid-run, return partial result with poll_error."""
    call_count = {"n": 0}

    def fake_studio_request(method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        if method == "POST" and "/jobs/submit" in path:
            return {"job_id": "job-poll-err"}
        call_count["n"] += 1
        if call_count["n"] == 1:
            return {"status": "running", "logs": ["step 1"]}
        # Second poll returns error
        return {"error": "studio_unavailable", "message": "connection lost"}

    monkeypatch.setattr(agent_execute, "studio_request", fake_studio_request)
    monkeypatch.setattr(agent_execute.time, "sleep", lambda _: None)

    result = execute_and_observe(recipe={"model_path": "x.onnx"}, timeout=60)

    assert result["side_effect"] is True
    assert result["job_id"] == "job-poll-err"
    assert "poll_error" in result
    _json_round_trip(result)


# ---------------------------------------------------------------------------
# Test: Internal exception produces internal_error
# ---------------------------------------------------------------------------


def test_internal_exception(monkeypatch: pytest.MonkeyPatch):
    """Unexpected exception inside tool produces internal_error response."""

    def fake_studio_request(method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("unexpected boom")

    monkeypatch.setattr(agent_execute, "studio_request", fake_studio_request)

    result = execute_and_observe(recipe={"model_path": "x.onnx"})

    assert result["error"] == "internal_error"
    assert "RuntimeError" in result["message"]
    assert "side_effect" not in result
    _json_round_trip(result)


# ---------------------------------------------------------------------------
# Test: Timeout clamping
# ---------------------------------------------------------------------------


def test_timeout_clamping_default():
    """None -> 600, below 10 -> 10, above 1800 -> 1800."""
    from olive_mcp_server.tools.agent_execute import _clamp_timeout

    assert _clamp_timeout(None) == 600
    assert _clamp_timeout(5) == 10
    assert _clamp_timeout(-100) == 10
    assert _clamp_timeout(0) == 10
    assert _clamp_timeout(10) == 10
    assert _clamp_timeout(1800) == 1800
    assert _clamp_timeout(9999) == 1800
    assert _clamp_timeout(300) == 300


# ---------------------------------------------------------------------------
# Test: Poll backoff
# ---------------------------------------------------------------------------


def test_next_poll_interval_doubles_then_caps():
    """Backoff doubles from 2s and caps at 30s."""
    from olive_mcp_server.tools.agent_execute import _next_poll_interval

    assert _next_poll_interval(2.0) == 4.0
    assert _next_poll_interval(4.0) == 8.0
    assert _next_poll_interval(8.0) == 16.0
    assert _next_poll_interval(16.0) == 30.0
    assert _next_poll_interval(30.0) == 30.0


def test_poll_sleep_uses_exponential_backoff(monkeypatch: pytest.MonkeyPatch):
    """Long-running jobs sleep with growing intervals, not a fixed 2s cadence."""
    call_count = {"n": 0}
    sleeps: list[float] = []

    def fake_studio_request(method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        if method == "POST" and "/jobs/submit" in path:
            return {"job_id": "job-backoff"}
        call_count["n"] += 1
        if call_count["n"] < 5:
            return {"status": "running"}
        return {"status": "completed", "logs": [], "exitCode": 0}

    monkeypatch.setattr(agent_execute, "studio_request", fake_studio_request)
    monkeypatch.setattr(agent_execute, "_sleep", lambda seconds: sleeps.append(seconds))

    result = execute_and_observe(recipe={"model_path": "slow.onnx"}, timeout=600)

    assert result["status"] == "completed"
    assert sleeps == [2.0, 4.0, 8.0, 16.0]


# ---------------------------------------------------------------------------
# Test: Logs truncated to last _MAX_LOG_ENTRIES (200) entries
# ---------------------------------------------------------------------------


def test_logs_truncated_to_last_200(monkeypatch: pytest.MonkeyPatch):
    """When the job returns more than 200 log lines, only the last 200 are kept."""
    all_logs = [f"line-{i}" for i in range(250)]

    def fake_studio_request(method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        if method == "POST" and "/jobs/submit" in path:
            return {"job_id": "job-trunc"}
        return {"status": "completed", "logs": all_logs, "exitCode": 0}

    monkeypatch.setattr(agent_execute, "studio_request", fake_studio_request)
    monkeypatch.setattr(agent_execute, "_sleep", lambda _: None)

    result = execute_and_observe(recipe={"model_path": "m.onnx"}, timeout=60)

    assert result["status"] == "completed"
    assert len(result["logs"]) == 200
    assert result["logs"][0] == "line-50"
    assert result["logs"][-1] == "line-249"
    _json_round_trip(result)


# ---------------------------------------------------------------------------
# Test: Artifact refs capped at 20
# ---------------------------------------------------------------------------


def test_artifact_refs_capped_at_20(monkeypatch: pytest.MonkeyPatch):
    """When logs contain more than 20 artifact references, only 20 are returned."""
    logs = [f"saved output/model_{i}.onnx" for i in range(25)]

    def fake_studio_request(method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        if method == "POST" and "/jobs/submit" in path:
            return {"job_id": "job-arts"}
        return {"status": "completed", "logs": logs, "exitCode": 0}

    monkeypatch.setattr(agent_execute, "studio_request", fake_studio_request)
    monkeypatch.setattr(agent_execute, "_sleep", lambda _: None)

    result = execute_and_observe(recipe={"model_path": "m.onnx"}, timeout=60)

    assert result["status"] == "completed"
    assert len(result["artifact_path_refs"]) == 20
    _json_round_trip(result)


# ---------------------------------------------------------------------------
# Test: mcp_access_disabled maps to submission_denied
# ---------------------------------------------------------------------------


def test_mcp_access_disabled_maps_to_submission_denied(monkeypatch: pytest.MonkeyPatch):
    """Studio returns mcp_access_disabled — maps to submission_denied, no side_effect."""

    def fake_studio_request(method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        return {"error": "mcp_access_disabled", "message": "Agent access is disabled"}

    monkeypatch.setattr(agent_execute, "studio_request", fake_studio_request)

    result = execute_and_observe(recipe={"model_path": "test.onnx"})

    assert result["error"] == "submission_denied"
    assert "side_effect" not in result
    _json_round_trip(result)
