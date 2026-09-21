"""Property-based tests for execute_and_observe (agent_execute module).

Feature: v0.3-agent-mcp-tools

Uses hypothesis to verify universal correctness properties that must hold
across all valid inputs.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from olive_mcp_server.tools.agent_execute import _clamp_timeout, execute_and_observe

# ---------------------------------------------------------------------------
# Feature: v0.3-agent-mcp-tools, Property 1: Timeout Clamping Invariant
#
# For any integer T (including negative, zero, large), the clamped result is
# in [10, 1800]. When T is None, result is 600.
#
# Validates: Requirements 1.10, 1.11, 1.12
# ---------------------------------------------------------------------------


class TestTimeoutClampingInvariant:
    """Property 1: Timeout clamping invariant."""

    @given(timeout=st.integers(min_value=-10000, max_value=100000))
    @settings(max_examples=100)
    def test_clamped_timeout_within_bounds(self, timeout: int) -> None:
        """For any integer T, _clamp_timeout(T) is in [10, 1800]."""
        result = _clamp_timeout(timeout)
        assert 10 <= result <= 1800, f"_clamp_timeout({timeout}) = {result}, expected in [10, 1800]"

    def test_none_defaults_to_600(self) -> None:
        """When timeout is None, _clamp_timeout returns 600."""
        assert _clamp_timeout(None) == 600

    @given(timeout=st.integers(min_value=10, max_value=1800))
    @settings(max_examples=50)
    def test_values_in_range_unchanged(self, timeout: int) -> None:
        """Values already in [10, 1800] pass through unchanged."""
        assert _clamp_timeout(timeout) == timeout

    @given(timeout=st.integers(min_value=1801, max_value=100000))
    @settings(max_examples=50)
    def test_values_above_ceiling_clamp_to_1800(self, timeout: int) -> None:
        """Values above 1800 clamp to 1800."""
        assert _clamp_timeout(timeout) == 1800

    @given(timeout=st.integers(min_value=-10000, max_value=9))
    @settings(max_examples=50)
    def test_values_below_floor_clamp_to_10(self, timeout: int) -> None:
        """Values below 10 clamp to 10."""
        assert _clamp_timeout(timeout) == 10


# ---------------------------------------------------------------------------
# Feature: v0.3-agent-mcp-tools, Property 2: Side-Effect Field Correctness
#
# For any successful submission (reaches polling), response contains
# "side_effect": True. For any pre-submission error, response does NOT
# contain "side_effect" key.
#
# Validates: Requirements 2.3
# ---------------------------------------------------------------------------


def _make_submission_success_mock(
    job_id: str = "test-job-123",
    terminal_status: str = "completed",
) -> Any:
    """Create a mock for studio_request that succeeds on submit and returns terminal on first poll."""

    def mock_studio_request(
        method: str,
        path: str,
        *,
        body: dict[str, Any] | None = None,
        timeout: float = 5.0,
    ) -> dict[str, Any]:
        if method == "POST" and "/jobs/submit" in path:
            return {"job_id": job_id}
        if method == "GET" and f"/status/{job_id}" in path:
            return {
                "status": terminal_status,
                "exitCode": 0,
                "logs": ["done"],
                "latestMetrics": {"latency_ms": 42.0},
            }
        return {"error": "unexpected_call", "message": f"Unexpected: {method} {path}"}

    return mock_studio_request


def _make_submission_error_mock(error_code: str) -> Any:
    """Create a mock for studio_request that returns an error on submission."""

    def mock_studio_request(
        method: str,
        path: str,
        *,
        body: dict[str, Any] | None = None,
        timeout: float = 5.0,
    ) -> dict[str, Any]:
        if method == "POST" and "/jobs/submit" in path:
            return {"error": error_code, "message": f"Test error: {error_code}"}
        return {"error": "unexpected_call", "message": f"Unexpected: {method} {path}"}

    return mock_studio_request


class TestSideEffectFieldCorrectness:
    """Property 2: Side-effect field presence/absence."""

    @pytest.mark.parametrize("terminal_status", ["completed", "failed", "cancelled"])
    def test_successful_submission_has_side_effect_true(self, terminal_status: str) -> None:
        """When submission succeeds and polling reaches terminal, side_effect is True."""
        mock = _make_submission_success_mock(terminal_status=terminal_status)
        with patch("olive_mcp_server.tools.agent_execute.studio_request", side_effect=mock):
            result = execute_and_observe(recipe={"input_model": {"type": "onnx"}}, timeout=10)

        assert "side_effect" in result, (
            f"Expected 'side_effect' key in successful result, got keys: {list(result.keys())}"
        )
        assert result["side_effect"] is True

        # JSON round-trip (Property 10 piggyback)
        assert json.loads(json.dumps(result)) == result

    @pytest.mark.parametrize("error_code", ["invalid_recipe", "submission_denied", "studio_unavailable"])
    def test_pre_submission_error_no_side_effect(self, error_code: str) -> None:
        """When submission fails (error before polling), side_effect key is absent."""
        mock = _make_submission_error_mock(error_code)
        with patch("olive_mcp_server.tools.agent_execute.studio_request", side_effect=mock):
            result = execute_and_observe(recipe={"input_model": {"type": "onnx"}}, timeout=10)

        assert "side_effect" not in result, (
            f"Expected no 'side_effect' key in error result for {error_code}, got keys: {list(result.keys())}"
        )

        # JSON round-trip (Property 10 piggyback)
        assert json.loads(json.dumps(result)) == result

    def test_poll_error_still_has_side_effect(self) -> None:
        """When submission succeeds but poll fails, side_effect is still True (job was submitted)."""

        def mock_studio_request(
            method: str,
            path: str,
            *,
            body: dict[str, Any] | None = None,
            timeout: float = 5.0,
        ) -> dict[str, Any]:
            if method == "POST" and "/jobs/submit" in path:
                return {"job_id": "poll-err-job"}
            # All poll attempts fail
            return {"error": "studio_unavailable", "message": "Studio went down"}

        with patch(
            "olive_mcp_server.tools.agent_execute.studio_request",
            side_effect=mock_studio_request,
        ):
            # Use short timeout so test doesn't take long
            result = execute_and_observe(recipe={"test": True}, timeout=10)

        # Job was submitted, so side_effect must be True
        assert result.get("side_effect") is True

    def test_missing_job_id_in_response_has_side_effect_true(self) -> None:
        """When submission response lacks job_id, side_effect is True (uncertain — job may be running)."""

        def mock_studio_request(
            method: str,
            path: str,
            *,
            body: dict[str, Any] | None = None,
            timeout: float = 5.0,
        ) -> dict[str, Any]:
            if method == "POST" and "/jobs/submit" in path:
                # Success response but missing job_id
                return {"status": "ok"}
            return {"error": "unexpected_call", "message": "Unexpected"}

        with patch(
            "olive_mcp_server.tools.agent_execute.studio_request",
            side_effect=mock_studio_request,
        ):
            result = execute_and_observe(recipe={"test": True}, timeout=10)

        # The recipe was submitted but job_id is unknown — uncertain side effect.
        assert result.get("side_effect") is True
        assert result.get("error") == "invalid_bridge_response"
