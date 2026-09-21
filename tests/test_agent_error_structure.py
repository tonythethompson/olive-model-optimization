# Feature: v0.3-agent-mcp-tools, Property 9: Error Structure Invariant
# Feature: v0.3-agent-mcp-tools, Property 10: JSON Serialization Round-Trip
"""Universal error structure and JSON round-trip tests for all 5 new agent tools.

Property 9: For every error returned by any of the 5 new tool functions, the response
must be a dict containing:
  - "error" key with a non-empty string matching ^[a-z][a-z0-9_]*$
  - "message" key with a non-empty string

Property 10: For any valid output dictionary (including errors), serializing to JSON
and parsing back produces a value-equal dictionary.

Validates: Requirements 12.1, 13.7
"""

from __future__ import annotations

import json
import re
from typing import Any
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ERROR_CODE_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


def assert_valid_error_structure(result: dict[str, Any]) -> None:
    """Assert that a result dict conforms to the universal error structure.

    Property 9: Error Structure Invariant.
    """
    assert isinstance(result, dict), f"Error response must be a dict, got {type(result)}"
    assert "error" in result, f"Error response missing 'error' key: {result}"
    assert "message" in result, f"Error response missing 'message' key: {result}"

    error_code = result["error"]
    assert isinstance(error_code, str), f"'error' must be a string, got {type(error_code)}"
    assert len(error_code) > 0, "'error' must be non-empty"
    assert _ERROR_CODE_PATTERN.match(error_code), f"'error' must match ^[a-z][a-z0-9_]*$, got: {error_code!r}"

    message = result["message"]
    assert isinstance(message, str), f"'message' must be a string, got {type(message)}"
    assert len(message) > 0, "'message' must be non-empty"


def assert_json_round_trip(result: dict[str, Any]) -> None:
    """Assert that a result dict survives JSON serialization round-trip.

    Property 10: JSON Serialization Round-Trip. Uses ``allow_nan=False`` so
    NaN, Infinity, and -Infinity are rejected as invalid JSON.
    """
    serialized = json.dumps(result, allow_nan=False)
    deserialized = json.loads(serialized)
    assert deserialized == result, f"JSON round-trip failed:\n  original: {result}\n  after: {deserialized}"


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


def _studio_unavailable_mock(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Mock studio_request that always returns studio_unavailable."""
    return {"error": "studio_unavailable", "message": "Studio is not reachable"}


def _troubleshoot_mock(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Mock troubleshoot_olive_error that returns a minimal diagnosis."""
    return {
        "matched_entry": None,
        "root_cause": "",
        "workaround": "",
    }


# ---------------------------------------------------------------------------
# Parametrized error cases: (tool_module, tool_function, args, kwargs, mocks, expected_error_code)
# ---------------------------------------------------------------------------

# We use lazy parametrize IDs and callables to avoid importing missing modules at collection time.


def _case_model_info_empty_id() -> dict[str, Any]:
    """get_model_info('') -> invalid_model_id"""
    from olive_mcp_server.tools.agent_model_info import get_model_info

    return get_model_info("")


def _case_execute_studio_unavailable() -> dict[str, Any]:
    """execute_and_observe({}, timeout=10) with studio_unavailable mock -> studio_unavailable"""
    from olive_mcp_server.tools.agent_execute import execute_and_observe

    with patch(
        "olive_mcp_server.tools.agent_execute.studio_request",
        side_effect=_studio_unavailable_mock,
    ):
        return execute_and_observe({}, timeout=10)


def _case_plan_optimization_empty() -> dict[str, Any]:
    """plan_optimization('') -> invalid_input"""
    from olive_mcp_server.tools.agent_planner import plan_optimization  # noqa: F811

    return plan_optimization("")


def _case_plan_optimization_unparseable() -> dict[str, Any]:
    """plan_optimization('hello world no keywords') -> unparseable_intent"""
    from olive_mcp_server.tools.agent_planner import plan_optimization  # noqa: F811

    return plan_optimization("hello world no keywords")


def _case_diagnose_empty_error() -> dict[str, Any]:
    """diagnose_and_fix('', {}) -> invalid_input"""
    from olive_mcp_server.tools.agent_diagnosis import diagnose_and_fix

    return diagnose_and_fix("", {})


def _case_diagnose_oversized_error() -> dict[str, Any]:
    """diagnose_and_fix('x' * 5000, {}) -> invalid_input"""
    from olive_mcp_server.tools.agent_diagnosis import diagnose_and_fix

    with (
        patch(
            "olive_mcp_server.tools.agent_diagnosis.studio_request",
            side_effect=_studio_unavailable_mock,
        ),
        patch(
            "olive_mcp_server.tools.agent_diagnosis.troubleshoot_olive_error",
            side_effect=_troubleshoot_mock,
        ),
    ):
        return diagnose_and_fix("x" * 5000, {})


def _case_compare_too_few_jobs() -> dict[str, Any]:
    """compare_results(['a']) -> invalid_job_count"""
    from olive_mcp_server.tools.agent_compare import compare_results

    return compare_results(["a"])


def _case_compare_invalid_job_id() -> dict[str, Any]:
    """compare_results(['a!!!', 'b!!!']) -> invalid_job_id"""
    from olive_mcp_server.tools.agent_compare import compare_results

    return compare_results(["a!!!", "b!!!"])


# ---------------------------------------------------------------------------
# Parametrized test: Property 9 - Error Structure Invariant
# ---------------------------------------------------------------------------

# Cases that DON'T require plan_optimization (which may not exist yet)
_IMPLEMENTED_ERROR_CASES = [
    pytest.param(
        _case_model_info_empty_id,
        "invalid_model_id",
        id="get_model_info_empty_id",
    ),
    pytest.param(
        _case_execute_studio_unavailable,
        "studio_unavailable",
        id="execute_and_observe_studio_unavailable",
    ),
    pytest.param(
        _case_diagnose_empty_error,
        "invalid_input",
        id="diagnose_and_fix_empty_error",
    ),
    pytest.param(
        _case_diagnose_oversized_error,
        "invalid_input",
        id="diagnose_and_fix_oversized_error",
    ),
    pytest.param(
        _case_compare_too_few_jobs,
        "invalid_job_count",
        id="compare_results_too_few_jobs",
    ),
    pytest.param(
        _case_compare_invalid_job_id,
        "invalid_job_id",
        id="compare_results_invalid_job_id",
    ),
]

# Cases that require plan_optimization (agent_planner.py)
from olive_mcp_server.tools.agent_planner import plan_optimization  # noqa: F401, E402

_PLANNER_ERROR_CASES = [
    pytest.param(
        _case_plan_optimization_empty,
        "invalid_input",
        id="plan_optimization_empty_intent",
    ),
    pytest.param(
        _case_plan_optimization_unparseable,
        "unparseable_intent",
        id="plan_optimization_unparseable_intent",
    ),
]

_ALL_ERROR_CASES = _IMPLEMENTED_ERROR_CASES + _PLANNER_ERROR_CASES


class TestErrorStructureInvariant:
    """Property 9: Error Structure Invariant.

    For every error returned by any of the 5 new tool functions, the response
    must be a dict containing a non-empty "error" (snake_case) and a non-empty
    "message".

    Validates: Requirements 12.1
    """

    @pytest.mark.parametrize("trigger_fn, expected_code", _ALL_ERROR_CASES)
    def test_error_structure_matches_invariant(
        self,
        trigger_fn: Any,
        expected_code: str,
    ) -> None:
        """Each error response satisfies the universal error structure contract."""
        result = trigger_fn()
        assert_valid_error_structure(result)
        assert result["error"] == expected_code, f"Expected error code {expected_code!r}, got {result['error']!r}"

    @pytest.mark.parametrize("trigger_fn, expected_code", _ALL_ERROR_CASES)
    def test_error_json_round_trip(
        self,
        trigger_fn: Any,
        expected_code: str,
    ) -> None:
        """Property 10: Every error response survives JSON serialization round-trip.

        Validates: Requirements 13.7
        """
        result = trigger_fn()
        assert_json_round_trip(result)


# ---------------------------------------------------------------------------
# Additional Property 10 verification: non-error outputs round-trip too
# ---------------------------------------------------------------------------


class TestJsonRoundTripForNonErrors:
    """Property 10: JSON Serialization Round-Trip for non-error tool outputs.

    Already covered in individual test files; this verifies a few representative
    successful outputs as well.
    """

    def test_compare_results_success_round_trips(self) -> None:
        """compare_results with valid jobs produces JSON-round-trippable output."""
        from olive_mcp_server.tools.agent_compare import compare_results

        def _mock_request(method: str, path: str, **kwargs: Any) -> dict[str, Any]:
            if "job-a" in path:
                return {
                    "id": "job-a",
                    "status": "completed",
                    "latestMetrics": {"latency_ms": 50.0, "model_size_mb": 200.0, "accuracy": 0.9},
                }
            if "job-b" in path:
                return {
                    "id": "job-b",
                    "status": "completed",
                    "latestMetrics": {"latency_ms": 100.0, "model_size_mb": 150.0, "accuracy": 0.85},
                }
            return {"error": "not_found", "message": "Job not found"}

        with patch(
            "olive_mcp_server.tools.agent_compare.studio_request",
            side_effect=_mock_request,
        ):
            result = compare_results(["job-a", "job-b"], "balanced")

        assert "error" not in result
        assert_json_round_trip(result)

    def test_model_info_heuristic_round_trips(self) -> None:
        """get_model_info with HF failure (heuristic path) round-trips."""
        from olive_mcp_server.tools.agent_model_info import get_model_info

        with patch(
            "olive_mcp_server.tools.agent_model_info.requests.get",
            side_effect=Exception("network down"),
        ):
            result = get_model_info("meta-llama/Llama-2-7B")

        assert "error" not in result
        assert_json_round_trip(result)

    def test_diagnose_and_fix_no_match_round_trips(self) -> None:
        """diagnose_and_fix with no KB match produces round-trippable output."""
        from olive_mcp_server.tools.agent_diagnosis import diagnose_and_fix

        with (
            patch(
                "olive_mcp_server.tools.agent_diagnosis.studio_request",
                side_effect=_studio_unavailable_mock,
            ),
            patch(
                "olive_mcp_server.tools.agent_diagnosis.troubleshoot_olive_error",
                return_value={
                    "matched_entry": None,
                    "root_cause": "",
                    "workaround": "",
                },
            ),
        ):
            result = diagnose_and_fix("RuntimeError: CUDA OOM", {"passes": {}})

        assert "error" not in result
        assert_json_round_trip(result)
