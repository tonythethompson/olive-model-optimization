"""Tests for get_mcp_capabilities and retrieval mode helpers."""

from __future__ import annotations

import pytest

from olive_mcp_server.mcp_server import _TOOL_IMPORTS, call_tool
from olive_mcp_server.tools.capabilities import TOOLSET_VERSION, get_mcp_capabilities
from olive_mcp_server.tools.retrieval import get_retrieval_mode, get_semantic_budget_ms, run_with_budget


def test_get_mcp_capabilities_registered():
    assert "get_mcp_capabilities" in _TOOL_IMPORTS


def test_get_mcp_capabilities_shape():
    caps = get_mcp_capabilities()
    assert caps["server"]["name"] == "olive-mcp-server"
    assert "version" in caps["server"]
    assert "version" in caps["kb"]
    # Phase 1 ships precomputed indexes with the package.
    assert "shipped" in caps["kb"]["index"]
    assert "available" in caps["semantic"]
    assert "ready" in caps["semantic"]
    assert caps["retrieval"]["default_mode"] in ("auto", "keyword", "semantic")
    assert caps["job_control"]["supported"] is True
    assert caps["job_control"]["inspection"] is True
    assert caps["job_control"]["submission"] is False
    assert "policy" in caps["job_control"]
    assert caps["toolset"]["version"] == TOOLSET_VERSION


def test_capabilities_default_no_network(monkeypatch: pytest.MonkeyPatch):
    """probe_studio=False must not call Studio even when URL is configured."""
    monkeypatch.setenv("OLIVE_STUDIO_API_URL", "http://127.0.0.1:3000")
    import olive_mcp_server.tools.capabilities as caps_mod

    def boom(*_a, **_k):
        raise AssertionError("network should not be used when probe_studio=False")

    monkeypatch.setattr(caps_mod, "_probe_studio", boom)
    monkeypatch.setattr(caps_mod, "_fetch_studio_policy", boom)
    caps = get_mcp_capabilities(probe_studio=False)
    assert caps["studio"]["configured"] is True
    assert caps["studio"]["reachable"] is None
    assert caps["job_control"]["reason"] == "studio_configured_unprobed"
    assert caps["job_control"]["ready"] is False
    assert caps["job_control"]["submission"] is False
    assert caps["job_control"]["policy"] is None


def test_capabilities_probe_loads_policy(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OLIVE_STUDIO_API_URL", "http://127.0.0.1:3000")
    import olive_mcp_server.tools.capabilities as caps_mod

    monkeypatch.setattr(caps_mod, "_probe_studio", lambda: True)
    monkeypatch.setattr(
        caps_mod,
        "_fetch_studio_policy",
        lambda: {
            "mcpAccess": True,
            "allowJobInspection": True,
            "allowJobSubmission": True,
            "allowJobCancellation": False,
        },
    )
    caps = get_mcp_capabilities(probe_studio=True)
    assert caps["studio"]["reachable"] is True
    assert caps["job_control"]["ready"] is True
    assert caps["job_control"]["submission"] is True
    assert caps["job_control"]["cancellation"] is False
    assert caps["job_control"]["policy"]["allowJobSubmission"] is True


def test_call_tool_capabilities():
    result = call_tool("get_mcp_capabilities", {})
    assert isinstance(result, dict)
    assert result["job_control"]["supported"] is True
    assert result["job_control"]["submission"] is False


def test_retrieval_mode_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OLIVE_MCP_RETRIEVAL_MODE", "keyword")
    assert get_retrieval_mode() == "keyword"
    assert get_retrieval_mode("semantic") == "semantic"
    monkeypatch.setenv("OLIVE_MCP_RETRIEVAL_MODE", "nope")
    assert get_retrieval_mode() == "auto"


def test_semantic_budget_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OLIVE_MCP_SEMANTIC_BUDGET_MS", "1234")
    assert get_semantic_budget_ms() == 1234
    monkeypatch.setenv("OLIVE_MCP_SEMANTIC_BUDGET_MS", "not-int")
    assert get_semantic_budget_ms() == 8000


def test_run_with_budget_timeout():
    import time

    def slow():
        """
        Pause briefly before returning a completion marker.

        Returns:
                str: The string "done".
        """
        time.sleep(0.5)
        return "done"

    result, outcome = run_with_budget(slow, budget_ms=50)
    assert outcome == "timeout"
    assert result is None


def test_run_with_budget_ok():
    result, outcome = run_with_budget(lambda: 42, budget_ms=5000)
    assert outcome == "ok"
    assert result == 42


def test_run_with_budget_single_flight_during_timeout(wait_inflight_semantic_clear):
    """Consecutive calls during a timeout must not start a second callable."""
    import time

    started: list[int] = []

    def slow():
        """Simulate a slow operation for budget timeout tests."""
        started.append(1)
        time.sleep(0.4)
        return "done"

    r1, o1 = run_with_budget(slow, budget_ms=50)
    assert o1 == "timeout" and r1 is None

    r2, o2 = run_with_budget(slow, budget_ms=50)
    assert o2 == "busy" and r2 is None
    assert started == [1]

    # After the in-flight work finishes, a new callable may start.
    wait_inflight_semantic_clear()
    r3, o3 = run_with_budget(lambda: "fresh", budget_ms=5000)
    assert o3 == "ok" and r3 == "fresh"
    assert started == [1]


def test_troubleshoot_keyword_mode_has_retrieval():
    result = call_tool(
        "troubleshoot_olive_error",
        {
            "error_message": "CUDA out of memory during quantization",
            "mode": "keyword",
        },
    )
    assert "retrieval" in result
    assert result["retrieval"]["mode"] == "keyword"
    assert result["retrieval"]["effective"] == "keyword"
    assert result["retrieval"]["degraded"] is False


def test_troubleshoot_auto_budget_degraded(monkeypatch: pytest.MonkeyPatch):
    """Force budget timeout path when model is not loaded."""
    import time

    import numpy as np

    import olive_mcp_server.tools.embeddings as emb
    import olive_mcp_server.tools.troubleshooting as ts

    monkeypatch.setattr(emb, "_model", None)
    monkeypatch.setenv("OLIVE_MCP_SEMANTIC_BUDGET_MS", "50")

    def slow_scores(entries, error_only):
        """Generate zero-valued score vectors after a deliberate delay.

        Parameters:
                entries: Entries to return and score.
                error_only: Whether to restrict scoring to error-related entries.

        Returns:
                A tuple containing the entries and an array of zero-valued score vectors."""
        time.sleep(0.4)
        n = len(list(entries))
        return list(entries), np.zeros((n, 384), dtype=np.float32)

    monkeypatch.setattr(ts, "_semantic_scores_for_entries", slow_scores)

    result = call_tool(
        "troubleshoot_olive_error",
        {
            "error_message": "CUDA out of memory during quantization",
            "mode": "auto",
        },
    )
    assert result["retrieval"]["degraded"] is True
    assert result["retrieval"]["reason"] == "semantic_budget_exceeded"
    assert result["retrieval"]["effective"] == "keyword"
    # Keyword path should still diagnose OOM (may match oom-quantization)
    assert isinstance(result.get("title"), str)
