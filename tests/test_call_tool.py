import pytest

from olive_mcp_server.mcp_server import _TOOL_IMPORTS, _resolve_tool, _resolved_tools, call_tool
from olive_mcp_server.tools.troubleshooting import reset_frequency_store


@pytest.fixture(autouse=True)
def _clean_frequency_store():
    reset_frequency_store()
    yield
    reset_frequency_store()


def test_call_tool_dispatches_troubleshoot():
    # Force keyword mode: this test exercises tool dispatch, not semantic
    # retrieval, and must not trigger a real embedding-model load in the
    # shared single-flight worker (which can outlive this test's process on
    # slow/offline CI runners and starve later tests' inflight-drain fixture).
    result = call_tool(
        "troubleshoot_olive_error",
        {"error_message": "unexpected keyword argument hf_config", "mode": "keyword"},
    )
    assert isinstance(result, dict)
    assert "title" in result
    assert "root_cause" in result
    assert "workaround" in result


def test_call_tool_unknown_returns_error():
    result = call_tool("not_a_real_tool", {})
    assert result.get("error")
    assert "Unknown tool" in result["error"]


def test_call_tool_names_match_registered_tools():
    assert "troubleshoot_olive_error" in _TOOL_IMPORTS


def test_resolve_tool_logs_import_failure(monkeypatch, caplog):
    monkeypatch.setitem(
        _TOOL_IMPORTS,
        "broken_test_tool",
        ("olive_mcp_server.tools.module_that_does_not_exist", "broken_test_tool"),
    )
    _resolved_tools.pop("broken_test_tool", None)

    with caplog.at_level("WARNING", logger="olive_mcp_server.mcp_server"):
        assert _resolve_tool("broken_test_tool") is None

    assert "Failed to resolve MCP tool broken_test_tool" in caplog.text
