"""Integration tests: call tools through the MCP server object."""

import asyncio
import json

from olive_mcp_server.mcp_server import mcp


def _run(coro):
    return asyncio.run(coro)


def test_server_lists_all_tools():
    from olive_mcp_server.mcp_server import _TOOL_IMPORTS

    tools = _run(mcp.list_tools())
    names = {t.name for t in tools}
    assert set(_TOOL_IMPORTS.keys()) <= names
    required = {"get_olive_passes", "search_olive_documentation", "troubleshoot_olive_error"}
    assert required <= names
    assert "get_context_for_pipeline" in names
    assert "diagnose_error" in names
    # Studio UIState recipe bridge tools (HTTP to local Studio; no Olive)
    assert "validate_ui_state_recipe" in names
    assert "get_recipe_for_ui_state" in names
    assert "get_runtime_ep_hints" in names
    assert "get_mcp_capabilities" in names
    assert "list_optimization_jobs" in names
    assert "get_optimization_job" in names
    assert "get_optimization_results" in names
    assert "validate_optimization_job" in names
    assert "submit_optimization_job" in names
    assert "cancel_optimization_job" in names


def test_get_olive_passes_via_server():
    result, _ = _run(mcp.call_tool("get_olive_passes", {"filter": "quantization"}))
    data = json.loads(result[0].text)
    assert data["filter"] == "quantization"
    assert all(p["type"] == "quantization" for p in data["passes"])


def test_quantization_strategy_via_server():
    result, _ = _run(
        mcp.call_tool(
            "get_quantization_strategy",
            {
                "model_type": "LLM",
                "target_hardware": "NVIDIA RTX 4090",
                "latency_budget": "<100ms",
                "accuracy_threshold": "<2% drop",
            },
        )
    )
    data = json.loads(result[0].text)
    assert data["target_hardware"] == "nvidia"
    assert "pass_chain" in data


def test_pass_chain_via_server():
    result, _ = _run(
        mcp.call_tool(
            "get_pass_chain",
            {"pass_names": ["OnnxConversion", "OnnxModelOptimizer", "OnnxQuantization"]},
        )
    )
    data = json.loads(result[0].text)
    assert data["valid"] is True
    assert len(data["chain"]) == 3


def test_troubleshoot_olive_error_via_server():
    # Test matched entry
    result, _ = _run(
        mcp.call_tool(
            "troubleshoot_olive_error",
            {
                "error_message": (
                    "ValueError: The model file size is larger than 2GB. Please use use_external_data_format=True"
                ),
                "pass_name": "OnnxConversion",
            },
        )
    )
    data = json.loads(result[0].text)
    assert data["matched_entry"] == "onnx-export-external-data"
    assert "use_external_data_format" in data["workaround"]

    # Test unmatched entry
    result_unmatched, _ = _run(
        mcp.call_tool(
            "troubleshoot_olive_error",
            {
                "error_message": "Some random unique unknown failure message 12345",
            },
        )
    )
    data_unmatched = json.loads(result_unmatched[0].text)
    assert data_unmatched["matched_entry"] is None
    assert data_unmatched["title"] == "No exact match found"
