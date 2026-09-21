"""Olive optimization MCP server entry point.

This server exposes tools that help AI agents query, configure, and
troubleshoot Microsoft Olive model optimization workflows.

Usage:
    python -m olive_mcp_server
    olive-mcp-server
"""

from __future__ import annotations

import importlib
import logging
import os
import sys
from collections.abc import Callable, Iterator
from typing import Any

logger = logging.getLogger(__name__)

# name → (module path, attribute). Lazy so HTTP diagnosis does not pull in
# optional deps like BeautifulSoup just to troubleshoot an Olive traceback.
_TOOL_IMPORTS: dict[str, tuple[str, str]] = {
    "get_olive_passes": ("olive_mcp_server.tools.pass_catalog", "get_olive_passes"),
    "get_pass_config_template": ("olive_mcp_server.tools.config_generator", "get_pass_config_template"),
    "get_quantization_strategy": ("olive_mcp_server.tools.strategy_advisor", "get_quantization_strategy"),
    "get_hardware_optimization_guide": (
        "olive_mcp_server.tools.hardware_guide",
        "get_hardware_optimization_guide",
    ),
    "get_pass_chain": ("olive_mcp_server.tools.pass_chain", "get_pass_chain"),
    "troubleshoot_olive_error": ("olive_mcp_server.tools.troubleshooting", "troubleshoot_olive_error"),
    "diagnose_error": ("olive_mcp_server.tools.troubleshooting", "diagnose_error"),
    "get_error_frequency_summary": (
        "olive_mcp_server.tools.troubleshooting",
        "get_error_frequency_summary",
    ),
    "get_model_compatibility": ("olive_mcp_server.tools.compatibility", "get_model_compatibility"),
    "get_cli_command": ("olive_mcp_server.tools.cli_helper", "get_cli_command"),
    "get_data_config_template": ("olive_mcp_server.tools.data_config", "get_data_config_template"),
    "search_olive_documentation": ("olive_mcp_server.tools.docs_search", "search_olive_documentation"),
    "get_integration_recipe": ("olive_mcp_server.tools.integration_recipes", "get_integration_recipe"),
    "get_pass_parameters": ("olive_mcp_server.tools.pass_parameters", "get_pass_parameters"),
    "evaluate_optimization_tradeoff": (
        "olive_mcp_server.tools.tradeoff",
        "evaluate_optimization_tradeoff",
    ),
    "get_context_for_pipeline": (
        "olive_mcp_server.tools.passive_context",
        "get_context_for_pipeline",
    ),
    "validate_ui_state_recipe": (
        "olive_mcp_server.tools.studio_recipe",
        "validate_ui_state_recipe",
    ),
    "get_recipe_for_ui_state": (
        "olive_mcp_server.tools.studio_recipe",
        "get_recipe_for_ui_state",
    ),
    "get_runtime_ep_hints": (
        "olive_mcp_server.tools.runtime_ep_hints",
        "get_runtime_ep_hints",
    ),
    "record_troubleshoot_feedback": (
        "olive_mcp_server.tools.feedback",
        "record_troubleshoot_feedback",
    ),
    "get_mcp_capabilities": (
        "olive_mcp_server.tools.capabilities",
        "get_mcp_capabilities",
    ),
    "list_optimization_jobs": (
        "olive_mcp_server.tools.studio_jobs",
        "list_optimization_jobs",
    ),
    "get_optimization_job": (
        "olive_mcp_server.tools.studio_jobs",
        "get_optimization_job",
    ),
    "get_optimization_results": (
        "olive_mcp_server.tools.studio_jobs",
        "get_optimization_results",
    ),
    "validate_optimization_job": (
        "olive_mcp_server.tools.studio_jobs",
        "validate_optimization_job",
    ),
    "submit_optimization_job": (
        "olive_mcp_server.tools.studio_jobs",
        "submit_optimization_job",
    ),
    "cancel_optimization_job": (
        "olive_mcp_server.tools.studio_jobs",
        "cancel_optimization_job",
    ),
    # Phase 3: Agent autonomous loop
    "plan_optimization": (
        "olive_mcp_server.tools.agent_planner",
        "plan_optimization",
    ),
    "execute_and_observe": (
        "olive_mcp_server.tools.agent_execute",
        "execute_and_observe",
    ),
    "diagnose_and_fix": (
        "olive_mcp_server.tools.agent_diagnosis",
        "diagnose_and_fix",
    ),
    "compare_results": (
        "olive_mcp_server.tools.agent_compare",
        "compare_results",
    ),
    "get_model_info": (
        "olive_mcp_server.tools.agent_model_info",
        "get_model_info",
    ),
}
# Studio's HTTP POST /api/mcp/tool proxies these tools but is loopback-only
# (mcpToolLocalOnly). That gate is required for write tools like feedback.

_mcp_instance: Any | None = None
_resolved_tools: dict[str, Any] = {}


def _resolve_tool(name: str) -> Callable[..., Any] | None:
    """Resolve and cache a registered tool by name.

    Parameters:
        name (str): Name of the tool to resolve.

    Returns:
        Callable or None: The resolved tool, or `None` if the name is not registered.
    """
    if name in _resolved_tools:
        return _resolved_tools[name]
    target = _TOOL_IMPORTS.get(name)
    if target is None:
        return None
    module_name, attr = target
    try:
        module = importlib.import_module(module_name)
        fn = getattr(module, attr)
    except (ImportError, ModuleNotFoundError, AttributeError) as exc:
        # Optional/unimplemented tools must not prevent the server from starting.
        # ImportError covers failures raised by imports inside an optional tool
        # module (e.g. a missing optional dependency such as bs4/requests).
        logger.warning(
            "Failed to resolve MCP tool %s from %s.%s: %s",
            name,
            module_name,
            attr,
            exc,
            exc_info=True,
        )
        return None
    _resolved_tools[name] = fn
    return fn


def call_tool(name: str, args: dict | None = None) -> Any:
    """
    Invoke a registered tool with the supplied arguments.

    Parameters:
        name (str): Name of the registered tool.
        args (dict | None): Keyword arguments to pass to the tool.

    Returns:
        The tool's result, or an error dictionary when the tool is unknown.
    """
    payload = args if isinstance(args, dict) else {}
    tool = _resolve_tool(name)
    if tool is None:
        return {"error": f"Unknown tool: {name}"}
    try:
        return tool(**payload)
    except TypeError as exc:
        return {"error": f"Invalid arguments for {name}: {exc}"}


def _iter_tools() -> Iterator[Callable[..., Any]]:
    """Lazily resolve and yield all registered tool functions."""
    for name in _TOOL_IMPORTS:
        fn = _resolve_tool(name)
        if fn is not None:
            yield fn


def _build_mcp() -> Any:
    """Create the FastMCP server (requires the optional ``mcp`` package)."""
    from mcp.server.fastmcp import FastMCP

    instance = FastMCP("olive-mcp-server")
    for tool in _iter_tools():
        instance.tool()(tool)
    return instance


def __getattr__(name: str) -> Any:
    """
    Lazily provides the module's `mcp` and `TOOLS` attributes.

    Parameters:
        name (str): Attribute name to resolve.

    Returns:
        The MCP server instance for `mcp`, or the resolved tool list for `TOOLS`.

    Raises:
        AttributeError: If `name` is not a supported module attribute.
    """
    global _mcp_instance
    if name == "mcp":
        if _mcp_instance is None:
            _mcp_instance = _build_mcp()
        return _mcp_instance
    if name == "TOOLS":
        return list(_iter_tools())
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def main() -> None:
    """Run the MCP server with the configured transport.

    Transport selection:
      - CLI flag: ``--sse`` or ``--stdio``
      - Environment: ``MCP_TRANSPORT=sse|stdio`` (default: stdio)

    When ``OLIVE_MCP_PRELOAD_EMBEDDINGS=1``, warm the embedding model and
    shipped/runtime KB indexes before accepting traffic.
    """
    try:
        from olive_mcp_server.tools.preload import maybe_preload_embeddings

        maybe_preload_embeddings()
    except Exception:
        # Preload failures must not block the server; tools still lazy-load.
        import logging

        logging.getLogger(__name__).warning("Embedding preload failed", exc_info=True)

    mcp = _build_mcp()
    transport = os.environ.get("MCP_TRANSPORT", "stdio").lower()
    if "--sse" in sys.argv:
        transport = "sse"
    elif "--stdio" in sys.argv:
        transport = "stdio"

    if transport == "sse":
        # Prefer settings API: mcp.run() kwargs for host/port vary by mcp version.
        mcp.settings.host = os.environ.get("MCP_HOST", "127.0.0.1")
        mcp.settings.port = int(os.environ.get("MCP_PORT", "8000"))
        mcp.run(transport="sse")
    else:
        mcp.run()


if __name__ == "__main__":
    main()
