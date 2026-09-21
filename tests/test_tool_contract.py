"""Regression tests for the documented MCP tool contract."""

import asyncio
import re
from pathlib import Path

from olive_mcp_server.mcp_server import _build_mcp

README = Path(__file__).resolve().parents[1] / "README.md"
TOOL_ROW = re.compile(r"^\| `([a-z][a-z0-9_]*)`\s*\|", re.MULTILINE)


def _documented_tools() -> list[str]:
    """Read tool names from the README's public contract table."""
    tools_section = README.read_text(encoding="utf-8").split("## Tools", maxsplit=1)[1]
    tools_table = tools_section.split("### Agent clients", maxsplit=1)[0]
    return TOOL_ROW.findall(tools_table)


def test_documented_tools_match_fastmcp_registration():
    """Every advertised tool is registered, and every registered tool is advertised."""
    registered = [tool.name for tool in asyncio.run(_build_mcp().list_tools())]
    documented = _documented_tools()

    assert len(documented) == len(set(documented)), "README tool table contains duplicate names"
    assert set(documented) == set(registered), (
        f"README/FastMCP tool mismatch; undocumented={sorted(set(registered) - set(documented))}, "
        f"unregistered={sorted(set(documented) - set(registered))}"
    )
