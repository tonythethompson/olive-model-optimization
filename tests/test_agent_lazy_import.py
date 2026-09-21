"""Test lazy-import isolation for Phase 3 agent tools.

Validates: Requirements 11.2, 11.4
"""

from __future__ import annotations

import subprocess
import sys


def test_new_tool_modules_not_in_sys_modules_after_server_import():
    """Importing mcp_server must NOT eagerly load agent tool modules.

    After verifying no agent modules were eagerly loaded, the second phase
    accesses ``_resolve_tool`` and resolves ``execute_and_observe`` to confirm
    the lazy import path works and the module then appears in ``sys.modules``.
    """
    agent_modules = [
        "olive_mcp_server.tools.agent_execute",
        "olive_mcp_server.tools.agent_planner",
        "olive_mcp_server.tools.agent_diagnosis",
        "olive_mcp_server.tools.agent_compare",
        "olive_mcp_server.tools.agent_model_info",
    ]

    script = f"""
import sys
import olive_mcp_server.mcp_server  # noqa: F401

agent_modules = {agent_modules!r}
eagerly_loaded = [name for name in agent_modules if name in sys.modules]
if eagerly_loaded:
    raise SystemExit(f"agent modules imported eagerly: {{eagerly_loaded}}")

# Phase 2: resolve a tool via _resolve_tool and confirm the module loads.
from olive_mcp_server.mcp_server import _resolve_tool
tool = _resolve_tool("execute_and_observe")
if tool is None:
    raise SystemExit("execute_and_observe resolved to None")
if "olive_mcp_server.tools.agent_execute" not in sys.modules:
    raise SystemExit("agent_execute not in sys.modules after resolve")
print("OK")
"""

    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
