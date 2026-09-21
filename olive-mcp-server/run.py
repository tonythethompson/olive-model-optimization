#!/usr/bin/env python3
"""Deterministic cross-platform launcher for the Olive MCP server.

Prefers the project virtual environment (olive-mcp-server/.venv, then repo
.venv), then falls back to the interpreter that invoked this script.

Use this entry from both ``.mcp.json`` and mcporter project config so agents
and Studio share one launch path.

Environment:
  OLIVE_MCP_REQUIRE_VENV=1  — exit non-zero if no project venv is found
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def find_venv_python(script_dir: Path, project_root: Path) -> Path | None:
    """
    Find the first existing virtual-environment Python executable.

    Parameters:
        script_dir (Path): Directory containing the launcher.
        project_root (Path): Repository root containing shared virtual environments.

    Returns:
        Path | None: The first matching executable path, or `None` when no supported virtual environment exists.
    """
    candidates = [
        # olive-mcp-server/.venv (created by the documented setup flow)
        script_dir / ".venv" / "Scripts" / "python.exe",
        script_dir / ".venv" / "bin" / "python",
        # Repository-root .venv / venv (legacy / monorepo-wide env)
        project_root / ".venv" / "Scripts" / "python.exe",
        project_root / ".venv" / "bin" / "python",
        project_root / "venv" / "Scripts" / "python.exe",
        project_root / "venv" / "bin" / "python",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _truthy(name: str) -> bool:
    """
    Determine whether an environment variable represents an enabled setting.

    Parameters:
        name (str): Name of the environment variable.

    Returns:
        bool: `true` if the value is `1`, `true`, `yes`, or `on`, ignoring case and
        surrounding whitespace; `false` otherwise.
    """
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def main() -> int:
    """
    Launch the Olive MCP server with the selected project Python interpreter.

    Returns:
        int: The exit code of the Olive MCP server process, or `1` when a required
        project virtual environment is unavailable.
    """
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    venv_python = find_venv_python(script_dir, project_root)
    require_venv = _truthy("OLIVE_MCP_REQUIRE_VENV")

    if venv_python is None:
        msg = (
            "olive-mcp-server: no project .venv found under "
            f"{script_dir / '.venv'} or {project_root / '.venv'}. "
            "Create one with: python -m venv olive-mcp-server/.venv "
            '&& pip install -e "olive-mcp-server[dev]" "mcp<2"'
        )
        if require_venv:
            print(msg, file=sys.stderr)
            return 1
        print(f"warning: {msg}; using {sys.executable}", file=sys.stderr)
        python = sys.executable
    else:
        python = str(venv_python)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(script_dir) + os.pathsep + env.get("PYTHONPATH", "")

    return subprocess.run(
        [python, "-m", "olive_mcp_server"],
        env=env,
        cwd=str(project_root),
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
