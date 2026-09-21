#!/usr/bin/env python3
"""Native MCP contract smoke (no mcporter).

Runs safe catalog tools + get_mcp_capabilities via call_tool.
Intended as a required PR gate path (Phase 0).

Usage (from repo root or olive-mcp-server):
  python olive-mcp-server/scripts/smoke_native.py
  olive-mcp-server/.venv/Scripts/python olive-mcp-server/scripts/smoke_native.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SERVER_DIR = SCRIPT_DIR.parent
REPO_ROOT = SERVER_DIR.parent


def _ensure_path() -> None:
    """Ensure the MCP server directory is available on the Python module search path."""
    if str(SERVER_DIR) not in sys.path:
        sys.path.insert(0, str(SERVER_DIR))


def main() -> int:
    """Run native MCP contract checks and return an exit status indicating whether they passed."""
    _ensure_path()
    from olive_mcp_server.mcp_server import _TOOL_IMPORTS, call_tool

    required = [
        "get_olive_passes",
        "get_pass_chain",
        "get_integration_recipe",
        "get_mcp_capabilities",
        "troubleshoot_olive_error",
    ]
    missing = [n for n in required if n not in _TOOL_IMPORTS]
    if missing:
        print(f"FAIL: tools not registered: {missing}", file=sys.stderr)
        return 1

    checks: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append((name, ok, detail))
        status = "ok" if ok else "FAIL"
        print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))

    t0 = time.perf_counter()
    caps = call_tool("get_mcp_capabilities", {})
    check(
        "get_mcp_capabilities",
        isinstance(caps, dict) and "server" in caps and "job_control" in caps,
        f"job_control.supported={caps.get('job_control', {}).get('supported') if isinstance(caps, dict) else '?'}",
    )
    if isinstance(caps, dict):
        jc = caps.get("job_control") or {}
        check(
            "job_control inspection only",
            jc.get("supported") is True and jc.get("inspection") is True and jc.get("submission") is False,
            str(jc.get("reason")),
        )

    t1 = time.perf_counter()
    passes = call_tool("get_olive_passes", {"filter": "quantization"})
    check(
        "get_olive_passes",
        isinstance(passes, dict) and isinstance(passes.get("passes"), list) and len(passes["passes"]) > 0,
        f"n={len(passes.get('passes', [])) if isinstance(passes, dict) else 0}",
    )
    t2 = time.perf_counter()

    chain = call_tool(
        "get_pass_chain",
        {"pass_names": ["OnnxConversion", "OnnxQuantization"], "source_format": "torch"},
    )
    check(
        "get_pass_chain",
        isinstance(chain, dict) and chain.get("valid") is True,
        f"valid={chain.get('valid') if isinstance(chain, dict) else None}",
    )

    recipes = call_tool("get_integration_recipe", {"model_type": "LLM"})
    check(
        "get_integration_recipe",
        isinstance(recipes, dict) and (recipes.get("count", 0) > 0 or "recipes" in recipes or "id" in recipes),
        f"keys={list(recipes)[:6] if isinstance(recipes, dict) else type(recipes)}",
    )

    # Keyword-only troubleshoot must complete quickly (no hang).
    t3 = time.perf_counter()
    diag = call_tool(
        "troubleshoot_olive_error",
        {
            "error_message": "CUDA out of memory during quantization",
            "pass_name": "OnnxQuantization",
            "mode": "keyword",
        },
    )
    t4 = time.perf_counter()
    check(
        "troubleshoot keyword",
        isinstance(diag, dict) and "title" in diag and (diag.get("retrieval") or {}).get("effective") == "keyword",
        f"entry={diag.get('matched_entry') if isinstance(diag, dict) else None} ms={int((t4 - t3) * 1000)}",
    )

    print(
        json.dumps(
            {
                "capabilities_ms": int((t1 - t0) * 1000),
                "catalog_ms": int((t2 - t1) * 1000),
                "troubleshoot_keyword_ms": int((t4 - t3) * 1000),
            }
        )
    )

    failed = [c for c in checks if not c[1]]
    if failed:
        print(f"FAIL: {len(failed)} check(s)", file=sys.stderr)
        return 1
    print("PASS: native MCP smoke")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
