#!/usr/bin/env python3
"""Split cold/warm tool-execution benchmarks (in-process; not mcporter E2E).

Prints JSON lines for CI artifacts. Does not require Studio.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parent.parent
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))


def _ms(t0: float, t1: float) -> int:
    """Convert an elapsed time interval from seconds to integer milliseconds.

    Parameters:
        t0 (float): Start time in seconds.
        t1 (float): End time in seconds.

    Returns:
        int: The elapsed interval in milliseconds, truncated to an integer.
    """
    return int((t1 - t0) * 1000)


def main() -> int:
    """
    Run cold and warm in-process benchmarks for Olive MCP tools and print the results as formatted JSON.

    Returns:
        int: Zero after the benchmark results are printed.
    """
    from olive_mcp_server.mcp_server import call_tool
    from olive_mcp_server.tools import embeddings as emb

    # Ensure cold embedding path for first semantic attempt (optional).
    emb._model = None  # type: ignore[attr-defined]

    rows: list[dict] = []

    t0 = time.perf_counter()
    call_tool("get_olive_passes", {"filter": "quantization"})
    t1 = time.perf_counter()
    call_tool("get_olive_passes", {"filter": "quantization"})
    t2 = time.perf_counter()
    rows.append(
        {
            "tool": "get_olive_passes",
            "cold_ms": _ms(t0, t1),
            "warm_ms": _ms(t1, t2),
            "scope": "tool_execution",
        }
    )

    t0 = time.perf_counter()
    call_tool(
        "troubleshoot_olive_error",
        {"error_message": "CUDA out of memory", "mode": "keyword"},
    )
    t1 = time.perf_counter()
    call_tool(
        "troubleshoot_olive_error",
        {"error_message": "CUDA out of memory", "mode": "keyword"},
    )
    t2 = time.perf_counter()
    rows.append(
        {
            "tool": "troubleshoot_olive_error",
            "mode": "keyword",
            "cold_ms": _ms(t0, t1),
            "warm_ms": _ms(t1, t2),
            "scope": "tool_execution",
        }
    )

    # auto with tiny budget forces degraded path if model not loaded
    import os

    os.environ["OLIVE_MCP_SEMANTIC_BUDGET_MS"] = "50"
    emb._model = None  # type: ignore[attr-defined]
    t0 = time.perf_counter()
    r = call_tool(
        "troubleshoot_olive_error",
        {"error_message": "CUDA out of memory during quantization", "mode": "auto"},
    )
    t1 = time.perf_counter()
    rows.append(
        {
            "tool": "troubleshoot_olive_error",
            "mode": "auto",
            "budget_ms": 50,
            "cold_ms": _ms(t0, t1),
            "degraded": bool((r or {}).get("retrieval", {}).get("degraded")) if isinstance(r, dict) else None,
            "effective": (r or {}).get("retrieval", {}).get("effective") if isinstance(r, dict) else None,
            "scope": "tool_execution",
        }
    )

    print(json.dumps({"benchmarks": rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
