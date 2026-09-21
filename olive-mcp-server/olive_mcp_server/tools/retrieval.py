"""Retrieval mode and semantic budget helpers (Phase 0).

Modes:
  - auto (default): semantic/hybrid when ready within budget; else keyword + degraded
  - keyword: never load embeddings
  - semantic: always attempt semantic (caller opts into wait)

Environment:
  OLIVE_MCP_RETRIEVAL_MODE=auto|keyword|semantic
  OLIVE_MCP_SEMANTIC_BUDGET_MS=8000  (auto mode cold-start budget; 0 = no limit)
"""

from __future__ import annotations

import concurrent.futures
import os
import threading
from collections.abc import Callable
from typing import Any, Literal, TypeVar

T = TypeVar("T")

VALID_MODES = frozenset({"auto", "keyword", "semantic"})
DEFAULT_MODE = "auto"
DEFAULT_SEMANTIC_BUDGET_MS = 8000

BudgetOutcome = Literal["ok", "timeout", "busy"]

# Single shared pool for budgeted semantic work so concurrent auto-mode
# timeouts cannot stack many embedding-model loads (one per abandoned ThreadPoolExecutor).
_SEMANTIC_BUDGET_POOL = concurrent.futures.ThreadPoolExecutor(
    max_workers=1,
    thread_name_prefix="olive-mcp-semantic-budget",
)

# Single-flight: at most one budgeted callable is tracked. Timed-out work may
# still be running; further callers get an immediate keyword-fallback signal
# instead of queueing another embedding-model load behind it.
_INFLIGHT_LOCK = threading.Lock()
_INFLIGHT_FUTURE: concurrent.futures.Future[Any] | None = None


def get_retrieval_mode(override: str | None = None) -> str:
    """
    Determine the effective retrieval mode from an override or environment setting.

    Parameters:
        override (str | None): Optional mode value that takes precedence over the environment setting.

    Returns:
        str: A supported retrieval mode, or the default mode when the selected value is empty or invalid.
    """
    if override is not None and str(override).strip():
        raw = str(override).strip().lower()
    else:
        raw = os.environ.get("OLIVE_MCP_RETRIEVAL_MODE", DEFAULT_MODE).strip().lower()
    if raw in VALID_MODES:
        return raw
    return DEFAULT_MODE


def get_semantic_budget_ms() -> int:
    """
    Determine the time budget for cold semantic retrieval work.

    Returns:
        int: A nonnegative budget in milliseconds; zero means unlimited. Invalid
        configuration values use the default budget.
    """
    raw = os.environ.get("OLIVE_MCP_SEMANTIC_BUDGET_MS", str(DEFAULT_SEMANTIC_BUDGET_MS))
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return DEFAULT_SEMANTIC_BUDGET_MS


def _clear_inflight_if_current(future: concurrent.futures.Future[Any]) -> None:
    """
    Clear the tracked in-flight future when it matches the specified future.

    Parameters:
        future (concurrent.futures.Future[Any]): Future to compare with the tracked in-flight operation.
    """
    global _INFLIGHT_FUTURE
    with _INFLIGHT_LOCK:
        if _INFLIGHT_FUTURE is future:
            _INFLIGHT_FUTURE = None


def run_with_budget(fn: Callable[[], T], budget_ms: int) -> tuple[T | None, BudgetOutcome]:
    """
    Run a callable synchronously or within a wall-clock time budget.

    Parameters:
        fn (Callable[[], T]): Callable to execute.
        budget_ms (int): Maximum execution time in milliseconds; nonpositive values
            allow unlimited execution.

    Returns:
        tuple[T | None, BudgetOutcome]: The callable result and an outcome:
        ``ok`` on success, ``timeout`` when this call's budget expires, or ``busy``
        when another budgeted callable is already in flight (single-flight).

    Exceptions:
        Exception: Propagates exceptions raised by the callable.
    """
    if budget_ms <= 0:
        return fn(), "ok"

    timeout_s = budget_ms / 1000.0
    global _INFLIGHT_FUTURE

    with _INFLIGHT_LOCK:
        if _INFLIGHT_FUTURE is not None and _INFLIGHT_FUTURE.done():
            _INFLIGHT_FUTURE = None
        if _INFLIGHT_FUTURE is not None and not _INFLIGHT_FUTURE.done():
            # Already running (often a prior timeout). Do not queue another load.
            return None, "busy"
        future = _SEMANTIC_BUDGET_POOL.submit(fn)
        _INFLIGHT_FUTURE = future

    try:
        result = future.result(timeout=timeout_s)
    except concurrent.futures.TimeoutError:
        future.add_done_callback(_clear_inflight_if_current)
        return None, "timeout"
    except Exception:
        _clear_inflight_if_current(future)
        raise
    else:
        _clear_inflight_if_current(future)
        return result, "ok"


def budget_degraded_reason(outcome: BudgetOutcome) -> str:
    """Map a non-ok budget outcome to a stable retrieval ``reason`` code."""
    if outcome == "busy":
        return "semantic_busy"
    return "semantic_budget_exceeded"


def retrieval_meta(
    *,
    mode: str,
    effective: str,
    degraded: bool = False,
    reason: str | None = None,
) -> dict[str, Any]:
    """Build a stable retrieval metadata object for tool responses."""
    out: dict[str, Any] = {
        "mode": mode,
        "effective": effective,
        "degraded": bool(degraded),
    }
    if reason:
        out["reason"] = reason
    return out


def merge_retrieval_meta(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Merge retrieval metadata from two search pools.

    Prefers the most informative effective mode and surfaces degradation from
    either pool. When degraded, keeps ``degraded=True`` and the existing reason
    selection, but still retains the strongest observed effective mode (e.g.
    ``hybrid`` if either pool reported hybrid results).
    """
    mode = a.get("mode") or b.get("mode") or get_retrieval_mode()
    degraded = bool(a.get("degraded") or b.get("degraded"))
    reasons = [r for r in (a.get("reason"), b.get("reason")) if r]
    modes = [a.get("effective"), b.get("effective")]
    if "hybrid" in modes:
        effective = "hybrid"
    elif "keyword" in modes:
        effective = "keyword"
    else:
        effective = next((m for m in modes if m and m != "none"), None) or "keyword"
    if degraded:
        return retrieval_meta(
            mode=str(mode),
            effective=str(effective),
            degraded=True,
            reason=str(reasons[0]) if reasons else "semantic_budget_exceeded",
        )
    return retrieval_meta(
        mode=str(mode),
        effective=str(effective),
        degraded=False,
        reason=None,
    )
