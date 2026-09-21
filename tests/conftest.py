"""Shared pytest fixtures for olive-mcp-server tests."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator

import pytest

from olive_mcp_server.tools import retrieval

# Generous ceiling for abandoned budget workers (tests use ≤0.55s sleeps today).
_INFLIGHT_DRAIN_TIMEOUT_S = 5.0

# Many tests exercise "auto"-mode retrieval without mocking the semantic
# path; whichever one runs first in a given session may pay for the
# embedding model's genuine first-use load (network fetch + weights load).
# That's a real network/disk-bound cost, not a leak, so give it a much
# longer ceiling than the steady-state one above. This intentionally does
# NOT eagerly load the model at collection time (no autouse warm-up
# fixture) — default CI/offline runs never depend on network access unless
# a test itself exercises the semantic path.
_INFLIGHT_DRAIN_COLD_START_TIMEOUT_S = 60.0


def _model_is_warm() -> bool:
    try:
        from olive_mcp_server.tools.embeddings import is_model_loaded

        return is_model_loaded()
    except Exception:
        return False


def wait_for_inflight_semantic_clear(
    *,
    timeout_s: float | None = None,
) -> None:
    """
    Poll retrieval single-flight state until the tracked future is idle.

    Clears a done future so the next ``run_with_budget`` call is not stuck on
    a stale handle. Fails if work is still active after ``timeout_s``. When
    the embedding model hasn't been loaded yet, a much longer ceiling is used
    since the in-flight work may be a genuine first-use model load rather
    than an abandoned worker.
    """
    if timeout_s is None:
        timeout_s = _INFLIGHT_DRAIN_TIMEOUT_S if _model_is_warm() else _INFLIGHT_DRAIN_COLD_START_TIMEOUT_S
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        with retrieval._INFLIGHT_LOCK:
            fut = retrieval._INFLIGHT_FUTURE
            if fut is None or fut.done():
                retrieval._INFLIGHT_FUTURE = None
                return
        time.sleep(0.05)
    with retrieval._INFLIGHT_LOCK:
        fut = retrieval._INFLIGHT_FUTURE
    raise AssertionError(f"semantic budget worker still in flight after {timeout_s}s (future={fut!r})")


@pytest.fixture
def wait_inflight_semantic_clear() -> Callable[..., None]:
    """Expose the drain helper to tests that must wait mid-body."""
    return wait_for_inflight_semantic_clear


@pytest.fixture(autouse=True)
def _drain_semantic_inflight_between_tests() -> Iterator[None]:
    """Ensure no abandoned ``run_with_budget`` worker leaks across tests."""
    wait_for_inflight_semantic_clear()
    yield
    wait_for_inflight_semantic_clear()
