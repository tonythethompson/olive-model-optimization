"""Fetchers for updating the Olive MCP knowledge base from external sources."""

from __future__ import annotations

import importlib
from typing import Any

_LAZY: dict[str, tuple[str, str]] = {
    "fetch_official_docs": (".official_docs_fetcher", "fetch_official_docs"),
    "fetch_github_issues": (".github_scraper", "fetch_github_issues"),
    "fetch_onnx_runtime_docs": (".onnx_runtime_fetcher", "fetch_onnx_runtime_docs"),
}


def __getattr__(name: str) -> Any:
    """Load and cache a lazily exported module attribute.

    Parameters:
        name (str): Name of the module attribute to load.

    Returns:
        Any: The requested exported attribute.

    Raises:
        AttributeError: If the requested name is not a lazily exported attribute.
    """
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr = target
    module = importlib.import_module(module_name, __name__)
    value = getattr(module, attr)
    globals()[name] = value
    return value


__all__ = ["fetch_github_issues", "fetch_official_docs", "fetch_onnx_runtime_docs"]
