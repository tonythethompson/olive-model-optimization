"""Optional warm-path for long-lived MCP hosts (Phase 1).

Set ``OLIVE_MCP_PRELOAD_EMBEDDINGS=1`` to load the embedding model and
shipped/runtime KB indexes at process start (before accepting traffic).
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def _truthy(name: str) -> bool:
    """
    Determine whether an environment variable contains a truthy value.

    Parameters:
        name (str): Name of the environment variable to evaluate.

    Returns:
        bool: `True` if the value is `1`, `true`, `yes`, or `on`, ignoring surrounding
        whitespace and letter case; `False` otherwise.
    """
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def maybe_preload_embeddings() -> dict[str, bool]:
    """
    Preloads the embedding model and troubleshooting indexes when enabled by the environment.

    Returns:
        dict[str, bool]: A mapping with `requested` indicating whether preload was enabled and
        `done` indicating whether preload completed.
    """
    if not _truthy("OLIVE_MCP_PRELOAD_EMBEDDINGS"):
        return {"requested": False, "done": False}

    logger.info("Preloading embeddings and KB indexes (OLIVE_MCP_PRELOAD_EMBEDDINGS=1)")
    from olive_mcp_server.tools import load_studio_troubleshooting, load_troubleshooting
    from olive_mcp_server.tools.docs_search import get_or_build_kb_index
    from olive_mcp_server.tools.embeddings import encode_query
    from olive_mcp_server.tools.troubleshooting import get_troubleshooting_index

    # Touch model
    encode_query("preload")
    get_or_build_kb_index()
    get_troubleshooting_index(load_troubleshooting())
    get_troubleshooting_index(load_studio_troubleshooting())
    logger.info("Embedding preload complete")
    return {"requested": True, "done": True}
