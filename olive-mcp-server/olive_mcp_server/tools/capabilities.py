"""Tool: get_mcp_capabilities.

Reports server capability state for agents (not transport/process health).
Process/transport health belongs to the client (mcporter list --status, IDE).
"""

from __future__ import annotations

import os
from typing import Any
from urllib.parse import urlparse

from olive_mcp_server import __version__
from olive_mcp_server.tools import KB_DIR
from olive_mcp_server.tools.embeddings import MODEL_NAME, is_model_loaded
from olive_mcp_server.tools.index_store import shipped_index_status
from olive_mcp_server.tools.retrieval import get_retrieval_mode, get_semantic_budget_ms
from olive_mcp_server.tools.studio_loopback import _is_loopback_host

# Versioned agent contract; bump when required tools/schemas change intentionally.
TOOLSET_VERSION = "2026.08.0-phase3"


def _kb_version() -> str:
    """
    Report a version marker for the knowledge-base JSON files.

    Returns:
        str: A modification-time marker, ``"empty"`` when no JSON files exist, or
                ``"unknown"`` when the directory cannot be accessed.
    """
    mtimes: list[float] = []
    try:
        # Sorted for stable iteration (mtime aggregate is order-independent,
        # but keep path walks consistent with docs_search indexing).
        for path in sorted(KB_DIR.glob("*.json"), key=lambda p: p.name):
            try:
                mtimes.append(path.stat().st_mtime)
            except OSError:
                continue
    except OSError:
        return "unknown"
    if not mtimes:
        return "empty"
    return f"mtime-{int(max(mtimes))}"


def _semantic_available() -> bool:
    """Cheap check — do not import sentence_transformers (slow cold start)."""
    try:
        import importlib.util

        return importlib.util.find_spec("sentence_transformers") is not None
    except Exception:
        return False


def _studio_url_status() -> tuple[bool, str | None]:
    """
    Validate the configured Studio API URL for supported schemes, loopback hosts, and absent credentials.

    Returns:
        tuple[bool, str | None]: A validity flag and an explanatory reason when the URL is invalid.
    """
    raw = (os.environ.get("OLIVE_STUDIO_API_URL") or "").strip()
    if not raw:
        return False, "OLIVE_STUDIO_API_URL is not set"
    try:
        parsed = urlparse(raw)
    except Exception:
        return False, "OLIVE_STUDIO_API_URL is not a valid URL"
    if parsed.scheme not in ("http", "https"):
        return False, "OLIVE_STUDIO_API_URL must use http or https"
    if not _is_loopback_host(parsed.hostname):
        return False, "OLIVE_STUDIO_API_URL host must be loopback"
    if parsed.username or parsed.password:
        return False, "OLIVE_STUDIO_API_URL must not include credentials"
    return True, None


def get_mcp_capabilities(probe_studio: bool = False) -> dict[str, Any]:
    """Report Olive MCP capability state for agent routing.

    Args:
        probe_studio: When True and Studio URL is configured, attempt a short
            reachability check and load agent-access policy. Default False
            means no network I/O (config flags only; job_control uses
            conservative defaults without live policy).

    Returns:
        Capability object: versions, semantic state, retrieval defaults,
        studio config flags, and job_control.
    """
    mode = get_retrieval_mode()
    budget_ms = get_semantic_budget_ms()
    sem_available = _semantic_available()
    sem_ready = is_model_loaded()
    studio_ok, studio_reason = _studio_url_status()

    # Network only when explicitly requested — never on the default path.
    studio_reachable: bool | None = None
    policy: dict[str, Any] | None = None
    if probe_studio and studio_ok:
        studio_reachable = _probe_studio()
        if studio_reachable:
            policy = _fetch_studio_policy()
    elif not studio_ok:
        studio_reachable = False

    # Without a live policy (unprobed or fetch failed): advertise read defaults
    # and deny side-effect capabilities rather than inventing permissive submit.
    inspection = True if policy is None else bool(policy.get("allowJobInspection", True))
    submission = False if policy is None else bool(policy.get("allowJobSubmission", False))
    cancellation = False if policy is None else bool(policy.get("allowJobCancellation", False))
    mcp_access = True if policy is None else bool(policy.get("mcpAccess", True))

    # Stable job_control shape across all branches (same keys always).
    if not studio_ok:
        job_control = {
            "supported": True,
            "enabled": False,
            "ready": False,
            "reason": "studio_unavailable",
            "inspection": True,
            "validation": True,
            "submission": False,
            "cancellation": False,
            "policy": None,
        }
    elif policy is not None and not mcp_access:
        job_control = {
            "supported": True,
            "enabled": False,
            "ready": False,
            "reason": "mcp_access_disabled",
            "inspection": False,
            "validation": False,
            "submission": False,
            "cancellation": False,
            "policy": policy,
        }
    elif studio_reachable is False:
        job_control = {
            "supported": True,
            "enabled": True,
            "ready": False,
            "reason": "studio_unreachable",
            "inspection": inspection,
            "validation": inspection or submission,
            "submission": submission,
            "cancellation": cancellation,
            "policy": policy,
        }
    else:
        # studio_reachable is True (probed) or None (configured, unprobed).
        ready = bool(inspection or submission or cancellation) and studio_reachable is True
        job_control = {
            "supported": True,
            "enabled": True,
            "ready": ready,
            "reason": "ready" if studio_reachable is True else "studio_configured_unprobed",
            "inspection": inspection,
            "validation": inspection or submission,
            "submission": submission,
            "cancellation": cancellation,
            "policy": policy,
        }

    return {
        "server": {
            "name": "olive-mcp-server",
            "version": __version__,
        },
        "kb": {
            "version": _kb_version(),
            "index": shipped_index_status(),
        },
        "semantic": {
            "available": sem_available,
            "ready": sem_ready,
            "model": MODEL_NAME if sem_available else None,
        },
        "retrieval": {
            "default_mode": mode,
            "semantic_budget_ms": budget_ms,
        },
        "studio": {
            "configured": studio_ok,
            "reachable": studio_reachable,
            "reason": None if studio_ok else studio_reason,
        },
        "job_control": job_control,
        "toolset": {
            "version": TOOLSET_VERSION,
        },
    }


def _probe_studio(timeout_s: float = 2.0) -> bool:
    """
    Check whether the configured Studio endpoint is reachable.

    Parameters:
        timeout_s (float): Maximum time in seconds allowed for the health check.

    Returns:
        bool: `True` if Studio responds successfully, `False` if it is unavailable or unreachable.
    """
    try:
        from .studio_loopback import studio_request

        payload = studio_request("GET", "/api/health", timeout=timeout_s)
        # health may be plain {ok:true} or error shape
        return not (isinstance(payload, dict) and payload.get("error") == "studio_unavailable")
    except Exception:
        return False


def _fetch_studio_policy() -> dict[str, Any] | None:
    """
    Retrieve Studio's agent-access policy when available.

    Returns:
        dict[str, Any] | None: The agent-access policy, or `None` if Studio is unavailable or the response is invalid.
    """
    try:
        from .studio_loopback import studio_request

        payload = studio_request("GET", "/api/olive/agent-access", timeout=2.0)
        if not isinstance(payload, dict) or payload.get("error"):
            return None
        policy = payload.get("policy")
        return policy if isinstance(policy, dict) else None
    except Exception:
        return None
