"""Shared loopback HTTP client for Olive Studio bridge tools.

Base URL comes only from ``OLIVE_STUDIO_API_URL`` (loopback HTTP(S) only).
No redirects, no credentials, no Olive execution.
"""

from __future__ import annotations

import ipaddress
import json
import os
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

ENV_API_URL = "OLIVE_STUDIO_API_URL"
DEFAULT_TIMEOUT_SECONDS = 5.0

_LOOPBACK_HOSTNAMES = frozenset({"localhost", "127.0.0.1", "::1"})


class _NoRedirect(HTTPRedirectHandler):
    """Refuse redirects so a loopback URL cannot bounce off-host (SSRF)."""

    def redirect_request(self, req, fp, code, msg, headers, newurl) -> None:  # noqa: ANN001
        return None


_OPENER = build_opener(_NoRedirect)


def err(
    code: str,
    message: str,
    *,
    detail: str | None = None,
    side_effect: bool | None = None,
) -> dict[str, Any]:
    """Build a structured error dict.

    When ``side_effect`` is not None, the returned dict includes a
    ``side_effect`` key with the given boolean value. This lets callers
    express uncertain side-effect state (e.g. post-submission errors).
    """
    out: dict[str, Any] = {"error": code, "message": message}
    if detail:
        out["detail"] = detail
    if side_effect is not None:
        out["side_effect"] = side_effect
    return out


def studio_unavailable(message: str, *, detail: str | None = None) -> dict[str, Any]:
    """``studio_unavailable`` error shape shared by bridge tools."""
    return err("studio_unavailable", message, detail=detail)


def _normalize_host(host: str | None) -> str:
    if not host:
        return ""
    h = host.strip().lower()
    if h.startswith("[") and h.endswith("]"):
        h = h[1:-1]
    return h


def _is_loopback_host(host: str | None) -> bool:
    """Allow localhost, 127.0.0.1, ::1, and any IP with is_loopback=True."""
    h = _normalize_host(host)
    if not h:
        return False
    if h in _LOOPBACK_HOSTNAMES:
        return True
    try:
        return ipaddress.ip_address(h).is_loopback
    except ValueError:
        return False


def resolve_studio_base() -> tuple[str | None, dict[str, Any] | None]:
    """Read only ``OLIVE_STUDIO_API_URL``; never accept a caller-supplied URL.

    Returns:
        ``(base_url, None)`` on success, or ``(None, error_dict)`` when the
        env var is missing or fails loopback/scheme/credential checks.
    """
    raw = os.environ.get(ENV_API_URL, "").strip()
    if not raw:
        return None, studio_unavailable(
            f"{ENV_API_URL} is not set. Start Olive Studio and point "
            f"{ENV_API_URL} at its loopback base URL (e.g. http://127.0.0.1:3000)."
        )

    parsed = urlparse(raw)
    if parsed.scheme not in ("http", "https"):
        return None, studio_unavailable(
            f"{ENV_API_URL} must be an http(s) loopback URL.",
            detail=f"scheme={parsed.scheme!r}",
        )
    if parsed.username is not None or parsed.password is not None:
        return None, studio_unavailable(
            f"{ENV_API_URL} must not include credentials.",
        )
    if not _is_loopback_host(parsed.hostname):
        return None, studio_unavailable(
            f"{ENV_API_URL} must target a loopback host (127.0.0.1, localhost, or ::1).",
            detail=f"host={parsed.hostname!r}",
        )
    try:
        # Eagerly validate port; urlparse stores invalid ports and raises
        # ValueError only when .port is accessed (out-of-range / malformed).
        _ = parsed.port
    except ValueError as exc:
        return None, studio_unavailable(
            f"{ENV_API_URL} has an invalid or out-of-range port.",
            detail=str(exc),
        )
    # Base URL only: reject path/query/fragment so `{base}{path}` stays correct.
    if parsed.path not in ("", "/") or parsed.params or parsed.query or parsed.fragment:
        return None, studio_unavailable(
            f"{ENV_API_URL} must be a loopback base URL without path, query, or fragment (e.g. http://127.0.0.1:3000).",
            detail=(
                f"path={parsed.path!r} params={parsed.params!r} query={parsed.query!r} fragment={parsed.fragment!r}"
            ),
        )
    return f"{parsed.scheme}://{parsed.netloc}", None


def _parse_json_body(raw: bytes | str) -> Any | None:
    try:
        text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        return json.loads(text)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None


def _is_timeout_reason(reason: Any) -> bool:
    if isinstance(reason, TimeoutError):
        return True
    return "timed out" in str(reason).lower()


def studio_request(
    method: str,
    path: str,
    *,
    body: dict[str, Any] | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """
    Send a JSON request to a path under the validated Olive Studio base URL.

    Parameters:
        method (str): HTTP method to use.
        path (str): Studio path, with or without a leading slash.
        body (dict[str, Any] | None): Optional JSON object to include in the request.
        timeout (float): Request timeout in seconds.

    Returns:
        dict[str, Any]: Parsed JSON object on success, or a structured error dictionary.
    """
    base, resolve_err = resolve_studio_base()
    if resolve_err is not None:
        return resolve_err

    if not path.startswith("/"):
        path = "/" + path
    endpoint = f"{base}{path}"

    # Trusted by Studio routes as the MCP bridge context; request bodies are not
    # used to establish agent identity.
    headers = {"Accept": "application/json", "X-Olive-MCP-Agent": "1"}
    data: bytes | None = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = Request(
        endpoint,
        data=data,
        method=method.upper(),
        headers=headers,
    )

    try:
        # URL is restricted to validated loopback http(s) only (SSRF guard).
        # Redirects disabled so the request cannot leave loopback.
        with _OPENER.open(request, timeout=timeout) as response:  # noqa: S310
            status = getattr(response, "status", None) or response.getcode()
            raw = response.read()
    except HTTPError as exc:
        try:
            err_body = exc.read() or b""
        except Exception:  # noqa: BLE001 — best-effort body read
            err_body = b""
        parsed = _parse_json_body(err_body)
        # Only forward structured error payloads; never treat an HTTP >=400
        # success-shaped body as a successful bridge response.
        if isinstance(parsed, dict) and isinstance(parsed.get("error"), str) and parsed["error"]:
            return parsed
        return studio_unavailable(
            "Olive Studio bridge returned an HTTP error.",
            detail=f"status={exc.code}",
        )
    except URLError as exc:
        reason = getattr(exc, "reason", exc)
        if _is_timeout_reason(reason):
            return studio_unavailable(
                "Olive Studio bridge timed out.",
                detail=f"timeout_seconds={timeout}",
            )
        return studio_unavailable(
            "Olive Studio bridge is not reachable.",
            detail=str(reason),
        )
    except TimeoutError:
        return studio_unavailable(
            "Olive Studio bridge timed out.",
            detail=f"timeout_seconds={timeout}",
        )
    except OSError as exc:
        return studio_unavailable(
            "Olive Studio bridge request failed.",
            detail=str(exc),
        )

    if status is not None and int(status) >= 400:
        return studio_unavailable(
            "Olive Studio bridge returned an HTTP error.",
            detail=f"status={status}",
        )

    parsed = _parse_json_body(raw)
    if not isinstance(parsed, dict):
        return err(
            "invalid_bridge_response",
            "Olive Studio bridge returned a non-object JSON payload.",
        )
    return parsed


def _ensure_session(session_id: str | None) -> tuple[str | None, dict[str, Any]]:
    """Get or transparently create the Express-owned agent session.

    This is the auto-creation path for the first agent tool call: when no
    ``session_id`` is supplied, the Studio POST endpoint creates one. Session
    state therefore survives MCP stdio process restarts for as long as the
    Express process remains alive.
    """
    if session_id:
        session = studio_request(
            "GET",
            f"/api/olive/agent/sessions/{quote(session_id, safe='')}",
        )
        if isinstance(session.get("error"), str) and session["error"]:
            return None, session
        return session_id, session

    session = studio_request(
        "POST",
        "/api/olive/agent/sessions",
        body={},
    )
    if isinstance(session.get("error"), str) and session["error"]:
        return None, session
    created_id = session.get("sessionId")
    if not isinstance(created_id, str) or not created_id:
        return None, err(
            "invalid_bridge_response",
            "Olive Studio session response missing sessionId.",
        )
    return created_id, session


def _update_session(session_id: str, **kwargs: Any) -> dict[str, Any]:
    """Write metadata without recording a new optimization attempt."""
    return studio_request(
        "PUT",
        f"/api/olive/agent/sessions/{quote(session_id, safe='')}",
        body=kwargs,
    )


def _record_attempt(session_id: str, **kwargs: Any) -> dict[str, Any]:
    """Record one optimization attempt and its resulting context."""
    return studio_request(
        "PUT",
        f"/api/olive/agent/sessions/{quote(session_id, safe='')}",
        body={"attempt": True, **kwargs},
    )
