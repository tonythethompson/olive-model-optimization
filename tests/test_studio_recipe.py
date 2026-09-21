"""Unit tests for studio_recipe bridge tools (no live Studio, no Olive)."""

from __future__ import annotations

import io
import json
from email.message import Message
from typing import Any
from unittest.mock import MagicMock
from urllib.error import HTTPError, URLError

import pytest

from olive_mcp_server.tools import studio_loopback, studio_recipe
from olive_mcp_server.tools.studio_recipe import (
    BRIDGE_PATH,
    ENV_API_URL,
    get_recipe_for_ui_state,
    validate_ui_state_recipe,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

_SAMPLE_UI_STATE: dict[str, Any] = {
    "modelPath": "/models/phi.onnx",
    "targetDevice": "GPU",
    "passes": [{"type": "OnnxConversion"}],
}

_SUCCESS_PAYLOAD: dict[str, Any] = {
    "effectiveState": {
        "modelPath": "/models/phi.onnx",
        "targetDevice": "GPU",
        "passes": [{"type": "OnnxConversion"}],
    },
    "recipe": {
        "input_model": {"type": "ONNXModel", "config": {"model_path": "/models/phi.onnx"}},
        "passes": {"conversion": {"type": "OnnxConversion"}},
    },
    "isRunnable": True,
    "schemaErrors": [],
    "pipelineIssues": [{"severity": "info", "message": "ok"}],
    "localExecutionIssues": [],
    "advisories": ["prefer external data for large models"],
    "pipelineCriticalCount": 0,
    "conversionWarnings": ["unused pass option ignored"],
}


@pytest.fixture(autouse=True)
def _clear_studio_url(monkeypatch: pytest.MonkeyPatch):
    """Isolate every test from the host environment's Studio URL."""
    monkeypatch.delenv(ENV_API_URL, raising=False)


def _set_loopback_url(monkeypatch: pytest.MonkeyPatch, base: str = "http://127.0.0.1:3000") -> None:
    monkeypatch.setenv(ENV_API_URL, base)


def _mock_response(payload: dict[str, Any] | bytes | str, *, status: int = 200) -> MagicMock:
    """Build a context-manager response like urllib's urlopen result."""
    if isinstance(payload, dict):
        raw = json.dumps(payload).encode("utf-8")
    elif isinstance(payload, str):
        raw = payload.encode("utf-8")
    else:
        raw = payload

    resp = MagicMock()
    resp.status = status
    resp.getcode.return_value = status
    resp.read.return_value = raw
    resp.__enter__.return_value = resp
    resp.__exit__.return_value = False
    return resp


def _patch_opener(monkeypatch: pytest.MonkeyPatch, side_effect=None, return_value=None):
    """Replace studio_loopback._OPENER.open at the HTTP boundary."""
    opener = MagicMock()
    if side_effect is not None:
        opener.open.side_effect = side_effect
    else:
        opener.open.return_value = return_value
    monkeypatch.setattr(studio_loopback, "_OPENER", opener)
    return opener


# ---------------------------------------------------------------------------
# Endpoint resolution / SSRF guards
# ---------------------------------------------------------------------------


def test_missing_api_url_returns_studio_unavailable():
    """Objective: unset OLIVE_STUDIO_API_URL → structured studio_unavailable."""
    # Arrange — env cleared by autouse fixture

    # Act
    result = validate_ui_state_recipe(_SAMPLE_UI_STATE)

    # Assert
    assert result["error"] == "studio_unavailable"
    assert ENV_API_URL in result["message"]
    assert "detail" not in result or result.get("detail") is None


def test_non_loopback_host_rejected(monkeypatch: pytest.MonkeyPatch):
    """Objective: non-loopback hosts are refused (SSRF guard)."""
    # Arrange
    _set_loopback_url(monkeypatch, "http://example.com:3000")
    opener = _patch_opener(monkeypatch, return_value=_mock_response(_SUCCESS_PAYLOAD))

    # Act
    result = validate_ui_state_recipe(_SAMPLE_UI_STATE)

    # Assert
    assert result["error"] == "studio_unavailable"
    assert "loopback" in result["message"].lower()
    opener.open.assert_not_called()


def test_invalid_scheme_rejected(monkeypatch: pytest.MonkeyPatch):
    """Objective: only http(s) schemes are accepted."""
    # Arrange
    _set_loopback_url(monkeypatch, "ftp://127.0.0.1:3000")
    opener = _patch_opener(monkeypatch, return_value=_mock_response(_SUCCESS_PAYLOAD))

    # Act
    result = get_recipe_for_ui_state(_SAMPLE_UI_STATE)

    # Assert
    assert result["error"] == "studio_unavailable"
    assert "http" in result["message"].lower()
    opener.open.assert_not_called()


def test_credentials_in_url_rejected(monkeypatch: pytest.MonkeyPatch):
    """Objective: URLs with userinfo are rejected."""
    # Arrange
    _set_loopback_url(monkeypatch, "http://user:pass@127.0.0.1:3000")
    opener = _patch_opener(monkeypatch, return_value=_mock_response(_SUCCESS_PAYLOAD))

    # Act
    result = validate_ui_state_recipe(_SAMPLE_UI_STATE)

    # Assert
    assert result["error"] == "studio_unavailable"
    assert "credential" in result["message"].lower()
    opener.open.assert_not_called()


@pytest.mark.parametrize(
    "bad_url",
    [
        "http://127.0.0.1:99999",
        "http://localhost:70000",
        "http://127.0.0.1:notaport",
    ],
)
def test_invalid_or_out_of_range_port_rejected(monkeypatch: pytest.MonkeyPatch, bad_url: str):
    """Objective: malformed / out-of-range ports → studio_unavailable (not request-time crash)."""
    _set_loopback_url(monkeypatch, bad_url)
    opener = _patch_opener(monkeypatch, return_value=_mock_response(_SUCCESS_PAYLOAD))

    result = validate_ui_state_recipe(_SAMPLE_UI_STATE)

    assert result["error"] == "studio_unavailable"
    assert "port" in result["message"].lower()
    opener.open.assert_not_called()


def test_resolve_studio_base_rejects_out_of_range_port(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(ENV_API_URL, "http://127.0.0.1:99999")
    base, err = studio_loopback.resolve_studio_base()
    assert base is None
    assert err is not None
    assert err["error"] == "studio_unavailable"
    assert "port" in err["message"].lower()


@pytest.mark.parametrize(
    "bad_url",
    [
        "http://127.0.0.1:3000/foo",
        "http://127.0.0.1:3000/api",
        "http://localhost:3000/?x=1",
        "http://127.0.0.1:3000/#frag",
        "http://127.0.0.1:3000/path?q=1#f",
    ],
)
def test_resolve_studio_base_rejects_path_query_fragment(monkeypatch: pytest.MonkeyPatch, bad_url: str):
    """Objective: OLIVE_STUDIO_API_URL must be a bare base URL."""
    monkeypatch.setenv(ENV_API_URL, bad_url)
    base, err = studio_loopback.resolve_studio_base()
    assert base is None
    assert err is not None
    assert err["error"] == "studio_unavailable"
    assert "base url" in err["message"].lower()


@pytest.mark.parametrize(
    "ok_url",
    [
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3000/",
        "http://localhost:3000",
        "https://127.0.0.1:3443",
    ],
)
def test_resolve_studio_base_accepts_bare_base_url(monkeypatch: pytest.MonkeyPatch, ok_url: str):
    monkeypatch.setenv(ENV_API_URL, ok_url)
    base, err = studio_loopback.resolve_studio_base()
    assert err is None
    assert base is not None
    assert base.endswith(("3000", "3443"))
    assert "/foo" not in base
    assert "?" not in base
    assert "#" not in base


@pytest.mark.parametrize(
    "base",
    [
        "http://127.0.0.1:3000",
        "http://localhost:3000",
        "http://[::1]:3000",
        "https://127.0.0.1:3443",
    ],
)
def test_loopback_hosts_accepted(monkeypatch: pytest.MonkeyPatch, base: str):
    """Objective: localhost / 127.0.0.1 / ::1 (http and https) reach the opener."""
    # Arrange
    _set_loopback_url(monkeypatch, base)
    opener = _patch_opener(monkeypatch, return_value=_mock_response(_SUCCESS_PAYLOAD))

    # Act
    result = validate_ui_state_recipe(_SAMPLE_UI_STATE)

    # Assert
    assert "error" not in result
    assert result["is_runnable"] is True
    opener.open.assert_called_once()
    request = opener.open.call_args.args[0]
    assert request.full_url.rstrip("/").endswith(BRIDGE_PATH) or BRIDGE_PATH in request.full_url


# ---------------------------------------------------------------------------
# Successful bridge forwarding
# ---------------------------------------------------------------------------


def test_validate_ui_state_recipe_success_forwards_and_projects(
    monkeypatch: pytest.MonkeyPatch,
):
    """Objective: happy-path validation view from camelCase bridge payload."""
    # Arrange
    _set_loopback_url(monkeypatch)
    opener = _patch_opener(monkeypatch, return_value=_mock_response(_SUCCESS_PAYLOAD))

    # Act
    result = validate_ui_state_recipe(_SAMPLE_UI_STATE)

    # Assert — request shape
    opener.open.assert_called_once()
    request = opener.open.call_args.args[0]
    assert request.get_method() == "POST"
    assert request.full_url == f"http://127.0.0.1:3000{BRIDGE_PATH}"
    content_type = request.get_header("Content-type") or request.get_header("Content-Type")
    assert content_type == "application/json"
    body = json.loads(request.data.decode("utf-8"))
    assert body == {"uiState": _SAMPLE_UI_STATE}
    timeout = opener.open.call_args.kwargs.get("timeout")
    assert timeout == studio_recipe.DEFAULT_TIMEOUT_SECONDS

    # Assert — validation projection (no full recipe)
    assert result["is_runnable"] is True
    assert result["effective_state"] == _SUCCESS_PAYLOAD["effectiveState"]
    assert result["schema_errors"] == []
    assert result["pipeline_issues"] == _SUCCESS_PAYLOAD["pipelineIssues"]
    assert result["local_execution_issues"] == []
    assert result["advisories"] == _SUCCESS_PAYLOAD["advisories"]
    assert result["pipeline_critical_count"] == 0
    assert "olive_recipe" not in result
    assert "conversion_warnings" not in result
    assert "error" not in result


def test_get_recipe_for_ui_state_success_includes_recipe(
    monkeypatch: pytest.MonkeyPatch,
):
    """Objective: recipe view adds olive_recipe + conversion_warnings."""
    # Arrange
    _set_loopback_url(monkeypatch)
    _patch_opener(monkeypatch, return_value=_mock_response(_SUCCESS_PAYLOAD))

    # Act
    result = get_recipe_for_ui_state(_SAMPLE_UI_STATE)

    # Assert
    assert result["is_runnable"] is True
    assert result["olive_recipe"] == _SUCCESS_PAYLOAD["recipe"]
    assert result["conversion_warnings"] == _SUCCESS_PAYLOAD["conversionWarnings"]
    assert result["effective_state"] == _SUCCESS_PAYLOAD["effectiveState"]
    assert result["schema_errors"] == []
    assert "error" not in result


def test_snake_case_bridge_payload_aliases(monkeypatch: pytest.MonkeyPatch):
    """Objective: snake_case field aliases are accepted from the bridge."""
    # Arrange
    snake_payload = {
        "effective_state": {"modelPath": "m.onnx"},
        "recipe": {"passes": {}},
        "is_runnable": False,
        "schema_errors": [{"path": "x", "message": "bad"}],
        "pipeline_issues": [],
        "local_execution_issues": ["missing olive"],
        "advisories": [],
        "critical_count": 2,
        "warnings": ["legacy warning key"],
    }
    _set_loopback_url(monkeypatch)
    _patch_opener(monkeypatch, return_value=_mock_response(snake_payload))

    # Act
    validation = validate_ui_state_recipe(_SAMPLE_UI_STATE)
    recipe_view = get_recipe_for_ui_state(_SAMPLE_UI_STATE)

    # Assert
    assert validation["is_runnable"] is False
    assert validation["schema_errors"] == snake_payload["schema_errors"]
    assert validation["local_execution_issues"] == ["missing olive"]
    assert validation["pipeline_critical_count"] == 2
    assert recipe_view["olive_recipe"] == {"passes": {}}
    assert recipe_view["conversion_warnings"] == ["legacy warning key"]


def test_blocked_pipeline_still_returns_projection(monkeypatch: pytest.MonkeyPatch):
    """Objective: non-runnable success payloads still project (no Olive run)."""
    # Arrange
    blocked = {
        **_SUCCESS_PAYLOAD,
        "isRunnable": False,
        "schemaErrors": [{"message": "model path required"}],
        "pipelineIssues": [{"severity": "error", "message": "blocked"}],
        "pipelineCriticalCount": 1,
    }
    _set_loopback_url(monkeypatch)
    _patch_opener(monkeypatch, return_value=_mock_response(blocked))

    # Act
    result = validate_ui_state_recipe({"passes": []})

    # Assert
    assert result["is_runnable"] is False
    assert len(result["schema_errors"]) == 1
    assert result["pipeline_critical_count"] == 1
    assert "error" not in result


# ---------------------------------------------------------------------------
# Timeout / unreachable / HTTP errors → studio_unavailable
# ---------------------------------------------------------------------------


def test_timeout_via_urlerror_returns_studio_unavailable(monkeypatch: pytest.MonkeyPatch):
    """Objective: URLError with timed-out reason maps to studio_unavailable."""
    # Arrange
    _set_loopback_url(monkeypatch)
    _patch_opener(monkeypatch, side_effect=URLError(TimeoutError("timed out")))

    # Act
    result = validate_ui_state_recipe(_SAMPLE_UI_STATE)

    # Assert
    assert result["error"] == "studio_unavailable"
    assert "timed out" in result["message"].lower()
    assert "timeout_seconds=" in result.get("detail", "")


def test_timeout_via_timeout_error_returns_studio_unavailable(
    monkeypatch: pytest.MonkeyPatch,
):
    """Objective: bare TimeoutError from opener maps to studio_unavailable."""
    # Arrange
    _set_loopback_url(monkeypatch)
    _patch_opener(monkeypatch, side_effect=TimeoutError())

    # Act
    result = get_recipe_for_ui_state(_SAMPLE_UI_STATE)

    # Assert
    assert result["error"] == "studio_unavailable"
    assert "timed out" in result["message"].lower()


def test_connection_refused_returns_studio_unavailable(monkeypatch: pytest.MonkeyPatch):
    """Objective: unreachable Studio → studio_unavailable, not a crash."""
    # Arrange
    _set_loopback_url(monkeypatch)
    _patch_opener(monkeypatch, side_effect=URLError(ConnectionRefusedError("refused")))

    # Act
    result = validate_ui_state_recipe(_SAMPLE_UI_STATE)

    # Assert
    assert result["error"] == "studio_unavailable"
    assert "not reachable" in result["message"].lower()
    assert "detail" in result


def test_oserror_returns_studio_unavailable(monkeypatch: pytest.MonkeyPatch):
    """Objective: low-level OSError is normalized to studio_unavailable."""
    # Arrange
    _set_loopback_url(monkeypatch)
    _patch_opener(monkeypatch, side_effect=OSError("network down"))

    # Act
    result = validate_ui_state_recipe(_SAMPLE_UI_STATE)

    # Assert
    assert result["error"] == "studio_unavailable"
    assert "failed" in result["message"].lower()
    assert "network down" in result["detail"]


def _http_error(code: int, body: bytes, *, msg: str = "Error") -> HTTPError:
    """Build a urllib HTTPError with a readable body (hdrs must be a Message)."""
    return HTTPError(
        url="http://127.0.0.1:3000" + BRIDGE_PATH,
        code=code,
        msg=msg,
        hdrs=Message(),
        fp=io.BytesIO(body),
    )


def test_http_error_without_json_body_returns_studio_unavailable(
    monkeypatch: pytest.MonkeyPatch,
):
    """Objective: bare HTTP 500 without structured body → studio_unavailable."""
    # Arrange
    _set_loopback_url(monkeypatch)
    _patch_opener(monkeypatch, side_effect=_http_error(500, b"plain text boom", msg="Internal Server Error"))

    # Act
    result = validate_ui_state_recipe(_SAMPLE_UI_STATE)

    # Assert
    assert result["error"] == "studio_unavailable"
    assert "HTTP error" in result["message"]
    assert "status=500" in result["detail"]


def test_http_error_with_error_payload_is_forwarded(monkeypatch: pytest.MonkeyPatch):
    """Objective: HTTPError body with error key is returned as-is."""
    # Arrange
    _set_loopback_url(monkeypatch)
    payload = {"error": "invalid_ui_state", "message": "passes must be an array"}
    err = _http_error(400, json.dumps(payload).encode("utf-8"), msg="Bad Request")
    _patch_opener(monkeypatch, side_effect=err)

    # Act
    result = validate_ui_state_recipe(_SAMPLE_UI_STATE)

    # Assert
    assert result["error"] == "invalid_ui_state"
    assert "passes" in result["message"]


def test_http_error_with_success_shaped_body_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
):
    """Objective: HTTPError with a success-shaped body is not treated as success."""
    _set_loopback_url(monkeypatch)
    err = _http_error(500, json.dumps(_SUCCESS_PAYLOAD).encode("utf-8"), msg="Error")
    _patch_opener(monkeypatch, side_effect=err)

    result = validate_ui_state_recipe(_SAMPLE_UI_STATE)

    assert result["error"] == "studio_unavailable"
    assert "status=500" in result["detail"]


def test_http_status_ge_400_on_success_path(monkeypatch: pytest.MonkeyPatch):
    """Objective: response status >= 400 without raising → studio_unavailable."""
    # Arrange
    _set_loopback_url(monkeypatch)
    _patch_opener(monkeypatch, return_value=_mock_response({"oops": True}, status=503))

    # Act
    result = get_recipe_for_ui_state(_SAMPLE_UI_STATE)

    # Assert
    assert result["error"] == "studio_unavailable"
    assert "status=503" in result["detail"]


# ---------------------------------------------------------------------------
# Input validation / response contracts
# ---------------------------------------------------------------------------


def test_none_ui_state_returns_invalid_ui_state(monkeypatch: pytest.MonkeyPatch):
    """Objective: missing ui_state is rejected before any HTTP call."""
    # Arrange
    _set_loopback_url(monkeypatch)
    opener = _patch_opener(monkeypatch, return_value=_mock_response(_SUCCESS_PAYLOAD))

    # Act
    result = validate_ui_state_recipe(None)

    # Assert
    assert result["error"] == "invalid_ui_state"
    assert "required" in result["message"].lower()
    opener.open.assert_not_called()


def test_non_object_ui_state_returns_invalid_ui_state(monkeypatch: pytest.MonkeyPatch):
    """Objective: non-dict ui_state is rejected."""
    # Arrange
    _set_loopback_url(monkeypatch)
    opener = _patch_opener(monkeypatch, return_value=_mock_response(_SUCCESS_PAYLOAD))

    # Act
    result = get_recipe_for_ui_state("not-an-object")  # type: ignore[arg-type]

    # Assert
    assert result["error"] == "invalid_ui_state"
    assert "JSON object" in result["message"]
    opener.open.assert_not_called()


def test_non_json_bridge_body_returns_invalid_bridge_response(
    monkeypatch: pytest.MonkeyPatch,
):
    """Objective: non-object JSON payload → invalid_bridge_response."""
    # Arrange
    _set_loopback_url(monkeypatch)
    _patch_opener(monkeypatch, return_value=_mock_response(b"not-json{{{"))

    # Act
    result = validate_ui_state_recipe(_SAMPLE_UI_STATE)

    # Assert
    assert result["error"] == "invalid_bridge_response"
    assert "non-object" in result["message"].lower() or "JSON" in result["message"]


def test_json_array_bridge_body_returns_invalid_bridge_response(
    monkeypatch: pytest.MonkeyPatch,
):
    """Objective: JSON array is not a valid bridge object."""
    # Arrange
    _set_loopback_url(monkeypatch)
    _patch_opener(monkeypatch, return_value=_mock_response(b"[1,2,3]"))

    # Act
    result = validate_ui_state_recipe(_SAMPLE_UI_STATE)

    # Assert
    assert result["error"] == "invalid_bridge_response"


def test_missing_required_fields_returns_invalid_bridge_response(
    monkeypatch: pytest.MonkeyPatch,
):
    """Objective: payload missing effectiveState/recipe/isRunnable is rejected."""
    # Arrange
    _set_loopback_url(monkeypatch)
    _patch_opener(
        monkeypatch,
        return_value=_mock_response({"effectiveState": {}, "isRunnable": True}),
    )

    # Act
    result = validate_ui_state_recipe(_SAMPLE_UI_STATE)

    # Assert
    assert result["error"] == "invalid_bridge_response"
    assert "missing" in result["message"].lower()
    assert "recipe" in result.get("detail", "")


def test_wrong_types_on_required_fields(monkeypatch: pytest.MonkeyPatch):
    """Objective: wrong types for required success fields → invalid_bridge_response."""
    # Arrange
    _set_loopback_url(monkeypatch)
    bad = {
        "effectiveState": "not-an-object",
        "recipe": {"ok": True},
        "isRunnable": True,
    }
    _patch_opener(monkeypatch, return_value=_mock_response(bad))

    # Act
    result = get_recipe_for_ui_state(_SAMPLE_UI_STATE)

    # Assert
    assert result["error"] == "invalid_bridge_response"
    assert "effectiveState" in result["message"]


def test_is_runnable_must_be_boolean(monkeypatch: pytest.MonkeyPatch):
    """Objective: isRunnable must be a real boolean."""
    # Arrange
    _set_loopback_url(monkeypatch)
    bad = {
        "effectiveState": {},
        "recipe": {},
        "isRunnable": "yes",
    }
    _patch_opener(monkeypatch, return_value=_mock_response(bad))

    # Act
    result = validate_ui_state_recipe(_SAMPLE_UI_STATE)

    # Assert
    assert result["error"] == "invalid_bridge_response"
    assert "isRunnable" in result["message"]


def test_bridge_error_payload_without_success_keys_is_returned(
    monkeypatch: pytest.MonkeyPatch,
):
    """Objective: structured error objects from bridge pass through unchanged."""
    # Arrange
    _set_loopback_url(monkeypatch)
    err_payload = {
        "error": "studio_unavailable",
        "message": "rate limited",
        "detail": "retry later",
    }
    _patch_opener(monkeypatch, return_value=_mock_response(err_payload))

    # Act
    result = validate_ui_state_recipe(_SAMPLE_UI_STATE)

    # Assert
    assert result == err_payload


def test_list_coercion_for_scalar_issue_fields(monkeypatch: pytest.MonkeyPatch):
    """Objective: scalar issue/advisory values are wrapped into lists."""
    # Arrange
    _set_loopback_url(monkeypatch)
    payload = {
        **_SUCCESS_PAYLOAD,
        "schemaErrors": "single-error",
        "advisories": "one-advisory",
    }
    _patch_opener(monkeypatch, return_value=_mock_response(payload))

    # Act
    result = validate_ui_state_recipe(_SAMPLE_UI_STATE)

    # Assert
    assert result["schema_errors"] == ["single-error"]
    assert result["advisories"] == ["one-advisory"]


# ---------------------------------------------------------------------------
# Registration smoke (import path used by MCP server)
# ---------------------------------------------------------------------------


def test_tools_registered_in_tool_imports():
    """Objective: both bridge tools are registered for MCP dispatch."""
    # Arrange / Act
    from olive_mcp_server.mcp_server import _TOOL_IMPORTS

    # Assert
    assert "validate_ui_state_recipe" in _TOOL_IMPORTS
    assert "get_recipe_for_ui_state" in _TOOL_IMPORTS
    assert _TOOL_IMPORTS["validate_ui_state_recipe"][1] == "validate_ui_state_recipe"
    assert _TOOL_IMPORTS["get_recipe_for_ui_state"][1] == "get_recipe_for_ui_state"


def test_call_tool_dispatches_validate_with_mocked_bridge(monkeypatch: pytest.MonkeyPatch):
    """Objective: mcp_server.call_tool reaches validate_ui_state_recipe (mocked HTTP)."""
    # Arrange
    from olive_mcp_server.mcp_server import call_tool

    _set_loopback_url(monkeypatch)
    _patch_opener(monkeypatch, return_value=_mock_response(_SUCCESS_PAYLOAD))

    # Act
    result = call_tool("validate_ui_state_recipe", {"ui_state": _SAMPLE_UI_STATE})

    # Assert
    assert isinstance(result, dict)
    assert result.get("is_runnable") is True
    assert "olive_recipe" not in result
    assert "error" not in result


def test_call_tool_dispatches_get_recipe_when_studio_unavailable(
    monkeypatch: pytest.MonkeyPatch,
):
    """Objective: call_tool surfaces studio_unavailable without raising."""
    # Arrange
    from olive_mcp_server.mcp_server import call_tool

    # ENV cleared by autouse — no OLIVE_STUDIO_API_URL

    # Act
    result = call_tool("get_recipe_for_ui_state", {"ui_state": _SAMPLE_UI_STATE})

    # Assert
    assert result["error"] == "studio_unavailable"
    assert ENV_API_URL in result["message"]
