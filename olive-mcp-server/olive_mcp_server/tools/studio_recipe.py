"""Tools: validate_ui_state_recipe and get_recipe_for_ui_state.

Bridge to a local Olive Studio HTTP API for UIState-backed recipe validation
and building. Base URL comes only from ``OLIVE_STUDIO_API_URL`` (loopback
HTTP(S) only). Never shells out to or runs Olive.
"""

from __future__ import annotations

from typing import Any

from .studio_loopback import (
    DEFAULT_TIMEOUT_SECONDS,
    ENV_API_URL,
    resolve_studio_base,
    studio_request,
)
from .studio_loopback import (
    err as _err,
)

BRIDGE_PATH = "/api/mcp/studio-recipe"

# Re-export for tests that import these symbols from studio_recipe.
__all__ = [
    "BRIDGE_PATH",
    "DEFAULT_TIMEOUT_SECONDS",
    "ENV_API_URL",
    "get_recipe_for_ui_state",
    "validate_ui_state_recipe",
]

# TS camelCase → snake_case aliases accepted on bridge payloads.
_FIELD_ALIASES: dict[str, str] = {
    "effectiveState": "effective_state",
    "recipe": "recipe",
    "isRunnable": "is_runnable",
    "schemaErrors": "schema_errors",
    "pipelineIssues": "pipeline_issues",
    "localExecutionIssues": "local_execution_issues",
    "advisories": "advisories",
    "pipelineCriticalCount": "pipeline_critical_count",
    "criticalCount": "critical_count",
    "warnings": "warnings",
    "conversionWarnings": "conversion_warnings",
}

_REQUIRED_SUCCESS = ("effectiveState", "recipe", "isRunnable")


def _resolve_bridge_endpoint() -> tuple[str | None, dict[str, Any] | None]:
    """Compose the fixed Studio recipe-bridge URL from the validated base."""
    base, err = resolve_studio_base()
    if err is not None:
        return None, err
    return f"{base}{BRIDGE_PATH}", None


def _request_studio_recipe(
    ui_state: dict[str, Any],
    *,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """POST ui_state to the fixed Studio bridge; failures → studio_unavailable."""
    return studio_request(
        "POST",
        BRIDGE_PATH,
        body={"uiState": ui_state},
        timeout=timeout,
    )


def _get(payload: dict[str, Any], camel: str, default: Any = None) -> Any:
    """Read camelCase field, falling back to snake_case alias."""
    if camel in payload:
        return payload[camel]
    snake = _FIELD_ALIASES.get(camel)
    if snake and snake in payload:
        return payload[snake]
    return default


def _has_required_success_keys(payload: dict[str, Any]) -> bool:
    for key in _REQUIRED_SUCCESS:
        snake = _FIELD_ALIASES.get(key, key)
        if key not in payload and snake not in payload:
            return False
    return True


def _is_error_payload(payload: dict[str, Any]) -> bool:
    return "error" in payload and not _has_required_success_keys(payload)


def _check_success_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Return an error dict if payload is not a successful bridge result."""
    if not _has_required_success_keys(payload):
        missing = [
            k
            for k in _REQUIRED_SUCCESS
            if _get(payload, k) is None and k not in payload and _FIELD_ALIASES.get(k, k) not in payload
        ]
        return _err(
            "invalid_bridge_response",
            "Olive Studio bridge payload missing required fields.",
            detail=f"missing={missing}",
        )

    effective = _get(payload, "effectiveState")
    if not isinstance(effective, dict):
        return _err("invalid_bridge_response", "effectiveState must be an object.")

    recipe = _get(payload, "recipe")
    if not isinstance(recipe, dict):
        return _err("invalid_bridge_response", "recipe must be an object.")

    runnable = _get(payload, "isRunnable")
    if not isinstance(runnable, bool):
        return _err("invalid_bridge_response", "isRunnable must be a boolean.")

    return None


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _validation_view(payload: dict[str, Any]) -> dict[str, Any]:
    """Compact validation projection (no full recipe)."""
    critical = _get(payload, "pipelineCriticalCount")
    if critical is None:
        critical = _get(payload, "criticalCount")

    return {
        "effective_state": _get(payload, "effectiveState"),
        "schema_errors": _as_list(_get(payload, "schemaErrors", [])),
        "pipeline_issues": _as_list(_get(payload, "pipelineIssues", [])),
        "pipeline_critical_count": critical,
        "local_execution_issues": _as_list(_get(payload, "localExecutionIssues", [])),
        "advisories": _as_list(_get(payload, "advisories", [])),
        "is_runnable": bool(_get(payload, "isRunnable")),
    }


def _recipe_view(payload: dict[str, Any]) -> dict[str, Any]:
    """Validation view plus olive_recipe and conversion_warnings."""
    conversion = _get(payload, "conversionWarnings")
    warnings = _as_list(conversion if conversion is not None else _get(payload, "warnings", []))
    return {
        **_validation_view(payload),
        "olive_recipe": _get(payload, "recipe"),
        "conversion_warnings": warnings,
    }


def _evaluate_ui_state(ui_state: Any) -> dict[str, Any]:
    """Validate input, call bridge once, check response shape."""
    if ui_state is None:
        return _err("invalid_ui_state", "ui_state is required and must be a JSON object.")
    if not isinstance(ui_state, dict):
        return _err("invalid_ui_state", "ui_state must be a JSON object.")

    payload = _request_studio_recipe(ui_state)
    if _is_error_payload(payload):
        return payload

    shape_err = _check_success_payload(payload)
    return shape_err if shape_err is not None else payload


def validate_ui_state_recipe(ui_state: dict[str, Any] | None = None) -> dict[str, Any]:
    """Validate a (partial) UIState via the local Olive Studio bridge.

    Returns effective state, schema/pipeline/runtime issues, advisories, and
    ``is_runnable``. Does not run Olive.

    Args:
        ui_state: Complete or allowed-partial UI state object.

    Returns:
        Validation projection, or structured error
        (``studio_unavailable`` / ``invalid_ui_state`` / ``invalid_bridge_response``).
    """
    payload = _evaluate_ui_state(ui_state)
    return payload if _is_error_payload(payload) else _validation_view(payload)


def get_recipe_for_ui_state(ui_state: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build and validate an Olive recipe for a (partial) UIState via Studio.

    Same evaluation as ``validate_ui_state_recipe``, plus ``olive_recipe`` and
    ``conversion_warnings``. Does not run Olive.

    Args:
        ui_state: Complete or allowed-partial UI state object.

    Returns:
        Full recipe projection, or structured error
        (``studio_unavailable`` / ``invalid_ui_state`` / ``invalid_bridge_response``).
    """
    payload = _evaluate_ui_state(ui_state)
    return payload if _is_error_payload(payload) else _recipe_view(payload)
