"""Phase 3: Autonomous agent error diagnosis with recipe repair.

Accepts an error message and current recipe, delegates to the existing
troubleshooting KB, and (when possible) produces a repaired recipe by
applying the KB's ``updated_config`` as an RFC 7386 JSON Merge Patch.

Tools:
  - diagnose_and_fix
"""

from __future__ import annotations

import copy
import json
import logging
import os
from typing import Any

from .studio_loopback import err, studio_request
from .troubleshooting import troubleshoot_olive_error

logger = logging.getLogger(__name__)

_MAX_ERROR_LEN = 4000
_MAX_CONFIG_CONTEXT_LEN = 200

_VALIDATE_TOOL_PATH = "/api/olive/jobs/validate"


def _strip_trust_remote_code(recipe: dict[str, Any]) -> bool:
    """Recursively remove or override ``trust_remote_code: true`` from *recipe*.

    Returns True if any automated elevation was stripped. Patches that set the
    flag to ``false`` are preserved.
    """
    overridden = False

    def _walk(obj: Any) -> None:
        nonlocal overridden
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key == "trust_remote_code" and value is True:
                    obj[key] = False
                    overridden = True
                else:
                    _walk(value)
        elif isinstance(obj, list):
            for item in obj:
                _walk(item)

    _walk(recipe)
    return overridden


def _apply_merge_patch(target: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    """Apply an RFC 7386 JSON Merge Patch to *target* (deep-copied first).

    For each key in *patch*:
      - If the value is ``None`` -> remove that key from target.
      - If the value is a dict and the target key is also a dict -> recurse.
      - Otherwise -> set/override the key.

    Returns the modified copy (original *target* is not mutated).
    """
    result = copy.deepcopy(target)
    _merge_in_place(result, patch)
    return result


def _merge_in_place(target: dict[str, Any], patch: dict[str, Any]) -> None:
    """Recursively apply merge-patch *patch* onto *target* in place (RFC 7386)."""
    for key, value in patch.items():
        if value is None:
            target.pop(key, None)
        elif isinstance(value, dict):
            # RFC 7386: if target value is not a dict, create a fresh one
            if not isinstance(target.get(key), dict):
                target[key] = {}
            _merge_in_place(target[key], value)
        else:
            target[key] = value


def _build_config_context(
    recipe: dict[str, Any],
    hardware_probe: dict[str, Any] | None,
) -> str:
    """Build a truncated config_context string for troubleshooting matching."""
    parts: list[str] = []

    # Summarize recipe keys for matching context
    recipe_keys = sorted(recipe.keys())
    parts.append(f"recipe_keys=[{', '.join(recipe_keys)}]")

    if hardware_probe:
        probe_summary = json.dumps(hardware_probe, separators=(",", ":"))
        parts.append(f"hw={probe_summary}")

    context = "; ".join(parts)
    if len(context) > _MAX_CONFIG_CONTEXT_LEN:
        context = context[: _MAX_CONFIG_CONTEXT_LEN - 3] + "..."
    return context


def _describe_changes(
    original: dict[str, Any],
    fixed: dict[str, Any],
    prefix: str = "",
) -> list[str]:
    """Generate human-readable change descriptions by diffing original vs fixed.

    Reports effective additions, removals, replacements, and nested changes
    rather than relying only on the updated_config patch. This catches
    empty-object patches that replace scalars or create missing keys, and
    removals of absent keys (no-op).
    """
    changes: list[str] = []
    all_keys = set(original) | set(fixed)
    for key in sorted(all_keys):
        path = f"{prefix}.{key}" if prefix else key
        old_val = original.get(key)
        new_val = fixed.get(key)

        if key not in original:
            # Addition: key exists in fixed but not in original
            changes.append(f"Added {path} = {json.dumps(new_val)}")
        elif key not in fixed:
            # Removal: key existed in original but removed from fixed
            changes.append(f"Removed {path}")
        elif isinstance(old_val, dict) and isinstance(new_val, dict):
            # Nested dict: recurse
            changes.extend(_describe_changes(old_val, new_val, prefix=path))
        elif old_val != new_val:
            # Replacement
            changes.append(f"Set {path} to {json.dumps(new_val)}")
        # else: unchanged — no description needed
    return changes


def _determine_confidence(diagnosis: dict[str, Any]) -> str:
    """Map diagnosis characteristics to fix confidence level.

    - "high": KB entry matched with ``updated_config`` present and ``applyable=True``
    - "medium": KB entry matched but fix was rule-based inference (has entry, no updated_config)
    - "low": Weak KB match (entry found but no actionable content)
    - "none": No match at all
    """
    matched_entry = diagnosis.get("matched_entry")
    if matched_entry is None:
        return "none"

    updated_config = diagnosis.get("updated_config")
    applyable = diagnosis.get("applyable", False)

    if isinstance(updated_config, dict) and updated_config and applyable:
        return "high"

    # Has a matched entry with some content (root_cause, workaround) but no
    # direct config fix -> rule-based inference
    root_cause = diagnosis.get("root_cause", "")
    workaround = diagnosis.get("workaround", "")
    if root_cause or workaround:
        return "medium"

    return "low"


def _validate_fixed_recipe(fixed_recipe: dict[str, Any]) -> bool:
    """Attempt to validate the fixed recipe via the Studio bridge.

    Returns True if validation succeeds, False if Studio is unreachable or
    validation fails.
    """
    response = studio_request(
        "POST",
        _VALIDATE_TOOL_PATH,
        body={"recipe": fixed_recipe},
    )
    # Non-dict responses (None, lists, etc.) cannot carry a validation verdict.
    if not isinstance(response, dict):
        return False
    # If Studio is down, returned an error, or rejected the recipe, treat it as
    # not validated. The endpoint returns validation failures with valid=false.
    if isinstance(response.get("error"), str) and response["error"]:
        return False
    return response.get("valid") is True


def diagnose_and_fix(
    error_message: str,
    recipe: dict[str, Any],
    hardware_probe: dict[str, Any] | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Diagnose an optimization error and attempt automated recipe repair.

    Args:
        error_message: The error text or traceback snippet (1-4000 chars).
        recipe: The current Olive optimization recipe as a dict.
        hardware_probe: Optional hardware context for hardware-aware diagnosis.
        session_id: Optional Studio agent-loop session ID.

    Returns:
        Structured result with diagnosis, optional fixed recipe, change list,
        validation status, and confidence level. Always includes
        ``side_effect: False``.
    """
    try:
        # --- Input validation (first in priority order) ---
        if not isinstance(error_message, str) or not (1 <= len(error_message) <= _MAX_ERROR_LEN):
            return err(
                "invalid_input",
                f"error_message must be 1\u2013{_MAX_ERROR_LEN} characters",
            )

        if not isinstance(recipe, dict):
            return err(
                "invalid_input",
                "recipe must be a JSON object",
            )

        from .studio_loopback import ENV_API_URL, _ensure_session, _update_session

        active_session_id: str | None = None
        if session_id or os.environ.get(ENV_API_URL):
            active_session_id, session = _ensure_session(session_id)
            if active_session_id is None:
                return session
        else:
            session = {}

        # --- Build config context for matching ---
        config_context = _build_config_context(recipe, hardware_probe)

        # --- Invoke troubleshooting KB ---
        diagnosis = troubleshoot_olive_error(
            error_message=error_message,
            pass_name="",
            config_context=config_context,
        )

        # --- Determine fix applicability ---
        updated_config = diagnosis.get("updated_config")
        applyable = diagnosis.get("applyable", False)

        fix_confidence = _determine_confidence(diagnosis)

        if isinstance(updated_config, dict) and updated_config and applyable:
            # Apply merge patch to produce fixed recipe
            fixed_recipe = _apply_merge_patch(recipe, updated_config)
            # KB patches must never automatically enable trust_remote_code.
            # Strip or override any nested trust_remote_code: true while
            # preserving patches that explicitly set it to false.
            trust_overridden = _strip_trust_remote_code(fixed_recipe)
            changes_made = _describe_changes(recipe, fixed_recipe)
            if trust_overridden:
                changes_made.append(
                    "trust_remote_code requires deliberate user action; "
                    "automated elevation was removed from the fixed recipe."
                )

            # Best-effort validation through Studio bridge
            recipe_validated = _validate_fixed_recipe(fixed_recipe)
        else:
            fixed_recipe = None
            changes_made = []
            recipe_validated = False

        result = {
            "diagnosis": diagnosis,
            "fixed_recipe": fixed_recipe,
            "changes_made": changes_made,
            "recipe_validated": recipe_validated,
            "fix_confidence": fix_confidence,
            "side_effect": False,
        }
        if active_session_id:
            update = _update_session(
                active_session_id,
                lastRecipe=fixed_recipe or recipe,
                lastFailure=error_message,
                success=False,
                diagnosticNotes=[
                    *session.get("diagnosticNotes", [])[-49:],
                    str(diagnosis.get("title") or "Diagnosis completed."),
                ],
            )
            if isinstance(update.get("error"), str) and update["error"]:
                return update
            result["session_id"] = active_session_id
        return result

    except Exception as exc:
        logger.warning("diagnose_and_fix unexpected error", exc_info=True)
        return err("internal_error", type(exc).__name__)
