# Feature: v0.3-agent-mcp-tools, Property 3: Unparseable Intent Rejection
"""Property-based tests for plan_optimization.

Property 3: For any random string that contains NONE of the recognized
hardware keywords, model reference patterns, or optimization keywords,
plan_optimization SHALL return an error with code "unparseable_intent".

Validates: Requirements 3.5
"""

from __future__ import annotations

from unittest.mock import patch

from hypothesis import given, settings
from hypothesis import strategies as st

from olive_mcp_server.tools.agent_planner import (
    _parse_intent,
    plan_optimization,
)

# ---------------------------------------------------------------------------
# Trigger detection — reuse production symbols so generated-text filtering
# stays synchronized with production parsing.
# ---------------------------------------------------------------------------


def _contains_any_trigger(text: str) -> bool:
    """Check if the text would be parsed by the production _parse_intent.

    Delegates to the production parser so the property test never drifts from
    the actual trigger definitions in agent_planner.py.
    """
    parsed = _parse_intent(text)
    return any(parsed.values())


# ---------------------------------------------------------------------------
# Strategy: generate safe text that avoids all triggers
# ---------------------------------------------------------------------------

# Use a character set that's unlikely to form trigger words:
# Letters, digits, space, and basic punctuation — then filter out any
# accidental trigger matches.
_safe_text = st.text(
    alphabet=st.characters(
        whitelist_categories=("Lu", "Ll", "Nd"),
        whitelist_characters=".,!? ",
    ),
    min_size=1,
    max_size=100,
).filter(lambda s: s.strip() and not _contains_any_trigger(s))


# ---------------------------------------------------------------------------
# Property 3: Unparseable Intent Rejection
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(intent=_safe_text)
def test_unparseable_intent_returns_error(intent: str):
    """Property 3: Random strings with no hardware/model/optimization keywords
    SHALL produce an 'unparseable_intent' error.

    Validates: Requirements 3.5
    """
    with patch("olive_mcp_server.tools.agent_planner.validate_ui_state_recipe") as mock_studio:
        result = plan_optimization(intent)

    assert isinstance(result, dict), f"Expected dict, got {type(result)}"
    assert "error" in result, f"Expected 'error' key in result for intent={intent!r}, got {result}"
    assert result["error"] == "unparseable_intent", (
        f"Expected error code 'unparseable_intent', got {result['error']!r} for intent={intent!r}"
    )
    assert "message" in result, "Error response must include 'message' field"
    assert isinstance(result["message"], str) and len(result["message"]) > 0, (
        "Error 'message' must be a non-empty string"
    )
    # Should NOT have side_effect key on error path
    assert "side_effect" not in result, "Error response should not contain 'side_effect' field"
    # studio_request should never be called for unparseable intents
    mock_studio.assert_not_called()
