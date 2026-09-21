"""Property-based tests for diagnose_and_fix (agent_diagnosis module).

Feature: v0.3-agent-mcp-tools

Uses hypothesis to verify universal correctness properties that must hold
across all valid inputs for the JSON Merge Patch logic and confidence mapping.
"""

from __future__ import annotations

import json
from typing import Any

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from olive_mcp_server.tools.agent_diagnosis import _apply_merge_patch, _determine_confidence

# ---------------------------------------------------------------------------
# Feature: v0.3-agent-mcp-tools, Property 4: JSON Merge Patch Correctness
#
# For any recipe R and any updated_config U:
#   (a) Non-null values in U override corresponding keys in R
#   (b) Null values in U result in key removal from result
#   (c) Keys in R not mentioned in U are preserved unchanged
#
# Validates: Requirements 5.3
# ---------------------------------------------------------------------------

# Strategy for generating leaf values (non-None, non-dict)
_leaf_values = st.one_of(
    st.integers(min_value=-1000, max_value=1000),
    st.text(min_size=1, max_size=20),
    st.booleans(),
    st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
)

# Strategy for flat dict keys
_dict_keys = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N"), whitelist_characters="_-"),
    min_size=1,
    max_size=10,
)

# Strategy for recipe dicts (flat, no None values since that's not valid in a real recipe)
_recipe_strategy = st.dictionaries(
    keys=_dict_keys,
    values=_leaf_values,
    min_size=0,
    max_size=8,
)

# Strategy for patch dicts (may contain None values for removal)
_patch_strategy = st.dictionaries(
    keys=_dict_keys,
    values=st.one_of(st.none(), _leaf_values),
    min_size=0,
    max_size=8,
)


class TestJsonMergePatchCorrectness:
    """Property 4: JSON Merge Patch Correctness (RFC 7386)."""

    @given(recipe=_recipe_strategy, patch=_patch_strategy)
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_non_null_values_override(self, recipe: dict[str, Any], patch: dict[str, Any]) -> None:
        """Non-null values in patch override corresponding keys in result."""
        result = _apply_merge_patch(recipe, patch)

        for key, value in patch.items():
            if value is not None:
                assert key in result, f"Key '{key}' with non-null value should be present in result"
                assert result[key] == value, f"Key '{key}' should be overridden to {value!r}, got {result[key]!r}"

    @given(recipe=_recipe_strategy, patch=_patch_strategy)
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_null_values_remove_keys(self, recipe: dict[str, Any], patch: dict[str, Any]) -> None:
        """Null values in patch result in key removal from result."""
        result = _apply_merge_patch(recipe, patch)

        for key, value in patch.items():
            if value is None:
                assert key not in result, (
                    f"Key '{key}' with None value should be removed from result, but found {result.get(key)!r}"
                )

    @given(recipe=_recipe_strategy, patch=_patch_strategy)
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_absent_keys_preserved(self, recipe: dict[str, Any], patch: dict[str, Any]) -> None:
        """Keys in recipe not mentioned in patch are preserved unchanged."""
        result = _apply_merge_patch(recipe, patch)

        for key, value in recipe.items():
            if key not in patch:
                assert key in result, f"Key '{key}' not in patch should be preserved in result"
                assert result[key] == value, f"Key '{key}' should be unchanged ({value!r}), got {result[key]!r}"

    @given(recipe=_recipe_strategy, patch=_patch_strategy)
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_original_recipe_not_mutated(self, recipe: dict[str, Any], patch: dict[str, Any]) -> None:
        """The original recipe dict is not mutated by merge patch."""
        recipe_copy = json.loads(json.dumps(recipe))
        _apply_merge_patch(recipe, patch)
        assert recipe == recipe_copy, "Original recipe was mutated by _apply_merge_patch"

    @given(recipe=_recipe_strategy, patch=_patch_strategy)
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_result_json_round_trip(self, recipe: dict[str, Any], patch: dict[str, Any]) -> None:
        """Result of merge patch is JSON-serializable and round-trips correctly."""
        result = _apply_merge_patch(recipe, patch)
        assert json.loads(json.dumps(result)) == result

    @given(
        recipe=st.dictionaries(
            keys=_dict_keys,
            values=st.dictionaries(keys=_dict_keys, values=_leaf_values, min_size=1, max_size=4),
            min_size=1,
            max_size=4,
        ),
        patch=st.dictionaries(
            keys=_dict_keys,
            values=st.one_of(
                st.none(),
                st.dictionaries(
                    keys=_dict_keys,
                    values=st.one_of(st.none(), _leaf_values),
                    min_size=1,
                    max_size=4,
                ),
            ),
            min_size=1,
            max_size=4,
        ),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_nested_merge_patch_semantics(self, recipe: dict[str, Any], patch: dict[str, Any]) -> None:
        """Nested dicts in patch recurse into nested dicts in recipe (RFC 7386 recursive rule)."""
        result = _apply_merge_patch(recipe, patch)

        for key, patch_value in patch.items():
            if patch_value is None:
                # Null removes key entirely
                assert key not in result
            elif isinstance(patch_value, dict) and isinstance(recipe.get(key), dict):
                # Nested dict: should recurse
                assert key in result
                nested_result = result[key]
                for sub_key, sub_value in patch_value.items():
                    if sub_value is None:
                        assert sub_key not in nested_result
                    else:
                        assert nested_result.get(sub_key) == sub_value
            elif isinstance(patch_value, dict):
                # RFC 7386 applies an object patch to an empty object when the
                # target value is absent or not an object, including removing
                # nested keys whose patch value is null. Independently derive
                # the expected nested object from patch_value (do not call
                # _apply_merge_patch to avoid circular verification).
                expected_nested: dict[str, Any] = {}
                for sub_key, sub_value in patch_value.items():
                    if sub_value is None:
                        continue
                    expected_nested[sub_key] = sub_value
                assert key in result
                assert result[key] == expected_nested
            else:
                # Non-dict non-null: override
                assert result.get(key) == patch_value

    def test_nested_merge_patch_preserves_existing_child_key(self) -> None:
        """An existing child key not in patch_value remains in nested_result."""
        recipe = {"outer": {"keep_me": "original", "replace_me": "old"}}
        patch = {"outer": {"replace_me": "new", "remove_me": None}}

        result = _apply_merge_patch(recipe, patch)

        assert result["outer"]["keep_me"] == "original"
        assert result["outer"]["replace_me"] == "new"
        assert "remove_me" not in result["outer"]


# ---------------------------------------------------------------------------
# Feature: v0.3-agent-mcp-tools, Property 5: Fix Confidence Determinism
#
# For any diagnosis result structure, confidence maps deterministically:
#   - matched_entry is not None AND updated_config non-empty AND applyable=True -> "high"
#   - matched_entry is not None AND (root_cause or workaround non-empty) AND no
#     updated_config -> "medium"
#   - matched_entry is not None AND no root_cause, no workaround, no updated_config -> "low"
#   - matched_entry is None -> "none"
#
# Validates: Requirements 5.7
# ---------------------------------------------------------------------------

# Strategy for generating "high" confidence diagnosis
_high_confidence_diagnosis = st.fixed_dictionaries(
    {
        "matched_entry": st.text(min_size=1, max_size=30),
        "updated_config": st.dictionaries(
            keys=_dict_keys,
            values=_leaf_values,
            min_size=1,
            max_size=5,
        ),
        "applyable": st.just(True),
    },
    optional={
        "root_cause": st.text(min_size=0, max_size=50),
        "workaround": st.text(min_size=0, max_size=50),
    },
)

# Strategy for "medium" confidence diagnosis (has root_cause or workaround, no updated_config)
_medium_confidence_diagnosis = st.fixed_dictionaries(
    {
        "matched_entry": st.text(min_size=1, max_size=30),
    },
    optional={
        "root_cause": st.text(min_size=1, max_size=50),
        "workaround": st.text(min_size=1, max_size=50),
    },
).filter(lambda d: d.get("root_cause", "") or d.get("workaround", ""))

# Strategy for "low" confidence diagnosis (matched entry but nothing actionable)
_low_confidence_diagnosis = st.fixed_dictionaries(
    {
        "matched_entry": st.text(min_size=1, max_size=30),
        "root_cause": st.just(""),
        "workaround": st.just(""),
    },
)

# Strategy for "none" confidence (no matched entry)
_none_confidence_diagnosis = st.fixed_dictionaries(
    {
        "matched_entry": st.none(),
    },
    optional={
        "root_cause": st.text(min_size=0, max_size=50),
        "workaround": st.text(min_size=0, max_size=50),
        "updated_config": st.dictionaries(keys=_dict_keys, values=_leaf_values, min_size=0, max_size=3),
        "applyable": st.booleans(),
    },
)


class TestFixConfidenceDeterminism:
    """Property 5: Fix Confidence Determinism."""

    @given(diagnosis=_high_confidence_diagnosis)
    @settings(max_examples=100)
    def test_high_confidence_when_updated_config_and_applyable(self, diagnosis: dict[str, Any]) -> None:
        """matched_entry + non-empty updated_config + applyable=True -> 'high'."""
        result = _determine_confidence(diagnosis)
        assert result == "high", f"Expected 'high' for diagnosis with updated_config and applyable=True, got '{result}'"

    @given(diagnosis=_medium_confidence_diagnosis)
    @settings(max_examples=100)
    def test_medium_confidence_when_rule_based(self, diagnosis: dict[str, Any]) -> None:
        """matched_entry + root_cause/workaround but no updated_config -> 'medium'."""
        result = _determine_confidence(diagnosis)
        assert result == "medium", (
            f"Expected 'medium' for diagnosis with root_cause/workaround but no config, got '{result}'"
        )

    @given(diagnosis=_low_confidence_diagnosis)
    @settings(max_examples=100)
    def test_low_confidence_when_weak_match(self, diagnosis: dict[str, Any]) -> None:
        """matched_entry but no root_cause, no workaround, no updated_config -> 'low'."""
        result = _determine_confidence(diagnosis)
        assert result == "low", f"Expected 'low' for diagnosis with matched_entry but no content, got '{result}'"

    @given(diagnosis=_none_confidence_diagnosis)
    @settings(max_examples=100)
    def test_none_confidence_when_no_match(self, diagnosis: dict[str, Any]) -> None:
        """matched_entry is None -> 'none' regardless of other fields."""
        result = _determine_confidence(diagnosis)
        assert result == "none", f"Expected 'none' for diagnosis with no matched_entry, got '{result}'"

    @given(
        diagnosis=st.one_of(
            _high_confidence_diagnosis,
            _medium_confidence_diagnosis,
            _low_confidence_diagnosis,
            _none_confidence_diagnosis,
        )
    )
    @settings(max_examples=100)
    def test_confidence_always_valid_value(self, diagnosis: dict[str, Any]) -> None:
        """Confidence is always one of the four valid values."""
        result = _determine_confidence(diagnosis)
        assert result in ("high", "medium", "low", "none"), f"Confidence '{result}' is not a valid value"

    @given(
        diagnosis=st.one_of(
            _high_confidence_diagnosis,
            _medium_confidence_diagnosis,
            _low_confidence_diagnosis,
            _none_confidence_diagnosis,
        )
    )
    @settings(max_examples=100)
    def test_confidence_is_deterministic(self, diagnosis: dict[str, Any]) -> None:
        """Same input always produces same output (deterministic mapping)."""
        result1 = _determine_confidence(diagnosis)
        result2 = _determine_confidence(diagnosis)
        assert result1 == result2, f"Non-deterministic: same input produced '{result1}' and '{result2}'"

    def test_high_confidence_with_applyable_false_not_high(self) -> None:
        """updated_config present but applyable=False -> NOT 'high'."""
        diagnosis = {
            "matched_entry": "some_entry",
            "updated_config": {"key": "value"},
            "applyable": False,
            "root_cause": "Something went wrong",
        }
        result = _determine_confidence(diagnosis)
        # With applyable=False, updated_config doesn't count -> falls to medium (has root_cause)
        assert result == "medium"

    def test_high_confidence_empty_updated_config_not_high(self) -> None:
        """Empty updated_config dict -> NOT 'high' even with applyable=True."""
        diagnosis = {
            "matched_entry": "some_entry",
            "updated_config": {},
            "applyable": True,
            "root_cause": "Something went wrong",
        }
        result = _determine_confidence(diagnosis)
        # Empty updated_config doesn't count -> falls to medium (has root_cause)
        assert result == "medium"
