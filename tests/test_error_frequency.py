"""Tests for the error frequency tracking feature in troubleshoot_olive_error.

The frequency tracker is inspired by the UI's import error timer pattern:
it records each occurrence with a timestamp and count, and provides a
human-readable frequency label so callers can distinguish transient from
persistent errors.
"""

import pytest

from olive_mcp_server.tools.troubleshooting import (
    get_error_frequency_summary,
    reset_frequency_store,
    troubleshoot_olive_error,
)


def _setup():
    """Reset the frequency store before each test."""
    reset_frequency_store()


# ---------------------------------------------------------------------------
# Response shape — frequency field must always be present
# ---------------------------------------------------------------------------


def test_frequency_field_always_present():
    _setup()
    resp = troubleshoot_olive_error("anything")
    assert "frequency" in resp
    freq = resp["frequency"]
    for key in ("occurrence_count", "first_seen", "last_seen", "label"):
        assert key in freq, f"Missing frequency key: {key}"


def test_frequency_field_for_unmatched_error():
    _setup()
    resp = troubleshoot_olive_error("completely unknown error 99999")
    freq = resp["frequency"]
    assert freq["occurrence_count"] == 1
    assert freq["label"] == "first_occurrence"
    assert isinstance(freq["first_seen"], str)
    assert isinstance(freq["last_seen"], str)


def test_frequency_field_for_matched_error():
    _setup()
    resp = troubleshoot_olive_error(
        "CUDA out of memory during quantization",
        pass_name="OnnxQuantization",
    )
    freq = resp["frequency"]
    assert freq["occurrence_count"] == 1
    assert freq["label"] == "first_occurrence"
    assert resp["matched_entry"] == "oom-quantization"


# ---------------------------------------------------------------------------
# Occurrence counting
# ---------------------------------------------------------------------------


def test_occurrence_count_increments_on_repeated_error():
    _setup()
    r1 = troubleshoot_olive_error("CUDA out of memory")
    r2 = troubleshoot_olive_error("CUDA out of memory")
    r3 = troubleshoot_olive_error("CUDA out of memory")
    assert r1["frequency"]["occurrence_count"] == 1
    assert r2["frequency"]["occurrence_count"] == 2
    assert r3["frequency"]["occurrence_count"] == 3


def test_different_errors_tracked_independently():
    _setup()
    r1 = troubleshoot_olive_error("CUDA out of memory")
    r2 = troubleshoot_olive_error("ONNX export failed")
    r3 = troubleshoot_olive_error("CUDA out of memory")
    assert r1["frequency"]["occurrence_count"] == 1
    assert r2["frequency"]["occurrence_count"] == 1
    assert r3["frequency"]["occurrence_count"] == 2


def test_matched_entry_keys_are_stable():
    """Two calls with the same matched entry should share the same frequency key."""
    _setup()
    r1 = troubleshoot_olive_error("CUDA out of memory")
    r2 = troubleshoot_olive_error("CUDA out of memory tried to allocate")
    # Both match oom-quantization, so counts should accumulate
    assert r1["frequency"]["occurrence_count"] == 1
    assert r2["frequency"]["occurrence_count"] == 2


# ---------------------------------------------------------------------------
# Frequency labels
# ---------------------------------------------------------------------------


def test_label_first_occurrence():
    _setup()
    resp = troubleshoot_olive_error("CUDA out of memory")
    assert resp["frequency"]["label"] == "first_occurrence"


def test_label_recurring():
    _setup()
    for _ in range(2):
        troubleshoot_olive_error("CUDA out of memory")
    resp = troubleshoot_olive_error("CUDA out of memory")
    assert resp["frequency"]["occurrence_count"] == 3
    assert resp["frequency"]["label"] == "recurring"


def test_label_frequent():
    _setup()
    for _ in range(9):
        troubleshoot_olive_error("CUDA out of memory")
    resp = troubleshoot_olive_error("CUDA out of memory")
    assert resp["frequency"]["occurrence_count"] == 10
    assert resp["frequency"]["label"] == "frequent"


def test_label_persistent():
    _setup()
    for _ in range(10):
        troubleshoot_olive_error("CUDA out of memory")
    resp = troubleshoot_olive_error("CUDA out of memory")
    assert resp["frequency"]["occurrence_count"] == 11
    assert resp["frequency"]["label"] == "persistent"


# ---------------------------------------------------------------------------
# Timestamp tracking
# ---------------------------------------------------------------------------


def test_first_seen_does_not_change():
    _setup()
    r1 = troubleshoot_olive_error("CUDA out of memory")
    r2 = troubleshoot_olive_error("CUDA out of memory")
    assert r1["frequency"]["first_seen"] == r2["frequency"]["first_seen"]


def test_last_seen_updates():
    _setup()
    r1 = troubleshoot_olive_error("CUDA out of memory")
    r2 = troubleshoot_olive_error("CUDA out of memory")
    # last_seen should be >= first_seen (same or later)
    assert r2["frequency"]["last_seen"] >= r1["frequency"]["last_seen"]


def test_timestamps_are_iso_format():
    _setup()
    resp = troubleshoot_olive_error("CUDA out of memory")
    ts = resp["frequency"]["first_seen"]
    # Should look like "2026-07-25T12:44:55Z"
    assert "T" in ts
    assert ts.endswith("Z")


# ---------------------------------------------------------------------------
# reset_frequency_store
# ---------------------------------------------------------------------------


def test_reset_clears_all_counts():
    _setup()
    troubleshoot_olive_error("CUDA out of memory")
    troubleshoot_olive_error("CUDA out of memory")
    reset_frequency_store()
    resp = troubleshoot_olive_error("CUDA out of memory")
    assert resp["frequency"]["occurrence_count"] == 1
    assert resp["frequency"]["label"] == "first_occurrence"


# ---------------------------------------------------------------------------
# Unmatched errors with similar messages share a frequency key
# ---------------------------------------------------------------------------


def test_unmatched_errors_with_same_prefix_share_key():
    """Unmatched errors that share the same 80-char prefix should share a frequency key."""
    _setup()
    # Use messages identical in the first 80 chars but differing after
    base = "A" * 80
    r1 = troubleshoot_olive_error(f"{base}---detail1")
    r2 = troubleshoot_olive_error(f"{base}---detail2")
    # Both are unmatched and share the first 80 chars, so counts accumulate
    assert r1["frequency"]["occurrence_count"] == 1
    assert r2["frequency"]["occurrence_count"] == 2


def test_unmatched_errors_with_different_messages_are_independent():
    _setup()
    r1 = troubleshoot_olive_error("Error type Alpha 999")
    r2 = troubleshoot_olive_error("Error type Beta 888")
    assert r1["frequency"]["occurrence_count"] == 1
    assert r2["frequency"]["occurrence_count"] == 1


def test_error_frequency_summary_basic():
    _setup()
    troubleshoot_olive_error("CUDA out of memory")
    troubleshoot_olive_error("CUDA out of memory")
    troubleshoot_olive_error("ONNX export failed")

    summary = get_error_frequency_summary()
    assert summary["total_tracked"] == 2
    assert summary["limit"] == 10
    assert len(summary["entries"]) == 2

    # Most frequent first (matches oom-quantization)
    top = summary["entries"][0]
    assert top["occurrence_count"] == 2
    assert top["label"] == "recurring"
    assert top["matched_entry"] == "oom-quantization"
    assert top["message_prefix"] == ""


def test_error_frequency_summary_respects_limit():
    _setup()
    for i in range(5):
        troubleshoot_olive_error(f"unique error {i}")

    summary = get_error_frequency_summary(limit=3)
    assert summary["total_tracked"] == 5
    assert len(summary["entries"]) == 3


def test_error_frequency_summary_rejects_negative_limit():
    _setup()
    with pytest.raises(ValueError, match="limit must be non-negative"):
        get_error_frequency_summary(limit=-1)
