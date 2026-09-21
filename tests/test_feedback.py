"""Tests for local troubleshoot feedback persistence and ranking bounds.

Covers record_troubleshoot_feedback, atomic store I/O, input validation,
bounded score deltas, temp-path isolation, and privacy (no free-form text).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from olive_mcp_server.tools import feedback as fb
from olive_mcp_server.tools.feedback import (
    ALLOWED_RATINGS,
    ALLOWED_REASON_CODES,
    FEEDBACK_BOOST_PER_NET,
    FEEDBACK_MAX_ADJUSTMENT,
    FEEDBACK_NET_VOTE_CAP,
    MAX_COUNT_PER_RATING,
    feedback_score_delta,
    get_all_feedback_aggregates,
    get_entry_feedback_counts,
    get_feedback_path,
    record_troubleshoot_feedback,
    reset_feedback_store,
    set_feedback_path,
)

# Stable KB entry ids used across positive paths.
_ENTRY_A = "oom-quantization"
_ENTRY_B = "ep-fallback-cpu"

# Keys that must never appear in the on-disk store (privacy / free-form ban).
_FORBIDDEN_STORE_KEYS = frozenset(
    {
        "log",
        "logs",
        "message",
        "error_message",
        "traceback",
        "stack",
        "note",
        "notes",
        "comment",
        "comments",
        "text",
        "free_text",
        "description",
        "raw",
        "payload",
    }
)


@pytest.fixture(autouse=True)
def _isolated_feedback_store(tmp_path: Path):
    """Point every test at a fresh temp file; clear override afterward."""
    path = tmp_path / "troubleshoot_feedback.json"
    set_feedback_path(path)
    reset_feedback_store()
    yield path
    reset_feedback_store()
    set_feedback_path(None)


def _read_store(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _assert_store_privacy(data: dict) -> None:
    """Recursively assert store JSON has only aggregate counters, no free text."""
    assert isinstance(data, dict)
    assert set(data.keys()) <= {"version", "entries"}
    assert data.get("version") == 1
    entries = data.get("entries", {})
    assert isinstance(entries, dict)
    for eid, entry in entries.items():
        assert isinstance(eid, str) and eid
        assert isinstance(entry, dict)
        assert set(entry.keys()) <= {"thumbs_up", "thumbs_down", "reason_codes"}
        assert isinstance(entry.get("thumbs_up", 0), int)
        assert isinstance(entry.get("thumbs_down", 0), int)
        for bad in _FORBIDDEN_STORE_KEYS:
            assert bad not in entry
        reasons = entry.get("reason_codes")
        if reasons is not None:
            assert isinstance(reasons, dict)
            for code, count in reasons.items():
                assert code in ALLOWED_REASON_CODES
                assert isinstance(count, int)
                assert count > 0


# ---------------------------------------------------------------------------
# Atomic local persistence
# ---------------------------------------------------------------------------


def test_record_persists_atomically_to_disk(tmp_path: Path):
    """Positive: successful rating writes a valid JSON store via atomic replace."""
    # Arrange
    path = get_feedback_path()
    assert path.parent == tmp_path

    # Act
    result = record_troubleshoot_feedback(_ENTRY_A, "thumbs-up", reason_code="accurate")

    # Assert
    assert result["status"] == "ok"
    assert result["matched_entry"] == _ENTRY_A
    assert result["thumbs_up"] == 1
    assert result["thumbs_down"] == 0
    assert result["total"] == 1
    assert path.is_file()
    data = _read_store(path)
    _assert_store_privacy(data)
    entry = data["entries"][_ENTRY_A]
    assert entry["thumbs_up"] == 1
    assert entry["thumbs_down"] == 0
    assert entry["reason_codes"]["accurate"] == 1
    # No leftover temp files from atomic write
    leftovers = list(path.parent.glob(".troubleshoot_feedback-*.tmp"))
    assert leftovers == []


def test_record_increments_existing_entry():
    """Positive: repeated ratings accumulate on the same entry."""
    # Arrange / Act
    r1 = record_troubleshoot_feedback(_ENTRY_A, "thumbs-up")
    r2 = record_troubleshoot_feedback(_ENTRY_A, "thumbs-down")
    r3 = record_troubleshoot_feedback(_ENTRY_A, "thumbs-up")

    # Assert
    assert r1["thumbs_up"] == 1 and r1["thumbs_down"] == 0
    assert r2["thumbs_up"] == 1 and r2["thumbs_down"] == 1
    assert r3["thumbs_up"] == 2 and r3["thumbs_down"] == 1
    counts = get_entry_feedback_counts(_ENTRY_A)
    assert counts == {"thumbs_up": 2, "thumbs_down": 1}


def test_corrupt_store_treated_as_empty(tmp_path: Path):
    """Negative: unreadable/corrupt JSON does not crash; starts fresh."""
    # Arrange
    path = get_feedback_path()
    path.write_text("{not-valid-json!!!", encoding="utf-8")

    # Act
    result = record_troubleshoot_feedback(_ENTRY_A, "thumbs-up")

    # Assert
    assert result["status"] == "ok"
    assert result["thumbs_up"] == 1
    data = _read_store(path)
    assert data["entries"][_ENTRY_A]["thumbs_up"] == 1


# ---------------------------------------------------------------------------
# Invalid input rejection
# ---------------------------------------------------------------------------


def test_valid_rating_and_reason_accepted():
    """Positive: allowlisted rating + reason_code succeed."""
    # Arrange / Act
    result = record_troubleshoot_feedback(_ENTRY_B, "thumbs-down", reason_code="wrong_match")

    # Assert
    assert result["status"] == "ok"
    assert result["rating"] == "thumbs-down"
    assert result["reason_code"] == "wrong_match"
    assert result["thumbs_down"] == 1
    assert result["score_delta"] == pytest.approx(-FEEDBACK_BOOST_PER_NET)


@pytest.mark.parametrize(
    "matched_entry,rating,reason_code,error_code",
    [
        ("", "thumbs-up", None, "invalid_matched_entry"),
        ("   ", "thumbs-up", None, "invalid_matched_entry"),
        ("not-a-real-kb-entry-xyz", "thumbs-up", None, "unknown_matched_entry"),
        (_ENTRY_A, "sideways", None, "invalid_rating"),
        (_ENTRY_A, "thumbs_up", None, "invalid_rating"),  # underscore form rejected
        (_ENTRY_A, "", None, "invalid_rating"),
        (_ENTRY_A, "thumbs-up", "totally-made-up", "invalid_reason_code"),
        (_ENTRY_A, "thumbs-up", "user wrote free text", "invalid_reason_code"),
    ],
)
def test_invalid_inputs_rejected(matched_entry, rating, reason_code, error_code):
    """Negative: bad entry id, rating, or free-form reason_code are rejected."""
    # Arrange / Act
    result = record_troubleshoot_feedback(matched_entry, rating, reason_code=reason_code)

    # Assert
    assert result["status"] == "error"
    assert result["error"] == error_code
    # Nothing persisted on validation failure
    path = get_feedback_path()
    assert not path.is_file() or _read_store(path).get("entries", {}) == {}


def test_invalid_rating_lists_allowed_values():
    """Negative: invalid_rating response includes allowlist for clients."""
    # Arrange / Act
    result = record_troubleshoot_feedback(_ENTRY_A, "meh")

    # Assert
    assert result["status"] == "error"
    assert result["error"] == "invalid_rating"
    assert set(result["allowed_ratings"]) == set(ALLOWED_RATINGS)


def test_invalid_reason_lists_allowed_codes():
    """Negative: invalid_reason_code response includes allowlist."""
    # Arrange / Act
    result = record_troubleshoot_feedback(_ENTRY_A, "thumbs-up", reason_code="please fix this bug for me")

    # Assert
    assert result["status"] == "error"
    assert result["error"] == "invalid_reason_code"
    assert set(result["allowed_reason_codes"]) == set(ALLOWED_REASON_CODES)


# ---------------------------------------------------------------------------
# Bounded score effects
# ---------------------------------------------------------------------------


def test_score_delta_scales_with_net_votes():
    """Positive: net thumbs produce expected small boost/demote."""
    # Arrange — 3 up, 1 down → net +2 → +0.02
    for _ in range(3):
        record_troubleshoot_feedback(_ENTRY_A, "thumbs-up")
    record_troubleshoot_feedback(_ENTRY_A, "thumbs-down")

    # Act
    delta = feedback_score_delta(_ENTRY_A)

    # Assert
    assert delta == pytest.approx(2 * FEEDBACK_BOOST_PER_NET)
    assert abs(delta) <= FEEDBACK_MAX_ADJUSTMENT


def test_score_delta_capped_at_max_adjustment():
    """Negative: many net votes cannot exceed FEEDBACK_MAX_ADJUSTMENT."""
    # Arrange — far beyond FEEDBACK_NET_VOTE_CAP
    for _ in range(FEEDBACK_NET_VOTE_CAP + 20):
        record_troubleshoot_feedback(_ENTRY_A, "thumbs-up")

    # Act
    delta = feedback_score_delta(_ENTRY_A)
    result = record_troubleshoot_feedback(_ENTRY_A, "thumbs-up")

    # Assert
    assert delta == pytest.approx(FEEDBACK_MAX_ADJUSTMENT)
    assert abs(delta) <= FEEDBACK_MAX_ADJUSTMENT
    assert result["score_delta"] == pytest.approx(FEEDBACK_MAX_ADJUSTMENT)
    assert result["max_score_adjustment"] == FEEDBACK_MAX_ADJUSTMENT


def test_score_delta_negative_cap():
    """Negative: heavy thumbs-down is clamped to -FEEDBACK_MAX_ADJUSTMENT."""
    # Arrange
    for _ in range(FEEDBACK_NET_VOTE_CAP + 10):
        record_troubleshoot_feedback(_ENTRY_A, "thumbs-down")

    # Act
    delta = feedback_score_delta(_ENTRY_A)

    # Assert
    assert delta == pytest.approx(-FEEDBACK_MAX_ADJUSTMENT)
    assert delta >= -FEEDBACK_MAX_ADJUSTMENT


def test_score_delta_zero_for_unknown_or_empty_entry():
    """Negative: missing/blank entry ids yield zero delta (no invent)."""
    # Arrange / Act / Assert
    assert feedback_score_delta("never-recorded-entry") == 0.0
    assert feedback_score_delta("") == 0.0
    assert get_entry_feedback_counts("") == {"thumbs_up": 0, "thumbs_down": 0}


def test_count_cap_stops_incrementing():
    """Negative: per-rating counts stop at MAX_COUNT_PER_RATING."""
    # Arrange — seed store at cap without looping 1000 times
    path = get_feedback_path()
    store = {
        "version": 1,
        "entries": {
            _ENTRY_A: {
                "thumbs_up": MAX_COUNT_PER_RATING,
                "thumbs_down": 0,
            }
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(store), encoding="utf-8")

    # Act
    result = record_troubleshoot_feedback(_ENTRY_A, "thumbs-up")

    # Assert — still ok acknowledgement, count stays capped
    assert result["status"] == "ok"
    assert result["thumbs_up"] == MAX_COUNT_PER_RATING
    assert get_entry_feedback_counts(_ENTRY_A)["thumbs_up"] == MAX_COUNT_PER_RATING


def test_delta_from_counts_pure_bounds():
    """Pure helper: net vote cap and max adjustment both enforced."""
    # Arrange / Act / Assert
    assert fb._delta_from_counts(0, 0) == 0.0
    assert fb._delta_from_counts(1, 0) == pytest.approx(FEEDBACK_BOOST_PER_NET)
    assert fb._delta_from_counts(0, 1) == pytest.approx(-FEEDBACK_BOOST_PER_NET)
    # Net beyond cap still clamps
    assert fb._delta_from_counts(100, 0) == pytest.approx(FEEDBACK_MAX_ADJUSTMENT)
    assert fb._delta_from_counts(0, 100) == pytest.approx(-FEEDBACK_MAX_ADJUSTMENT)
    # Exactly at net cap
    assert fb._delta_from_counts(FEEDBACK_NET_VOTE_CAP, 0) == pytest.approx(
        min(FEEDBACK_NET_VOTE_CAP * FEEDBACK_BOOST_PER_NET, FEEDBACK_MAX_ADJUSTMENT)
    )


# ---------------------------------------------------------------------------
# Reset / isolation via temp path
# ---------------------------------------------------------------------------


def test_reset_clears_store_file():
    """Positive: reset_feedback_store deletes the resolved file."""
    # Arrange
    record_troubleshoot_feedback(_ENTRY_A, "thumbs-up")
    path = get_feedback_path()
    assert path.is_file()

    # Act
    reset_feedback_store()

    # Assert
    assert not path.is_file()
    assert get_entry_feedback_counts(_ENTRY_A) == {"thumbs_up": 0, "thumbs_down": 0}
    assert get_all_feedback_aggregates() == {}


def test_reset_safe_when_file_missing():
    """Negative: reset on missing file is a no-op (no raise)."""
    # Arrange
    path = get_feedback_path()
    if path.is_file():
        path.unlink()

    # Act / Assert — must not raise
    reset_feedback_store()
    assert not path.is_file()


def test_path_isolation_between_temp_dirs(tmp_path: Path):
    """Positive: distinct set_feedback_path targets do not share data."""
    # Arrange
    path_a = tmp_path / "a" / "feedback.json"
    path_b = tmp_path / "b" / "feedback.json"

    # Act — write to A
    set_feedback_path(path_a)
    record_troubleshoot_feedback(_ENTRY_A, "thumbs-up")
    assert get_entry_feedback_counts(_ENTRY_A)["thumbs_up"] == 1

    # Switch to B — empty
    set_feedback_path(path_b)
    assert get_entry_feedback_counts(_ENTRY_A)["thumbs_up"] == 0
    record_troubleshoot_feedback(_ENTRY_B, "thumbs-down")
    assert get_entry_feedback_counts(_ENTRY_B)["thumbs_down"] == 1

    # Back to A — original data intact, B entry absent
    set_feedback_path(path_a)
    assert get_entry_feedback_counts(_ENTRY_A)["thumbs_up"] == 1
    assert get_entry_feedback_counts(_ENTRY_B)["thumbs_down"] == 0

    # Cleanup for autouse fixture
    set_feedback_path(tmp_path / "troubleshoot_feedback.json")


def test_env_path_used_when_no_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Positive: OLIVE_MCP_FEEDBACK_PATH resolves when override is cleared."""
    # Arrange
    env_file = tmp_path / "from-env.json"
    monkeypatch.setenv("OLIVE_MCP_FEEDBACK_PATH", str(env_file))
    set_feedback_path(None)

    # Act
    assert get_feedback_path() == env_file
    result = record_troubleshoot_feedback(_ENTRY_A, "thumbs-up")

    # Assert
    assert result["status"] == "ok"
    assert env_file.is_file()
    _assert_store_privacy(_read_store(env_file))

    # Restore isolation for remaining fixture teardown
    set_feedback_path(tmp_path / "troubleshoot_feedback.json")
    monkeypatch.delenv("OLIVE_MCP_FEEDBACK_PATH", raising=False)


# ---------------------------------------------------------------------------
# Privacy: feedback file has no logs / free-form text
# ---------------------------------------------------------------------------


def test_store_contains_only_aggregates_no_free_form():
    """Positive: on-disk JSON is version + capped counters only."""
    # Arrange / Act
    record_troubleshoot_feedback(_ENTRY_A, "thumbs-up", reason_code="fixed_issue")
    record_troubleshoot_feedback(_ENTRY_A, "thumbs-down", reason_code="outdated")
    record_troubleshoot_feedback(_ENTRY_B, "thumbs-up")

    # Assert
    data = _read_store(get_feedback_path())
    _assert_store_privacy(data)
    raw = get_feedback_path().read_text(encoding="utf-8")
    # No log-like or free-form content leaked into the file body
    for needle in (
        "traceback",
        "Traceback",
        "error_message",
        "CUDA out of memory",
        "please fix",
        "user note",
    ):
        assert needle not in raw


def test_sanitize_strips_unexpected_free_form_fields(tmp_path: Path):
    """Negative: free-form / log fields on disk are dropped on load."""
    # Arrange — poison the store with fields that must never be kept
    path = get_feedback_path()
    poisoned = {
        "version": 1,
        "entries": {
            _ENTRY_A: {
                "thumbs_up": 2,
                "thumbs_down": 1,
                "note": "user free-form comment that must not persist",
                "error_message": "CUDA OOM full traceback here",
                "logs": ["line1", "line2"],
                "reason_codes": {
                    "accurate": 1,
                    "not-allowlisted-reason": 99,
                    "outdated": 2,
                },
            }
        },
        "extra_top_level": "should vanish on rewrite",
    }
    path.write_text(json.dumps(poisoned), encoding="utf-8")

    # Act — load via public API then rewrite via a new rating
    counts = get_entry_feedback_counts(_ENTRY_A)
    assert counts == {"thumbs_up": 2, "thumbs_down": 1}
    record_troubleshoot_feedback(_ENTRY_A, "thumbs-up")

    # Assert — rewritten file is clean
    data = _read_store(path)
    _assert_store_privacy(data)
    entry = data["entries"][_ENTRY_A]
    assert entry["thumbs_up"] == 3
    assert entry["thumbs_down"] == 1
    assert "note" not in entry
    assert "error_message" not in entry
    assert "logs" not in entry
    assert set(entry["reason_codes"].keys()) <= set(ALLOWED_REASON_CODES)
    assert "not-allowlisted-reason" not in entry["reason_codes"]
    assert "extra_top_level" not in data


def test_ok_response_is_aggregate_only():
    """Positive: tool acknowledgement has no paths, logs, or free-form notes."""
    # Arrange / Act
    result = record_troubleshoot_feedback(_ENTRY_A, "thumbs-up", reason_code="clear_fix")

    # Assert
    assert result["status"] == "ok"
    allowed_keys = {
        "status",
        "matched_entry",
        "rating",
        "reason_code",
        "thumbs_up",
        "thumbs_down",
        "total",
        "count_cap",
        "score_delta",
        "max_score_adjustment",
    }
    assert set(result.keys()) <= allowed_keys
    for bad in _FORBIDDEN_STORE_KEYS | {"path", "file", "store_path"}:
        assert bad not in result


def test_api_does_not_accept_free_form_kwargs():
    """Negative: signature rejects unexpected free-form parameters."""
    # Arrange / Act / Assert
    with pytest.raises(TypeError):
        record_troubleshoot_feedback(  # type: ignore[call-arg]
            _ENTRY_A,
            "thumbs-up",
            note="this free-form note must not be accepted",
        )
    with pytest.raises(TypeError):
        record_troubleshoot_feedback(  # type: ignore[call-arg]
            _ENTRY_A,
            "thumbs-up",
            error_message="CUDA OOM traceback",
        )


# ---------------------------------------------------------------------------
# Inter-process lock
# ---------------------------------------------------------------------------


def _worker_increment(store_path: str, entry_id: str, n: int) -> None:
    """Child-process helper: n thumbs-up increments against a shared store path."""
    import os

    os.environ["OLIVE_MCP_FEEDBACK_PATH"] = store_path
    from olive_mcp_server.tools import feedback as feedback_mod

    feedback_mod.set_feedback_path(None)  # force env path
    from olive_mcp_server.tools.feedback import record_troubleshoot_feedback as record

    for _ in range(n):
        result = record(entry_id, "thumbs-up")
        assert result["status"] == "ok", result


def test_concurrent_processes_preserve_increments(tmp_path: Path, monkeypatch):
    """Overlapping proxy-style processes must not drop acknowledged votes."""
    import multiprocessing as mp

    shared = tmp_path / "worker_feedback.json"
    monkeypatch.setenv("OLIVE_MCP_FEEDBACK_PATH", str(shared))
    set_feedback_path(None)
    reset_feedback_store()

    seed = record_troubleshoot_feedback(_ENTRY_A, "thumbs-up")
    assert seed["status"] == "ok"
    assert seed["thumbs_up"] == 1

    procs = [mp.Process(target=_worker_increment, args=(str(shared), _ENTRY_A, 5)) for _ in range(4)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=60)
        assert p.exitcode == 0, f"worker failed with {p.exitcode}"

    data = _read_store(shared)
    # 1 seed + 4 workers * 5 increments
    assert data["entries"][_ENTRY_A]["thumbs_up"] == 21
