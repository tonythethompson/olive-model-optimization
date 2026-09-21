"""Tool: record_troubleshoot_feedback.

Local-only, aggregate feedback for troubleshooting KB matches.
Stores thumbs-up/down counts (and optional bounded reason codes) keyed by
matched-entry ID. Never persists log text, tracebacks, or free-form content.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any, Final, Literal

from . import load_studio_troubleshooting, load_troubleshooting

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public constants (ratings, reason codes, caps)
# ---------------------------------------------------------------------------

ALLOWED_RATINGS: Final[frozenset[str]] = frozenset({"thumbs-up", "thumbs-down"})
RatingName = Literal["thumbs-up", "thumbs-down"]

# Bounded reason codes only — never free text.
ALLOWED_REASON_CODES: Final[frozenset[str]] = frozenset(
    {
        "accurate",
        "clear_fix",
        "fixed_issue",
        "wrong_match",
        "outdated",
        "incomplete",
        "incorrect_fix",
    }
)

# Store caps: limit disk growth and ranking influence inputs.
MAX_TRACKED_ENTRIES: Final[int] = 512
MAX_COUNT_PER_RATING: Final[int] = 1000
MAX_REASON_COUNT: Final[int] = 1000

# Ranking influence caps (consumed by troubleshooting hybrid scorer).
# Net votes beyond FEEDBACK_NET_VOTE_CAP do not increase the delta further.
FEEDBACK_BOOST_PER_NET: Final[float] = 0.01
FEEDBACK_MAX_ADJUSTMENT: Final[float] = 0.05
FEEDBACK_NET_VOTE_CAP: Final[int] = 5

_STORE_VERSION: Final[int] = 1
_ENV_FEEDBACK_PATH: Final[str] = "OLIVE_MCP_FEEDBACK_PATH"
_DEFAULT_FILENAME: Final[str] = "troubleshoot_feedback.json"

# Internal JSON keys (snake_case; API ratings use hyphens).
_KEY_UP: Final[str] = "thumbs_up"
_KEY_DOWN: Final[str] = "thumbs_down"
_KEY_REASONS: Final[str] = "reason_codes"

# ---------------------------------------------------------------------------
# Path override + in-process + inter-process locks
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_path_override: Path | None = None


@contextmanager
def _interprocess_store_lock(path: Path) -> Iterator[None]:
    """Exclusive lock across processes for feedback read-modify-write.

    The HTTP MCP proxy spawns a fresh Python process per request, so a
    threading.Lock alone cannot serialize overlapping increments. Lock a
    sibling ``*.lock`` file with ``fcntl`` (POSIX) or ``msvcrt`` (Windows).
    """
    lock_path = path.with_name(path.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o644)
    try:
        if os.name == "nt":
            import msvcrt

            # LK_LOCK blocks until the region is free; retry on transient errors.
            while True:
                try:
                    msvcrt.locking(fd, msvcrt.LK_LOCK, 1)
                    break
                except OSError:
                    time.sleep(0.01)
        else:
            import fcntl

            fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        try:
            if os.name == "nt":
                import msvcrt

                try:
                    os.lseek(fd, 0, os.SEEK_SET)
                    msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
                except OSError:
                    pass
            else:
                import fcntl

                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


@contextmanager
def _store_lock(path: Path) -> Iterator[None]:
    """Hold both the cross-process file lock and the in-process threading lock."""
    with _interprocess_store_lock(path), _lock:
        yield


def set_feedback_path(path: str | Path | None) -> None:
    """Override the feedback file path (tests). Pass ``None`` to clear."""
    global _path_override
    with _lock:
        _path_override = Path(path) if path is not None else None


def get_feedback_path() -> Path:
    """Resolve the feedback store path (override → env → XDG user data)."""
    if _path_override is not None:
        return _path_override

    env_path = os.environ.get(_ENV_FEEDBACK_PATH, "").strip()
    if env_path:
        return Path(env_path)

    xdg = os.environ.get("XDG_DATA_HOME", "").strip()
    base = Path(xdg) if xdg else Path.home() / ".local" / "share"
    return base / "olive-mcp" / _DEFAULT_FILENAME


def reset_feedback_store() -> None:
    """Clear the on-disk feedback store (tests).

    Deletes the resolved feedback file when present. Safe if the file is
    missing. Does not clear ``set_feedback_path`` / env overrides.
    """
    path = get_feedback_path()
    with _store_lock(path):
        try:
            if path.is_file():
                path.unlink()
        except OSError as exc:
            logger.warning("Failed to remove feedback store %s: %s", path, exc)


# ---------------------------------------------------------------------------
# Known entry IDs
# ---------------------------------------------------------------------------


def _known_entry_ids() -> set[str]:
    """Return the set of valid troubleshooting KB entry IDs."""
    ids: set[str] = set()
    for entry in load_troubleshooting():
        eid = entry.get("id")
        if isinstance(eid, str) and eid:
            ids.add(eid)
    for entry in load_studio_troubleshooting():
        eid = entry.get("id")
        if isinstance(eid, str) and eid:
            ids.add(eid)
    return ids


# ---------------------------------------------------------------------------
# Store load / atomic write
# ---------------------------------------------------------------------------


def _empty_store() -> dict[str, Any]:
    return {"version": _STORE_VERSION, "entries": {}}


def _sanitize_entry(raw: Any) -> dict[str, Any] | None:
    """Keep only aggregate counters; drop any unexpected free-form fields."""
    if not isinstance(raw, dict):
        return None
    up = raw.get(_KEY_UP, 0)
    down = raw.get(_KEY_DOWN, 0)
    if not isinstance(up, int) or not isinstance(down, int):
        # Coerce numeric-looking values; reject everything else.
        try:
            up = int(up)
            down = int(down)
        except (TypeError, ValueError):
            return None
    up = max(0, min(up, MAX_COUNT_PER_RATING))
    down = max(0, min(down, MAX_COUNT_PER_RATING))

    reasons_out: dict[str, int] = {}
    reasons_raw = raw.get(_KEY_REASONS)
    if isinstance(reasons_raw, dict):
        for code, count in reasons_raw.items():
            if code not in ALLOWED_REASON_CODES:
                continue
            try:
                n = int(count)
            except (TypeError, ValueError):
                continue
            if n > 0:
                reasons_out[code] = min(n, MAX_REASON_COUNT)

    out: dict[str, Any] = {_KEY_UP: up, _KEY_DOWN: down}
    if reasons_out:
        out[_KEY_REASONS] = reasons_out
    return out


def _load_store_unlocked(path: Path) -> dict[str, Any]:
    """Load and sanitize the store from disk. Caller must hold ``_lock``."""
    if not path.is_file():
        return _empty_store()
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Corrupt or unreadable feedback store %s: %s", path, exc)
        return _empty_store()

    if not isinstance(data, dict):
        return _empty_store()

    entries_raw = data.get("entries")
    if not isinstance(entries_raw, dict):
        return _empty_store()

    entries: dict[str, Any] = {}
    for key, value in entries_raw.items():
        if not isinstance(key, str) or not key:
            continue
        cleaned = _sanitize_entry(value)
        if cleaned is not None:
            entries[key] = cleaned
        if len(entries) >= MAX_TRACKED_ENTRIES:
            break

    return {"version": _STORE_VERSION, "entries": entries}


def _atomic_write_unlocked(path: Path, store: dict[str, Any]) -> None:
    """Atomically write JSON store via temp file + ``os.replace``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # Restrict payload to version + capped entry aggregates only.
    payload = {
        "version": _STORE_VERSION,
        "entries": store.get("entries", {}),
    }
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent),
        prefix=".troubleshoot_feedback-",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    except Exception:
        with suppress(OSError):
            os.unlink(tmp_name)
        raise


# ---------------------------------------------------------------------------
# Aggregate helpers (for ranking / tests)
# ---------------------------------------------------------------------------


def get_entry_feedback_counts(matched_entry: str) -> dict[str, int]:
    """Return capped thumbs counts for one entry (zeros if unknown/absent)."""
    if not isinstance(matched_entry, str) or not matched_entry:
        return {_KEY_UP: 0, _KEY_DOWN: 0}
    path = get_feedback_path()
    with _store_lock(path):
        store = _load_store_unlocked(path)
        entry = store["entries"].get(matched_entry) or {}
        return {
            _KEY_UP: int(entry.get(_KEY_UP, 0)),
            _KEY_DOWN: int(entry.get(_KEY_DOWN, 0)),
        }


def get_all_feedback_aggregates() -> dict[str, dict[str, int]]:
    """Return ``{entry_id: {thumbs_up, thumbs_down}}`` for all tracked entries."""
    path = get_feedback_path()
    with _store_lock(path):
        store = _load_store_unlocked(path)
        result: dict[str, dict[str, int]] = {}
        for eid, entry in store["entries"].items():
            result[eid] = {
                _KEY_UP: int(entry.get(_KEY_UP, 0)),
                _KEY_DOWN: int(entry.get(_KEY_DOWN, 0)),
            }
        return result


def _delta_from_counts(thumbs_up: int, thumbs_down: int) -> float:
    """Pure bounded score delta from aggregate counts."""
    net = int(thumbs_up) - int(thumbs_down)
    if net > FEEDBACK_NET_VOTE_CAP:
        net = FEEDBACK_NET_VOTE_CAP
    elif net < -FEEDBACK_NET_VOTE_CAP:
        net = -FEEDBACK_NET_VOTE_CAP
    delta = net * FEEDBACK_BOOST_PER_NET
    if delta > FEEDBACK_MAX_ADJUSTMENT:
        return FEEDBACK_MAX_ADJUSTMENT
    if delta < -FEEDBACK_MAX_ADJUSTMENT:
        return -FEEDBACK_MAX_ADJUSTMENT
    return float(delta)


def feedback_score_delta(matched_entry: str) -> float:
    """Bounded ranking adjustment from aggregate feedback.

    Positive net thumbs → slight boost; negative → slight demote.
    Absolute value is capped at ``FEEDBACK_MAX_ADJUSTMENT``.
    """
    counts = get_entry_feedback_counts(matched_entry)
    return _delta_from_counts(counts[_KEY_UP], counts[_KEY_DOWN])


# ---------------------------------------------------------------------------
# MCP tool
# ---------------------------------------------------------------------------


def record_troubleshoot_feedback(
    matched_entry: str,
    rating: str,
    reason_code: str | None = None,
) -> dict[str, Any]:
    """Record local aggregate feedback for a troubleshooting KB match.

    Privacy: only ``matched_entry``, ``rating``, and an optional allowlisted
    ``reason_code`` are accepted. Log text, tracebacks, and free-form notes
    are rejected by the signature and never written to disk.

    Args:
        matched_entry: Troubleshooting KB entry id (must exist in olive or
            studio troubleshooting knowledge bases).
        rating: ``"thumbs-up"`` or ``"thumbs-down"`` only.
        reason_code: Optional allowlisted reason code (not free text).

    Returns:
        Aggregate acknowledgement with status and capped counts, or a
        structured error payload when validation fails.
    """
    # --- validate matched_entry ---
    if not isinstance(matched_entry, str) or not matched_entry.strip():
        return {
            "status": "error",
            "error": "invalid_matched_entry",
            "message": "matched_entry must be a non-empty string entry id.",
        }
    matched_entry = matched_entry.strip()

    known = _known_entry_ids()
    if matched_entry not in known:
        return {
            "status": "error",
            "error": "unknown_matched_entry",
            "message": f"Unknown matched_entry id: {matched_entry!r}.",
        }

    # --- validate rating ---
    if not isinstance(rating, str) or rating not in ALLOWED_RATINGS:
        return {
            "status": "error",
            "error": "invalid_rating",
            "message": "rating must be 'thumbs-up' or 'thumbs-down'.",
            "allowed_ratings": sorted(ALLOWED_RATINGS),
        }

    # --- validate optional reason_code (allowlist only) ---
    normalized_reason: str | None = None
    if reason_code is not None:
        if not isinstance(reason_code, str) or reason_code not in ALLOWED_REASON_CODES:
            return {
                "status": "error",
                "error": "invalid_reason_code",
                "message": "reason_code must be one of the allowlisted values or omitted.",
                "allowed_reason_codes": sorted(ALLOWED_REASON_CODES),
            }
        normalized_reason = reason_code

    path = get_feedback_path()
    with _store_lock(path):
        store = _load_store_unlocked(path)
        entries: dict[str, Any] = store["entries"]

        if matched_entry not in entries:
            if len(entries) >= MAX_TRACKED_ENTRIES:
                return {
                    "status": "error",
                    "error": "entry_cap_reached",
                    "message": (
                        f"Feedback store already tracks {MAX_TRACKED_ENTRIES} entries; cannot add a new matched_entry."
                    ),
                    "max_tracked_entries": MAX_TRACKED_ENTRIES,
                }
            entries[matched_entry] = {_KEY_UP: 0, _KEY_DOWN: 0}

        entry = entries[matched_entry]
        count_key = _KEY_UP if rating == "thumbs-up" else _KEY_DOWN
        current = int(entry.get(count_key, 0))
        if current < MAX_COUNT_PER_RATING:
            entry[count_key] = current + 1
        # else: silently stay at cap (still acknowledge current aggregates)

        if normalized_reason is not None:
            reasons = entry.setdefault(_KEY_REASONS, {})
            if not isinstance(reasons, dict):
                reasons = {}
                entry[_KEY_REASONS] = reasons
            rc = int(reasons.get(normalized_reason, 0))
            if rc < MAX_REASON_COUNT:
                reasons[normalized_reason] = rc + 1

        entries[matched_entry] = entry
        store["entries"] = entries

        try:
            _atomic_write_unlocked(path, store)
        except OSError as exc:
            logger.warning("Failed to persist feedback store %s: %s", path, exc)
            return {
                "status": "error",
                "error": "persist_failed",
                "message": "Could not write local feedback store.",
            }

        up = int(entry.get(_KEY_UP, 0))
        down = int(entry.get(_KEY_DOWN, 0))
        delta = _delta_from_counts(up, down)

    # Aggregate acknowledgement only — no logs, paths, or free-form content.
    return {
        "status": "ok",
        "matched_entry": matched_entry,
        "rating": rating,
        "reason_code": normalized_reason,
        "thumbs_up": up,
        "thumbs_down": down,
        "total": up + down,
        "count_cap": MAX_COUNT_PER_RATING,
        "score_delta": delta,
        "max_score_adjustment": FEEDBACK_MAX_ADJUSTMENT,
    }
