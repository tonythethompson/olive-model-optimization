"""Phase 3: Multi-job comparison with preference-weighted scoring.

Fetches results for multiple completed optimization jobs, normalizes metrics,
applies preference weighting, and selects a winner. Non-terminal, failed, or
unfetchable jobs are excluded with reasons.

Tools:
  - compare_results
"""

from __future__ import annotations

import logging
import re
from typing import Any

from .studio_loopback import err, studio_request

logger = logging.getLogger(__name__)

_STATUS_PATH = "/api/olive/agent/status"  # + /{job_id}

_VALID_PREFERENCES = frozenset({"latency", "size", "accuracy", "balanced"})
_JOB_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,128}$")

_MIN_JOBS = 2
_MAX_JOBS = 10
_METRIC_KEYS = ("latency_ms", "model_size_mb", "accuracy")
_LOWER_IS_BETTER = frozenset({"latency_ms", "model_size_mb"})


def _is_error(payload: dict[str, Any]) -> bool:
    """Check if a bridge response is an error (has non-empty 'error' key)."""
    err_val = payload.get("error")
    return isinstance(err_val, str) and bool(err_val)


def _extract_metrics(job_response: dict[str, Any]) -> dict[str, float | None]:
    """Extract scoring metrics from a job status response.

    Looks for latency_ms, model_size_mb, and accuracy in the latestMetrics
    field (or latest_metrics fallback).
    """
    raw = job_response.get("latestMetrics") or job_response.get("latest_metrics") or {}
    if not isinstance(raw, dict):
        raw = {}

    def _as_float(val: Any) -> float | None:
        if val is None:
            return None
        try:
            f = float(val)
            # Reject NaN/inf as non-scoreable
            if f != f or f == float("inf") or f == float("-inf"):
                return None
            return f
        except (TypeError, ValueError):
            return None

    return {
        "latency_ms": _as_float(raw.get("latency_ms")),
        "model_size_mb": _as_float(raw.get("model_size_mb")),
        "accuracy": _as_float(raw.get("accuracy")),
    }


def _metric_bounds(
    scoreable: list[dict[str, Any]],
) -> dict[str, tuple[float, float]]:
    """Return observed min/max bounds for each populated metric."""
    values: dict[str, list[float]] = {key: [] for key in _METRIC_KEYS}
    for entry in scoreable:
        for key in _METRIC_KEYS:
            value = entry["metrics"].get(key)
            if value is not None:
                values[key].append(value)
    return {key: (min(metric_values), max(metric_values)) for key, metric_values in values.items() if metric_values}


def _metric_weights(preference: str) -> dict[str, float]:
    """Return scoring weights for the selected preference."""
    preferred_key = {
        "latency": "latency_ms",
        "size": "model_size_mb",
        "accuracy": "accuracy",
    }.get(preference)
    return {key: 2.0 if key == preferred_key else 1.0 for key in _METRIC_KEYS}


def _normalized_metric_score(key: str, value: float, bounds: tuple[float, float]) -> float:
    """Normalize one metric to a higher-is-better score in [0, 1].

    Degenerate ranges (maximum == minimum) return the neutral midpoint 0.5
    regardless of direction, since inversion of a constant is still a constant
    but the neutral score makes the tie explicit in weighted averages.
    """
    minimum, maximum = bounds
    if maximum == minimum:
        return 0.5
    normalized = (value - minimum) / (maximum - minimum)
    return 1.0 - normalized if key in _LOWER_IS_BETTER else normalized


def _score_entry(
    entry: dict[str, Any],
    bounds: dict[str, tuple[float, float]],
    weights: dict[str, float],
) -> dict[str, Any]:
    """Return one comparison entry with its weighted normalized score.

    Missing metrics receive a penalty score of 0.0 (worst) and still
    contribute their full weight to ``total_weight``. This ensures every
    compared job is scored on the same metric set: a job missing a metric
    cannot inflate its average by dropping that metric's weight.
    """
    weighted_sum = 0.0
    total_weight = 0.0
    for key in _METRIC_KEYS:
        weight = weights[key]
        total_weight += weight
        value = entry["metrics"].get(key)
        if value is None or key not in bounds:
            # Penalty: missing metric contributes 0.0 to the weighted sum
            # but still consumes its full weight, pulling the average down.
            continue
        weighted_sum += _normalized_metric_score(key, value, bounds[key]) * weight

    score = weighted_sum / total_weight if total_weight else 0.0
    return {
        "job_id": entry["job_id"],
        "status": entry["status"],
        "metrics": entry["metrics"],
        "score": round(score, 6),
    }


def _normalize_and_score(
    scoreable: list[dict[str, Any]],
    preference: str,
) -> list[dict[str, Any]]:
    """Min-max normalize metrics and compute weighted scores.

    For latency and size: lower is better → invert (1 - normalized).
    For accuracy: higher is better → use normalized directly.
    Preference weighting: specified metric gets 2x, others 1x; balanced = all 1x.
    Missing metrics receive a penalty score of 0.0 but still consume their
    full weight, so every job is scored on the same metric set.
    """
    bounds = _metric_bounds(scoreable)
    weights = _metric_weights(preference)
    return [_score_entry(entry, bounds, weights) for entry in scoreable]


def _generate_reasoning(
    scored: list[dict[str, Any]],
    winner: str | None,
    preference: str,
    excluded_count: int,
) -> str:
    """Generate a human-readable reasoning string for the comparison."""
    parts: list[str] = []

    if not winner:
        if not scored:
            parts.append("No jobs could be scored.")
        else:
            parts.append("Fewer than 2 jobs were scoreable, so no winner could be determined.")
        if excluded_count > 0:
            parts.append(f"{excluded_count} job(s) were excluded from scoring.")
        return " ".join(parts)

    parts.append(f"Winner: {winner} (preference: {preference}).")
    if len(scored) > 1:
        winner_entry = next((s for s in scored if s["job_id"] == winner), None)
        if winner_entry:
            parts.append(f"Score: {winner_entry['score']:.4f}.")
    if excluded_count > 0:
        parts.append(f"{excluded_count} job(s) excluded from scoring.")
    return " ".join(parts)


def compare_results(
    job_ids: list[str],
    preference: str = "balanced",
) -> dict[str, Any]:
    """Compare multiple optimization job results with preference-weighted scoring.

    Args:
        job_ids: List of 2–10 job IDs to compare.
        preference: Scoring preference — "latency", "size", "accuracy", or "balanced".

    Returns:
        Structured comparison with scored jobs, winner, reasoning, excluded jobs,
        and the effective preference. Always includes ``side_effect: False``.
    """
    try:
        # --- Input validation ---
        if not isinstance(job_ids, list) or not (_MIN_JOBS <= len(job_ids) <= _MAX_JOBS):
            return err(
                "invalid_job_count",
                f"job_ids must contain {_MIN_JOBS}\u2013{_MAX_JOBS} entries.",
                side_effect=False,
            )

        # Validate each job_id format and reject duplicates
        seen_ids: set[str] = set()
        for jid in job_ids:
            if not isinstance(jid, str) or not _JOB_ID_PATTERN.match(jid):
                return err(
                    "invalid_job_id",
                    f"Invalid job_id: {jid}",
                    side_effect=False,
                )
            if jid in seen_ids:
                return err(
                    "invalid_job_id",
                    f"Duplicate job_id: {jid}",
                    side_effect=False,
                )
            seen_ids.add(jid)

        # Normalize preference
        effective_preference = preference if preference in _VALID_PREFERENCES else "balanced"

        # --- Fetch job statuses ---
        excluded_jobs: list[dict[str, str]] = []
        scoreable: list[dict[str, Any]] = []

        for jid in job_ids:
            response = studio_request("GET", f"{_STATUS_PATH}/{jid}")

            if _is_error(response):
                excluded_jobs.append({"job_id": jid, "reason": "fetch_failed"})
                continue

            status = response.get("status", "unknown")

            if status in {"failed", "cancelled"}:
                excluded_jobs.append({"job_id": jid, "reason": f"job_{status}"})
                continue

            if status != "completed":
                excluded_jobs.append({"job_id": jid, "reason": f"status_{status}"})
                continue

            # A completed job is comparable only when it exposes at least one
            # optimization metric. latestMetrics is GPU sampling telemetry, so
            # do not treat an empty telemetry payload as a scored result.
            metrics = _extract_metrics(response)
            if not any(value is not None for value in metrics.values()):
                excluded_jobs.append({"job_id": jid, "reason": "no_comparable_metrics"})
                continue
            scoreable.append(
                {
                    "job_id": jid,
                    "status": status,
                    "metrics": metrics,
                }
            )

        # --- Score and select winner ---
        if len(scoreable) < _MIN_JOBS:
            # Not enough scoreable jobs for comparison
            comparison = []
            for entry in scoreable:
                comparison.append(
                    {
                        "job_id": entry["job_id"],
                        "status": entry["status"],
                        "metrics": entry["metrics"],
                        "score": 0.0,
                    }
                )

            return {
                "comparison": comparison,
                "winner": None,
                "reasoning": _generate_reasoning(comparison, None, effective_preference, len(excluded_jobs)),
                "excluded_jobs": excluded_jobs,
                "preference": effective_preference,
                "side_effect": False,
            }

        scored = _normalize_and_score(scoreable, effective_preference)

        # Select winner (highest score)
        winner_entry = max(scored, key=lambda x: x["score"])
        winner = winner_entry["job_id"]

        reasoning = _generate_reasoning(scored, winner, effective_preference, len(excluded_jobs))

        return {
            "comparison": scored,
            "winner": winner,
            "reasoning": reasoning,
            "excluded_jobs": excluded_jobs,
            "preference": effective_preference,
            "side_effect": False,
        }

    except Exception as exc:
        logger.warning("compare_results unexpected error", exc_info=True)
        return err("internal_error", type(exc).__name__, side_effect=False)
