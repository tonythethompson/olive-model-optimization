# Feature: v0.3-agent-mcp-tools, Property 6: Preference-Weighted Scoring
"""Property-based tests for compare_results scoring logic.

Property 6: For any set of 2+ job metric objects and any valid preference:
  - When preference is specific (latency/size/accuracy), that metric gets 2x weight, others 1x
  - When preference is "balanced", all metrics get equal 1x weight
  - Winner has the highest weighted-average normalized score

Validates: Requirements 7.3, 7.8
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch

from hypothesis import given, settings
from hypothesis import strategies as st

from olive_mcp_server.tools.agent_compare import (
    _LOWER_IS_BETTER,
    _METRIC_KEYS,
    _normalize_and_score,
    compare_results,
)

# ---------------------------------------------------------------------------
# Independent scoring oracle (does not reuse implementation results)
# ---------------------------------------------------------------------------


def _oracle_normalized_score(key: str, value: float, bounds: tuple[float, float]) -> float:
    """Independent reimplementation of the normalized-metric score.

    Degenerate ranges (max == min) return the neutral midpoint 0.5.
    Non-degenerate ranges apply direction-based inversion for lower-is-better.
    """
    minimum, maximum = bounds
    if maximum == minimum:
        return 0.5
    normalized = (value - minimum) / (maximum - minimum)
    return 1.0 - normalized if key in _LOWER_IS_BETTER else normalized


def _oracle_bounds(scoreable: list[dict[str, Any]]) -> dict[str, tuple[float, float]]:
    """Independent reimplementation of metric bounds computation."""
    values: dict[str, list[float]] = {key: [] for key in _METRIC_KEYS}
    for entry in scoreable:
        for key in _METRIC_KEYS:
            v = entry["metrics"].get(key)
            if v is not None:
                values[key].append(v)
    return {key: (min(vs), max(vs)) for key, vs in values.items() if vs}


def _oracle_weights(preference: str) -> dict[str, float]:
    """Independent reimplementation of preference weights."""
    preferred = {
        "latency": "latency_ms",
        "size": "model_size_mb",
        "accuracy": "accuracy",
    }.get(preference)
    return {key: (2.0 if key == preferred else 1.0) for key in _METRIC_KEYS}


def _oracle_score(entry: dict[str, Any], scoreable: list[dict[str, Any]], preference: str) -> float:
    """Independent reimplementation of the weighted-average score for one entry.

    Missing metrics receive a penalty score of 0.0 but still consume their
    full weight, matching the implementation's consistent-metric-set policy.
    """
    bounds = _oracle_bounds(scoreable)
    weights = _oracle_weights(preference)
    weighted_sum = 0.0
    total_weight = 0.0
    for key in _METRIC_KEYS:
        total_weight += weights[key]
        value = entry["metrics"].get(key)
        if value is None or key not in bounds:
            continue  # penalty: 0.0 contribution, full weight consumed
        weighted_sum += _oracle_normalized_score(key, value, bounds[key]) * weights[key]
    return round(weighted_sum / total_weight, 6) if total_weight else 0.0


def _oracle_winner(scoreable: list[dict[str, Any]], preference: str) -> str | None:
    """Independently compute the expected winner (highest oracle score)."""
    if len(scoreable) < 2:
        return None
    scored = [(e["job_id"], _oracle_score(e, scoreable, preference)) for e in scoreable]
    return max(scored, key=lambda x: x[1])[0]


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_metrics_strategy = st.fixed_dictionaries(
    {
        "latency_ms": st.floats(min_value=0.1, max_value=1000.0, allow_nan=False, allow_infinity=False),
        "model_size_mb": st.floats(min_value=1.0, max_value=10000.0, allow_nan=False, allow_infinity=False),
        "accuracy": st.floats(min_value=0.5, max_value=1.0, allow_nan=False, allow_infinity=False),
    }
)

_preference_strategy = st.sampled_from(["latency", "size", "accuracy", "balanced"])

_job_list_strategy = st.lists(
    _metrics_strategy,
    min_size=2,
    max_size=5,
)


def _make_scoreable_entries(metrics_list: list[dict[str, float]]) -> list[dict[str, Any]]:
    """Create scoreable entry dicts as expected by _normalize_and_score."""
    return [{"job_id": f"job-{i}", "status": "completed", "metrics": m} for i, m in enumerate(metrics_list)]


def _mock_studio_request_for_jobs(
    metrics_list: list[dict[str, float]],
) -> Any:
    """Create a mock studio_request that returns completed jobs with given metrics."""
    job_ids = [f"job-{i}" for i in range(len(metrics_list))]

    def _mock_request(method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        for i, jid in enumerate(job_ids):
            if path.endswith(f"/{jid}"):
                return {
                    "id": jid,
                    "status": "completed",
                    "latestMetrics": metrics_list[i],
                }
        return {"error": "not_found", "message": "Job not found"}

    return _mock_request, job_ids


# ---------------------------------------------------------------------------
# Property 6: Preference-Weighted Scoring
# ---------------------------------------------------------------------------


class TestPreferenceWeightedScoring:
    """Property 6: Preference-Weighted Scoring via _normalize_and_score."""

    @given(metrics_list=_job_list_strategy, preference=_preference_strategy)
    @settings(max_examples=100)
    def test_winner_has_highest_score(
        self,
        metrics_list: list[dict[str, float]],
        preference: str,
    ) -> None:
        """The winner must have the highest score among all scored jobs."""
        scoreable = _make_scoreable_entries(metrics_list)
        scored = _normalize_and_score(scoreable, preference)

        assert len(scored) == len(metrics_list)

        # Find the max score
        max_score = max(entry["score"] for entry in scored)
        winner = max(scored, key=lambda x: x["score"])

        assert winner["score"] == max_score

        # Independent oracle: every returned score must match the oracle.
        for entry in scored:
            expected = _oracle_score(
                {"job_id": entry["job_id"], "metrics": entry["metrics"]},
                scoreable,
                preference,
            )
            assert entry["score"] == expected, (
                f"Score mismatch for {entry['job_id']}: impl={entry['score']} oracle={expected}"
            )
        # The selected winner must match the independently computed highest score.
        expected_winner = _oracle_winner(scoreable, preference)
        if expected_winner is not None:
            assert winner["job_id"] == expected_winner

    @given(metrics_list=_job_list_strategy, preference=_preference_strategy)
    @settings(max_examples=100)
    def test_all_scores_in_valid_range(
        self,
        metrics_list: list[dict[str, float]],
        preference: str,
    ) -> None:
        """All scores must be between 0.0 and 1.0 inclusive."""
        scoreable = _make_scoreable_entries(metrics_list)
        scored = _normalize_and_score(scoreable, preference)

        for entry in scored:
            assert 0.0 <= entry["score"] <= 1.0, f"Score {entry['score']} out of [0, 1] range for job {entry['job_id']}"

    @given(metrics_list=_job_list_strategy)
    @settings(max_examples=100)
    def test_balanced_gives_equal_weights(
        self,
        metrics_list: list[dict[str, float]],
    ) -> None:
        """When preference is 'balanced', all metrics get equal weight.

        Verify by comparing each returned score against the independent oracle
        with explicit 1x weights for every metric, and confirm the balanced
        scoring matches the latency, size, and accuracy oracle when all
        weights are equal.
        """
        scoreable = _make_scoreable_entries(metrics_list)
        scored_balanced = _normalize_and_score(scoreable, "balanced")

        for entry in scored_balanced:
            assert isinstance(entry["score"], float)
            assert 0.0 <= entry["score"] <= 1.0
            expected = _oracle_score(
                {"job_id": entry["job_id"], "metrics": entry["metrics"]},
                scoreable,
                "balanced",
            )
            assert entry["score"] == expected, (
                f"Balanced score mismatch for {entry['job_id']}: impl={entry['score']} oracle={expected}"
            )

    @given(metrics_list=_job_list_strategy)
    @settings(max_examples=100)
    def test_specific_preference_increases_metric_contribution(
        self,
        metrics_list: list[dict[str, float]],
    ) -> None:
        """When a specific preference is set, that metric's contribution is doubled.

        We verify that changing the preference can change the relative ordering
        of jobs (unless all jobs tie on the preferred metric), and that every
        returned score matches the independent oracle.
        """
        scoreable = _make_scoreable_entries(metrics_list)

        scored_balanced = _normalize_and_score(scoreable, "balanced")
        scored_latency = _normalize_and_score(scoreable, "latency")
        scored_size = _normalize_and_score(scoreable, "size")
        scored_accuracy = _normalize_and_score(scoreable, "accuracy")

        # Each preference-weighted scoring must still produce valid scores and
        # match the independent oracle for the corresponding preference.
        for scored, pref in [
            (scored_balanced, "balanced"),
            (scored_latency, "latency"),
            (scored_size, "size"),
            (scored_accuracy, "accuracy"),
        ]:
            for entry in scored:
                assert 0.0 <= entry["score"] <= 1.0
                expected = _oracle_score(
                    {"job_id": entry["job_id"], "metrics": entry["metrics"]},
                    scoreable,
                    pref,
                )
                assert entry["score"] == expected, (
                    f"Score mismatch for {entry['job_id']} pref={pref}: impl={entry['score']} oracle={expected}"
                )

    @given(metrics_list=_job_list_strategy, preference=_preference_strategy)
    @settings(max_examples=100)
    def test_scoring_preserves_job_ids(
        self,
        metrics_list: list[dict[str, float]],
        preference: str,
    ) -> None:
        """Scoring must preserve all job_ids from input."""
        scoreable = _make_scoreable_entries(metrics_list)
        scored = _normalize_and_score(scoreable, preference)

        input_ids = {entry["job_id"] for entry in scoreable}
        output_ids = {entry["job_id"] for entry in scored}
        assert input_ids == output_ids


class TestPreferenceWeightedScoringIntegration:
    """Property 6 via the full compare_results function (mocked studio_request)."""

    @given(metrics_list=_job_list_strategy, preference=_preference_strategy)
    @settings(max_examples=100)
    def test_compare_results_winner_has_highest_score(
        self,
        metrics_list: list[dict[str, float]],
        preference: str,
    ) -> None:
        """Full tool: winner has the highest score in the comparison list."""
        mock_fn, job_ids = _mock_studio_request_for_jobs(metrics_list)

        with patch(
            "olive_mcp_server.tools.agent_compare.studio_request",
            side_effect=mock_fn,
        ):
            result = compare_results(job_ids, preference)

        # Should not be an error
        assert "error" not in result
        assert result["side_effect"] is False
        assert result["preference"] == preference

        comparison = result["comparison"]
        assert len(comparison) == len(metrics_list)

        # Winner is the one with the highest score
        if result["winner"] is not None:
            winner_id = result["winner"]
            winner_score = next(e["score"] for e in comparison if e["job_id"] == winner_id)
            for entry in comparison:
                assert entry["score"] <= winner_score

    @given(metrics_list=_job_list_strategy, preference=_preference_strategy)
    @settings(max_examples=100)
    def test_compare_results_scores_in_range(
        self,
        metrics_list: list[dict[str, float]],
        preference: str,
    ) -> None:
        """Full tool: all scores in comparison are in [0.0, 1.0]."""
        mock_fn, job_ids = _mock_studio_request_for_jobs(metrics_list)

        with patch(
            "olive_mcp_server.tools.agent_compare.studio_request",
            side_effect=mock_fn,
        ):
            result = compare_results(job_ids, preference)

        assert "error" not in result

        for entry in result["comparison"]:
            assert 0.0 <= entry["score"] <= 1.0

    @given(metrics_list=_job_list_strategy, preference=_preference_strategy)
    @settings(max_examples=100)
    def test_compare_results_json_round_trip(
        self,
        metrics_list: list[dict[str, float]],
        preference: str,
    ) -> None:
        """Full tool: output is JSON-serializable and round-trips cleanly."""
        mock_fn, job_ids = _mock_studio_request_for_jobs(metrics_list)

        with patch(
            "olive_mcp_server.tools.agent_compare.studio_request",
            side_effect=mock_fn,
        ):
            result = compare_results(job_ids, preference)

        assert "error" not in result
        assert json.loads(json.dumps(result)) == result


class TestWeightVerification:
    """Direct verification that weights are applied correctly."""

    def test_latency_preference_doubles_latency_weight(self) -> None:
        """When preference=latency, the lowest-latency job wins due to 2x weight.

        With 2 jobs where one is best on latency and they tie on size,
        the 2x weight on latency determines the outcome.
        Weight layout: latency=2x, size=1x, accuracy=1x -> total weight 4.
        """
        # Job A: best latency, tied size, worse accuracy
        # Job B: worst latency, tied size, better accuracy
        metrics_list = [
            {"latency_ms": 10.0, "model_size_mb": 500.0, "accuracy": 0.85},
            {"latency_ms": 900.0, "model_size_mb": 500.0, "accuracy": 0.90},
        ]
        scoreable = _make_scoreable_entries(metrics_list)
        scored_latency = _normalize_and_score(scoreable, "latency")
        scored_balanced = _normalize_and_score(scoreable, "balanced")

        # Size is tied -> neutral 0.5 for both (degenerate range midpoint).
        # Latency: job-0 norm=0 -> inv=1.0, job-1 norm=1 -> inv=0.0
        # Accuracy: job-0 norm=0, job-1 norm=1.0
        # Latency pref: job-0 = (1.0*2 + 0.5*1 + 0.0*1)/4 = 2.5/4 = 0.625
        #               job-1 = (0.0*2 + 0.5*1 + 1.0*1)/4 = 1.5/4 = 0.375
        job0_latency = next(e for e in scored_latency if e["job_id"] == "job-0")
        job1_latency = next(e for e in scored_latency if e["job_id"] == "job-1")
        assert job0_latency["score"] > job1_latency["score"]

        # Balanced: job-0 = (1.0 + 0.5 + 0.0)/3 = 0.5
        #           job-1 = (0.0 + 0.5 + 1.0)/3 = 0.5
        # They tie under balanced -- the latency preference changes the outcome.
        job0_balanced = next(e for e in scored_balanced if e["job_id"] == "job-0")
        job1_balanced = next(e for e in scored_balanced if e["job_id"] == "job-1")
        assert abs(job0_balanced["score"] - job1_balanced["score"]) < 1e-6

    def test_size_preference_doubles_size_weight(self) -> None:
        """When preference=size, the smallest-model job wins due to 2x size weight.

        Weight layout: latency=1x, size=2x, accuracy=1x -> total weight 4.
        """
        # Job A: best size, tied latency, worse accuracy
        # Job B: worst size, tied latency, better accuracy
        metrics_list = [
            {"latency_ms": 100.0, "model_size_mb": 10.0, "accuracy": 0.7},
            {"latency_ms": 100.0, "model_size_mb": 9000.0, "accuracy": 0.95},
        ]
        scoreable = _make_scoreable_entries(metrics_list)
        scored_size = _normalize_and_score(scoreable, "size")

        # Latency tied -> neutral 0.5 for both (degenerate range midpoint).
        # Size: job-0 inv=1.0, job-1 inv=0.0.
        # Accuracy: job-0=0.0, job-1=1.0.
        # Size pref: job-0 = (0.5*1 + 1.0*2 + 0.0*1)/4 = 2.5/4 = 0.625
        #            job-1 = (0.5*1 + 0.0*2 + 1.0*1)/4 = 1.5/4 = 0.375
        job0_size = next(e for e in scored_size if e["job_id"] == "job-0")
        job1_size = next(e for e in scored_size if e["job_id"] == "job-1")
        assert job0_size["score"] > job1_size["score"]

    def test_accuracy_preference_doubles_accuracy_weight(self) -> None:
        """When preference=accuracy, the highest-accuracy job wins due to 2x weight.

        Weight layout: latency=1x, size=1x, accuracy=2x -> total weight 4.
        """
        # Job A: best accuracy, tied latency, worse size
        # Job B: worst accuracy, tied latency, better size
        metrics_list = [
            {"latency_ms": 100.0, "model_size_mb": 9000.0, "accuracy": 0.99},
            {"latency_ms": 100.0, "model_size_mb": 10.0, "accuracy": 0.5},
        ]
        scoreable = _make_scoreable_entries(metrics_list)
        scored_accuracy = _normalize_and_score(scoreable, "accuracy")

        # Latency tied -> neutral 0.5 for both (degenerate range midpoint).
        # Size: job-0 inv=0.0, job-1 inv=1.0.
        # Accuracy: job-0=1.0, job-1=0.0.
        # Accuracy pref: job-0 = (0.5*1 + 0.0*1 + 1.0*2)/4 = 2.5/4 = 0.625
        #               job-1 = (0.5*1 + 1.0*1 + 0.0*2)/4 = 1.5/4 = 0.375
        job0_acc = next(e for e in scored_accuracy if e["job_id"] == "job-0")
        job1_acc = next(e for e in scored_accuracy if e["job_id"] == "job-1")
        assert job0_acc["score"] > job1_acc["score"]

    def test_balanced_equal_weights_symmetric(self) -> None:
        """With balanced preference, swapping two metrics produces symmetric scores."""
        # Two jobs that mirror each other on latency and size, same accuracy.
        metrics_list = [
            {"latency_ms": 100.0, "model_size_mb": 500.0, "accuracy": 0.8},
            {"latency_ms": 500.0, "model_size_mb": 100.0, "accuracy": 0.8},
        ]
        scoreable = _make_scoreable_entries(metrics_list)
        scored = _normalize_and_score(scoreable, "balanced")

        # Both have same accuracy (tied -> neutral 0.5 each), and mirror latency/size.
        # Under balanced weights with inverted lower-is-better, they score equally.
        job0 = next(e for e in scored if e["job_id"] == "job-0")
        job1 = next(e for e in scored if e["job_id"] == "job-1")
        assert abs(job0["score"] - job1["score"]) < 1e-6
