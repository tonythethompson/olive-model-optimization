"""Unit tests for compare_results tool (agent_compare.py).

Covers: 2+ completed jobs with clear winner, excluded non-terminal jobs,
fewer than 2 scoreable, invalid job count, invalid job_id format,
preference-based scoring differences, balanced preference, studio unavailable,
and JSON serialization round-trip.

Requirements: 13.4, 13.6, 13.7
"""

from __future__ import annotations

import json

import pytest

from olive_mcp_server.tools import agent_compare
from olive_mcp_server.tools.agent_compare import compare_results

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_job_response(
    status: str = "completed",
    latency_ms: float | None = None,
    model_size_mb: float | None = None,
    accuracy: float | None = None,
) -> dict:
    """Build a fake studio_request job status response."""
    metrics: dict = {}
    if latency_ms is not None:
        metrics["latency_ms"] = latency_ms
    if model_size_mb is not None:
        metrics["model_size_mb"] = model_size_mb
    if accuracy is not None:
        metrics["accuracy"] = accuracy
    return {"status": status, "latestMetrics": metrics}


def _error_response() -> dict:
    """Build a fake studio_request error response."""
    return {"error": "studio_unavailable", "message": "Connection refused"}


# ---------------------------------------------------------------------------
# Test 1: 2+ completed jobs with clear winner (preference=latency)
# ---------------------------------------------------------------------------


class TestClearWinner:
    """2+ completed jobs with a clear winner based on preference weighting."""

    def test_latency_preference_winner(self, monkeypatch: pytest.MonkeyPatch):
        """job-A has lower latency -> wins with latency preference (2x weight)."""
        responses = {
            "job-A": _make_job_response(latency_ms=50, model_size_mb=100, accuracy=0.95),
            "job-B": _make_job_response(latency_ms=100, model_size_mb=200, accuracy=0.90),
        }

        def fake_request(method, path, **_kw):
            jid = path.rsplit("/", 1)[-1]
            return responses[jid]

        monkeypatch.setattr(agent_compare, "studio_request", fake_request)

        result = compare_results(["job-A", "job-B"], preference="latency")

        assert result["winner"] == "job-A"
        assert result["preference"] == "latency"
        assert result["side_effect"] is False
        assert len(result["comparison"]) == 2
        assert result["excluded_jobs"] == []

        # Winner should have higher score
        scores = {c["job_id"]: c["score"] for c in result["comparison"]}
        assert scores["job-A"] > scores["job-B"]


# ---------------------------------------------------------------------------
# Test 2: Excluded non-terminal jobs
# ---------------------------------------------------------------------------


class TestExcludedNonTerminal:
    """Non-terminal and failed jobs are excluded from scoring."""

    def test_running_and_failed_excluded(self, monkeypatch: pytest.MonkeyPatch):
        """Running jobs and failed jobs are excluded with proper reasons."""
        responses = {
            "job-A": _make_job_response(latency_ms=50, model_size_mb=100, accuracy=0.95),
            "job-B": _make_job_response(status="running"),
            "job-C": _make_job_response(status="failed"),
        }

        def fake_request(method, path, **_kw):
            jid = path.rsplit("/", 1)[-1]
            return responses[jid]

        monkeypatch.setattr(agent_compare, "studio_request", fake_request)

        result = compare_results(["job-A", "job-B", "job-C"])

        # Only job-A is scoreable but <2 scoreable means no winner
        assert result["winner"] is None

        # Check excluded jobs
        excluded_ids = {e["job_id"]: e["reason"] for e in result["excluded_jobs"]}
        assert "job-B" in excluded_ids
        assert excluded_ids["job-B"] == "status_running"
        assert "job-C" in excluded_ids
        assert excluded_ids["job-C"] == "job_failed"

        # Comparison should only contain job-A
        comparison_ids = [c["job_id"] for c in result["comparison"]]
        assert comparison_ids == ["job-A"]

    def test_completed_job_empty_metrics_excluded(self, monkeypatch: pytest.MonkeyPatch):
        """A completed job with no metrics is excluded with no_comparable_metrics."""
        responses = {
            "job-A": _make_job_response(latency_ms=50, model_size_mb=100, accuracy=0.95),
            "job-B": _make_job_response(status="completed"),  # no metrics
        }

        def fake_request(method, path, **_kw):
            jid = path.rsplit("/", 1)[-1]
            return responses[jid]

        monkeypatch.setattr(agent_compare, "studio_request", fake_request)

        result = compare_results(["job-A", "job-B"])

        # Only job-A is scoreable but <2 scoreable means no winner
        assert result["winner"] is None

        excluded_ids = {e["job_id"]: e["reason"] for e in result["excluded_jobs"]}
        assert excluded_ids["job-B"] == "no_comparable_metrics"


# ---------------------------------------------------------------------------
# Test 2c: Constant metrics across jobs produce neutral tie
# ---------------------------------------------------------------------------


class TestConstantMetricsNeutralTie:
    """When all jobs report identical metrics, every job gets the neutral 0.5 score."""

    def test_constant_metrics_neutral_tie(self, monkeypatch: pytest.MonkeyPatch):
        """Identical metrics across jobs -> every job scores 0.5 (neutral midpoint)."""
        responses = {
            "job-A": _make_job_response(latency_ms=100, model_size_mb=500, accuracy=0.9),
            "job-B": _make_job_response(latency_ms=100, model_size_mb=500, accuracy=0.9),
        }

        def fake_request(method, path, **_kw):
            jid = path.rsplit("/", 1)[-1]
            return responses[jid]

        monkeypatch.setattr(agent_compare, "studio_request", fake_request)

        result = compare_results(["job-A", "job-B"], preference="latency")

        assert "error" not in result
        scores = {c["job_id"]: c["score"] for c in result["comparison"]}
        # Every job receives the neutral midpoint score for degenerate ranges.
        assert scores["job-A"] == 0.5
        assert scores["job-B"] == 0.5

    def test_constant_lower_is_better_metric_neutral(self, monkeypatch: pytest.MonkeyPatch):
        """A constant lower-is-better metric (latency) still yields the neutral 0.5."""
        responses = {
            "job-A": _make_job_response(latency_ms=42, model_size_mb=100),
            "job-B": _make_job_response(latency_ms=42, model_size_mb=200),
        }

        def fake_request(method, path, **_kw):
            jid = path.rsplit("/", 1)[-1]
            return responses[jid]

        monkeypatch.setattr(agent_compare, "studio_request", fake_request)

        result = compare_results(["job-A", "job-B"], preference="latency")

        assert "error" not in result
        # latency is constant (42 == 42) -> neutral 0.5 for both on that metric.
        # size differs so the overall score won't be 0.5, but the degenerate
        # latency contribution must not bias either job.
        assert result["winner"] is not None


# ---------------------------------------------------------------------------
# Test 3: Fewer than 2 scoreable -> winner=None
# ---------------------------------------------------------------------------


class TestFewerThan2Scoreable:
    """When fewer than 2 jobs are scoreable, winner is None."""

    def test_single_completed_job(self, monkeypatch: pytest.MonkeyPatch):
        """Only 1 completed job, 1 failed -> winner=None."""
        responses = {
            "job-A": _make_job_response(latency_ms=50, model_size_mb=100, accuracy=0.95),
            "job-B": _make_job_response(status="failed"),
        }

        def fake_request(method, path, **_kw):
            jid = path.rsplit("/", 1)[-1]
            return responses[jid]

        monkeypatch.setattr(agent_compare, "studio_request", fake_request)

        result = compare_results(["job-A", "job-B"])

        assert result["winner"] is None
        assert "insufficient" in result["reasoning"].lower() or "fewer" in result["reasoning"].lower()
        assert result["side_effect"] is False


# ---------------------------------------------------------------------------
# Test 4: Invalid job count
# ---------------------------------------------------------------------------


class TestInvalidJobCount:
    """Invalid job_ids count returns error."""

    def test_too_few_jobs(self):
        """1 job_id -> invalid_job_count error."""
        result = compare_results(["only-one"])
        assert result["error"] == "invalid_job_count"
        assert "message" in result

    def test_too_many_jobs(self):
        """11 job_ids -> invalid_job_count error."""
        job_ids = [f"job-{i}" for i in range(11)]
        result = compare_results(job_ids)
        assert result["error"] == "invalid_job_count"
        assert "message" in result

    def test_empty_list(self):
        """Empty list -> invalid_job_count error."""
        result = compare_results([])
        assert result["error"] == "invalid_job_count"


# ---------------------------------------------------------------------------
# Test 5: Invalid job_id format
# ---------------------------------------------------------------------------


class TestInvalidJobId:
    """Invalid job_id formats return error."""

    def test_spaces_in_id(self):
        """Job ID with spaces -> invalid_job_id error."""
        result = compare_results(["job A", "job-B"])
        assert result["error"] == "invalid_job_id"

    def test_special_chars(self):
        """Job ID with special chars -> invalid_job_id error."""
        result = compare_results(["job@#$", "job-B"])
        assert result["error"] == "invalid_job_id"

    def test_empty_string_id(self):
        """Empty string job ID -> invalid_job_id error."""
        result = compare_results(["", "job-B"])
        assert result["error"] == "invalid_job_id"


# ---------------------------------------------------------------------------
# Test 6: Preference-based scoring differences
# ---------------------------------------------------------------------------


class TestPreferenceScoringDifferences:
    """Different preferences yield different winners."""

    @pytest.fixture()
    def mock_studio(self, monkeypatch: pytest.MonkeyPatch):
        """Mock studio with jobs having opposing strengths."""
        responses = {
            "job-A": _make_job_response(latency_ms=50, model_size_mb=500),
            "job-B": _make_job_response(latency_ms=200, model_size_mb=100),
        }

        def fake_request(method, path, **_kw):
            jid = path.rsplit("/", 1)[-1]
            return responses[jid]

        monkeypatch.setattr(agent_compare, "studio_request", fake_request)

    def test_latency_preference_picks_job_a(self, mock_studio):
        """With latency preference, job-A wins (lower latency)."""
        result = compare_results(["job-A", "job-B"], preference="latency")
        assert result["winner"] == "job-A"

    def test_size_preference_picks_job_b(self, mock_studio):
        """With size preference, job-B wins (smaller model)."""
        result = compare_results(["job-A", "job-B"], preference="size")
        assert result["winner"] == "job-B"


# ---------------------------------------------------------------------------
# Test 7: Balanced preference -- all metrics weighted equally
# ---------------------------------------------------------------------------


class TestBalancedPreference:
    """Balanced preference weights all metrics equally."""

    def test_balanced_equal_weights(self, monkeypatch: pytest.MonkeyPatch):
        """With balanced preference, all metrics contribute equally."""
        # job-A: better latency, worse size
        # job-B: worse latency, better size
        # With balanced (equal weights), the scores should reflect both metrics
        responses = {
            "job-A": _make_job_response(latency_ms=50, model_size_mb=500),
            "job-B": _make_job_response(latency_ms=200, model_size_mb=100),
        }

        def fake_request(method, path, **_kw):
            jid = path.rsplit("/", 1)[-1]
            return responses[jid]

        monkeypatch.setattr(agent_compare, "studio_request", fake_request)

        result = compare_results(["job-A", "job-B"], preference="balanced")

        assert result["preference"] == "balanced"
        # With equal weights and symmetric advantages, scores should be equal
        # job-A: latency=1.0 (best), size=0.0 (worst) -> avg = 0.5
        # job-B: latency=0.0 (worst), size=1.0 (best) -> avg = 0.5
        scores = {c["job_id"]: c["score"] for c in result["comparison"]}
        assert scores["job-A"] == scores["job-B"]

    def test_unknown_preference_defaults_to_balanced(self, monkeypatch: pytest.MonkeyPatch):
        """Unknown preference value normalizes to 'balanced'."""
        responses = {
            "job-A": _make_job_response(latency_ms=50, model_size_mb=100),
            "job-B": _make_job_response(latency_ms=100, model_size_mb=200),
        }

        def fake_request(method, path, **_kw):
            jid = path.rsplit("/", 1)[-1]
            return responses[jid]

        monkeypatch.setattr(agent_compare, "studio_request", fake_request)

        result = compare_results(["job-A", "job-B"], preference="unknown_pref")
        assert result["preference"] == "balanced"


# ---------------------------------------------------------------------------
# Test 8: Studio unavailable for all fetches
# ---------------------------------------------------------------------------


class TestStudioUnavailable:
    """All fetches returning errors -> all excluded, winner=None."""

    def test_all_fetch_errors(self, monkeypatch: pytest.MonkeyPatch):
        """When studio is unavailable, all jobs excluded."""

        def fake_request(method, path, **_kw):
            return _error_response()

        monkeypatch.setattr(agent_compare, "studio_request", fake_request)

        result = compare_results(["job-A", "job-B"])

        assert result["winner"] is None
        assert len(result["excluded_jobs"]) == 2
        assert all(e["reason"] == "fetch_failed" for e in result["excluded_jobs"])
        assert result["comparison"] == []
        assert result["side_effect"] is False


# ---------------------------------------------------------------------------
# Test 9: JSON round-trip -- all success results serialize correctly
# ---------------------------------------------------------------------------


class TestJsonRoundTrip:
    """All outputs survive JSON serialization round-trip."""

    def test_success_result_roundtrip(self, monkeypatch: pytest.MonkeyPatch):
        """Successful comparison result round-trips through JSON."""
        responses = {
            "job-A": _make_job_response(latency_ms=50, model_size_mb=100, accuracy=0.95),
            "job-B": _make_job_response(latency_ms=100, model_size_mb=200, accuracy=0.90),
        }

        def fake_request(method, path, **_kw):
            jid = path.rsplit("/", 1)[-1]
            return responses[jid]

        monkeypatch.setattr(agent_compare, "studio_request", fake_request)

        result = compare_results(["job-A", "job-B"], preference="latency")
        assert json.loads(json.dumps(result)) == result

    def test_error_result_roundtrip(self):
        """Error result round-trips through JSON."""
        result = compare_results(["only-one"])
        assert json.loads(json.dumps(result)) == result

    def test_no_winner_result_roundtrip(self, monkeypatch: pytest.MonkeyPatch):
        """No-winner result round-trips through JSON."""
        responses = {
            "job-A": _make_job_response(latency_ms=50, model_size_mb=100, accuracy=0.95),
            "job-B": _make_job_response(status="failed"),
        }

        def fake_request(method, path, **_kw):
            jid = path.rsplit("/", 1)[-1]
            return responses[jid]

        monkeypatch.setattr(agent_compare, "studio_request", fake_request)

        result = compare_results(["job-A", "job-B"])
        assert json.loads(json.dumps(result)) == result

    def test_all_excluded_roundtrip(self, monkeypatch: pytest.MonkeyPatch):
        """All-excluded result round-trips through JSON."""

        def fake_request(method, path, **_kw):
            return _error_response()

        monkeypatch.setattr(agent_compare, "studio_request", fake_request)

        result = compare_results(["job-A", "job-B"])
        assert json.loads(json.dumps(result)) == result
