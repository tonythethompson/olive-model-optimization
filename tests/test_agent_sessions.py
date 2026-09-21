"""Agent-loop session bridge contract tests."""

from olive_mcp_server.tools import studio_loopback


def test_ensure_session_creates_when_id_is_missing(monkeypatch):
    calls = []

    def fake_request(method, path, *, body=None, timeout=5.0):
        calls.append((method, path, body))
        return {"sessionId": "created-session", "attemptCount": 0}

    monkeypatch.setattr(studio_loopback, "studio_request", fake_request)

    session_id, session = studio_loopback._ensure_session(None)

    assert session_id == "created-session"
    assert session["attemptCount"] == 0
    assert calls == [("POST", "/api/olive/agent/sessions", {})]


def test_ensure_session_reads_existing_session(monkeypatch):
    calls = []

    def fake_request(method, path, *, body=None, timeout=5.0):
        calls.append((method, path, body))
        return {"sessionId": "existing-session", "attemptCount": 2}

    monkeypatch.setattr(studio_loopback, "studio_request", fake_request)

    session_id, session = studio_loopback._ensure_session("existing-session")

    assert session_id == "existing-session"
    assert session["attemptCount"] == 2
    assert calls == [("GET", "/api/olive/agent/sessions/existing-session", None)]


def test_record_attempt_sets_dispatch_flag(monkeypatch):
    calls = []

    def fake_request(method, path, *, body=None, timeout=5.0):
        calls.append((method, path, body))
        return {"sessionId": "session-1", "attemptCount": 1}

    monkeypatch.setattr(studio_loopback, "studio_request", fake_request)

    studio_loopback._record_attempt(
        "session-1",
        recipe={"passes": {}},
        failure="boom",
        success=False,
        note="diagnosed",
    )

    assert calls == [
        (
            "PUT",
            "/api/olive/agent/sessions/session-1",
            {
                "attempt": True,
                "recipe": {"passes": {}},
                "failure": "boom",
                "success": False,
                "note": "diagnosed",
            },
        )
    ]
