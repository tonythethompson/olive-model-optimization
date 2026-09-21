"""Mocked HTTP tests for knowledge-base fetchers."""

from __future__ import annotations

import pytest

from olive_mcp_server.fetchers import _http, github_scraper
from olive_mcp_server.fetchers import official_docs_fetcher as docs
from olive_mcp_server.fetchers import onnx_runtime_fetcher as ort


class Response:
    def __init__(self, text: str = "", status_code: int = 200, payload=None, headers=None):
        self.text = text
        self.status_code = status_code
        self.headers = headers or {"Content-Type": "text/html"}
        self._payload = payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


class Session:
    def __init__(self, responses):
        self.responses = responses
        self.urls = []

    def get(self, url, **kwargs):
        self.urls.append(url)
        return self.responses[url].pop(0)


HTML = """
<html><body><nav>bad nav</nav><main><h1>Title</h1>
<aside class="sidebar">bad sidebar</aside><p>Text</p><ul><li>one</li></ul>
<pre>key: value</pre></main></body></html>
"""


def test_official_docs_resolution_and_markdown(monkeypatch):
    session = Session(
        {
            "https://microsoft.github.io/Olive/index.html": [Response(HTML)],
        }
    )
    monkeypatch.setattr("olive_mcp_server.fetchers._http.get_session", lambda: session)
    result = docs.fetch_official_docs(pages=["index"])
    assert result["status"] == "ok"
    assert session.urls == ["https://microsoft.github.io/Olive/index.html"]
    assert "# Title" in result["pages"]["index"]
    assert "- one" in result["pages"]["index"]
    assert "```" in result["pages"]["index"]
    assert "bad nav" not in result["pages"]["index"]
    assert result["sources"]["index"].endswith("index.html")


def test_official_docs_unknown_falls_back(monkeypatch):
    url = "https://microsoft.github.io/Olive/custom.html"
    session = Session(
        {
            "https://microsoft.github.io/Olive/custom": [Response(status_code=404)],
            url: [Response(HTML)],
        }
    )
    monkeypatch.setattr("olive_mcp_server.fetchers._http.get_session", lambda: session)
    result = docs.fetch_official_docs(pages=["custom"])
    assert result["status"] == "ok"
    assert session.urls == ["https://microsoft.github.io/Olive/custom", url]


def test_official_docs_failures_set_partial_and_error(monkeypatch):
    session = Session(
        {
            "https://microsoft.github.io/Olive/index.html": [Response(status_code=404)],
            "https://microsoft.github.io/Olive/why-olive.html": [Response(HTML)],
        }
    )
    monkeypatch.setattr("olive_mcp_server.fetchers._http.get_session", lambda: session)
    one_failed = docs.fetch_official_docs(pages=["index", "why-olive"])
    assert one_failed["status"] == "partial"
    assert isinstance(one_failed["pages"]["index"], dict)
    failed_session = Session(
        {
            "https://microsoft.github.io/Olive/index.html": [Response(status_code=404)],
        }
    )
    monkeypatch.setattr("olive_mcp_server.fetchers._http.get_session", lambda: failed_session)
    all_failed = docs.fetch_official_docs(pages=["index"])
    assert all_failed["status"] == "error"


def test_ort_uses_case_sensitive_slugs_and_cpu_overview(monkeypatch):
    urls = {
        "https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html": [Response(HTML)],
        "https://onnxruntime.ai/docs/execution-providers/": [Response(HTML)],
    }
    session = Session(urls)
    monkeypatch.setattr("olive_mcp_server.fetchers._http.get_session", lambda: session)
    result = ort.fetch_onnx_runtime_docs(["CUDAExecutionProvider", "CPUExecutionProvider"])
    assert result["status"] == "ok"
    assert not any("cuda-executionprovider" in url for url in session.urls)
    assert "CPUExecutionProvider" not in result["pages"]
    assert result["informational"]["CPUExecutionProvider"]


def test_ort_overview_failure_is_partial(monkeypatch):
    session = Session(
        {
            "https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html": [Response(HTML)],
            "https://onnxruntime.ai/docs/execution-providers/": [Response(status_code=503)],
        }
    )
    monkeypatch.setattr("olive_mcp_server.fetchers._http.get_session", lambda: session)
    result = ort.fetch_onnx_runtime_docs(["CUDAExecutionProvider"])
    assert result["status"] == "partial"
    assert "overview_error" in result


def test_http_retries_503_and_honors_retry_after(monkeypatch):
    response = Response(HTML, status_code=503, headers={"Content-Type": "text/html", "Retry-After": "7"})
    session = Session({"https://example.test/page": [response, Response(HTML)]})
    delays = []
    monkeypatch.setattr(_http, "get_session", lambda: session)
    monkeypatch.setattr(_http.time, "sleep", delays.append)
    assert _http.fetch_html("https://example.test/page")
    assert delays == [7.0]


def test_http_follows_same_host_redirect_and_rejects_cross_host(monkeypatch):
    session = Session(
        {
            "https://example.test/page": [
                Response(status_code=302, headers={"Location": "/final"}),
            ],
            "https://example.test/final": [Response(HTML)],
        }
    )
    monkeypatch.setattr(_http, "get_session", lambda: session)
    assert _http.fetch_html("https://example.test/page")
    assert session.urls == ["https://example.test/page", "https://example.test/final"]

    cross_host = Session(
        {
            "https://example.test/page": [
                Response(status_code=302, headers={"Location": "https://other.test/final"}),
            ],
        }
    )
    monkeypatch.setattr(_http, "get_session", lambda: cross_host)
    with pytest.raises(Exception, match="another host"):
        _http.fetch_html("https://example.test/page")


def test_http_retry_budget_does_not_consume_redirect_budget(monkeypatch):
    session = Session(
        {
            "https://example.test/page": [
                Response(status_code=503),
                Response(status_code=503),
                Response(status_code=302, headers={"Location": "/final"}),
            ],
            "https://example.test/final": [Response(status_code=503), Response(HTML)],
        }
    )
    monkeypatch.setattr(_http, "get_session", lambda: session)
    monkeypatch.setattr(_http.time, "sleep", lambda _seconds: None)
    assert _http.fetch_html("https://example.test/page")


def test_github_filters_prs_and_paginates(monkeypatch):
    calls = []
    issue_pages = [
        [
            {"number": 1, "title": "issue", "updated_at": "2024-01-01T00:00:00Z"},
            {"number": 2, "pull_request": {}, "title": "pr"},
        ],
        [{"number": 3, "title": "issue 2", "updated_at": "2024-02-01T00:00:00Z"}],
    ]

    def get(self, url, **kwargs):
        calls.append((url, kwargs))
        if url == github_scraper.ISSUES_URL:
            payload = issue_pages.pop(0)
        else:
            payload = [{"name": "release", "published_at": "2024-03-01T00:00:00Z"}]
        return Response(payload=payload)

    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    session = type("Session", (), {"get": get})()
    monkeypatch.setattr("olive_mcp_server.fetchers._http.get_session", lambda: session)
    result = github_scraper.fetch_github_issues(max_results=2)
    assert "## #2 pr" not in result["content"]
    assert "## #3 issue 2" in result["content"]
    assert result["source_timestamp"] == "2024-03-01T00:00:00Z"
    assert "Authorization" not in calls[0][1]["headers"]


@pytest.mark.parametrize("variable", ["GITHUB_TOKEN", "GH_TOKEN"])
def test_github_auth_header_is_optional_and_token_is_not_returned(monkeypatch, variable):
    token = "test-secret-token"
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.setenv(variable, token)
    calls = []

    def get(self, url, **kwargs):
        calls.append(kwargs)
        return Response(payload=[])

    session = type("Session", (), {"get": get})()
    monkeypatch.setattr("olive_mcp_server.fetchers._http.get_session", lambda: session)
    result = github_scraper.fetch_github_issues(max_results=1)
    assert calls[0]["headers"]["Authorization"] == f"Bearer {token}"
    assert token not in repr(result)


def test_github_rate_limit_is_error(monkeypatch):
    response = Response(
        status_code=403,
        headers={"x-ratelimit-remaining": "0", "Content-Type": "application/json"},
    )
    session = type("Session", (), {"get": lambda self, *args, **kwargs: response})()
    monkeypatch.setattr("olive_mcp_server.fetchers._http.get_session", lambda: session)
    result = github_scraper.fetch_github_issues()
    assert result["status"] == "error"
    assert "rate limit" in result["issues_error"].lower()
    assert "rate limit" in result["releases_error"].lower()


def test_github_releases_failure_keeps_issues(monkeypatch):
    calls = []

    def get(self, url, **kwargs):
        calls.append(url)
        if url == github_scraper.ISSUES_URL:
            return Response(payload=[{"number": 4, "title": "kept"}])
        return Response(status_code=500, headers={"Content-Type": "application/json"})

    session = type("Session", (), {"get": get})()
    monkeypatch.setattr("olive_mcp_server.fetchers._http.get_session", lambda: session)
    result = github_scraper.fetch_github_issues(max_results=1)
    assert result["status"] == "partial"
    assert "## #4 kept" in result["content"]
    assert "releases_error" in result
