"""Fetcher for Olive GitHub releases and issues."""

from __future__ import annotations

import os
from typing import Any

import requests

from ._http import fetch_json

REPO = "microsoft/Olive"
RELEASES_URL = f"https://api.github.com/repos/{REPO}/releases"
ISSUES_URL = f"https://api.github.com/repos/{REPO}/issues"


def _issue_to_text(issue: dict[str, Any]) -> str:
    title = issue.get("title", "")
    number = issue.get("number", "")
    body = issue.get("body") or ""
    labels = [label.get("name", "") for label in issue.get("labels", [])]
    return f"## #{number} {title}\nLabels: {', '.join(labels)}\n{body}"


def _release_to_text(release: dict[str, Any]) -> str:
    name = release.get("name", "")
    tag = release.get("tag_name", "")
    body = release.get("body") or ""
    flags = []
    if release.get("prerelease"):
        flags.append("prerelease")
    if release.get("draft"):
        flags.append("draft")
    suffix = f" [{' / '.join(flags)}]" if flags else ""
    return f"## {name} ({tag}){suffix}\n{body}"


def _headers() -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _request_error(response: requests.Response) -> None:
    if response.status_code in {403, 429} and response.headers.get("x-ratelimit-remaining") == "0":
        raise requests.HTTPError("GitHub API rate limit exhausted (x-ratelimit-remaining: 0)")
    response.raise_for_status()


def fetch_github_issues(labels: list[str] | None = None, max_results: int = 50) -> dict[str, Any]:
    """Fetch recent Olive GitHub issues and release notes.

    Args:
        labels: Issue labels to filter on, e.g. ["bug", "quantization"].
        max_results: Maximum number of issues to retrieve.

    Returns:
        Dict with releases and issue summaries as a single text body.
    """
    result: dict[str, Any] = {
        "status": "ok",
        "source": f"https://github.com/{REPO}",
        "labels": labels or [],
        "max_results": max_results,
    }

    errors: dict[str, str] = {}
    issues: list[dict[str, Any]] = []
    releases: list[dict[str, Any]] = []
    headers = _headers()
    try:
        per_page = min(max(max_results, 1), 100)
        params: dict[str, str | int] = {
            "state": "open",
            "sort": "updated",
            "direction": "desc",
            "per_page": per_page,
        }
        if labels:
            params["labels"] = ",".join(labels)
        for page in range(1, 4):
            params["page"] = page
            issues_response = fetch_json(ISSUES_URL, params=params, headers=headers)
            _request_error(issues_response)
            raw_batch = issues_response.json()
            batch = [item for item in raw_batch if "pull_request" not in item]
            issues.extend(batch)
            if len(issues) >= max_results or len(raw_batch) < per_page:
                break
        issues = issues[:max_results]
    except Exception as exc:  # noqa: BLE001
        errors["issues_error"] = str(exc)

    try:
        releases_response = fetch_json(RELEASES_URL, params={"per_page": 5}, headers=headers)
        _request_error(releases_response)
        releases = releases_response.json()[:5]
    except Exception as exc:  # noqa: BLE001
        errors["releases_error"] = str(exc)

    sections = ["# Recent Releases"]
    sections.extend(_release_to_text(r) for r in releases)
    sections.append("# Open Issues")
    sections.extend(_issue_to_text(i) for i in issues)
    result["content"] = "\n\n".join(sections)
    stamps = [item["updated_at"] for item in issues if isinstance(item.get("updated_at"), str)]
    stamps.extend(item["published_at"] for item in releases if isinstance(item.get("published_at"), str))
    if stamps:
        result["source_timestamp"] = max(stamps)
    if errors:
        result.update(errors)
        result["status"] = "error" if len(errors) == 2 else "partial"

    return result
