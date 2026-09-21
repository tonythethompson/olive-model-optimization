#!/usr/bin/env python3
"""Weekly update script: refresh knowledge base from official sources.

Fetches Olive docs, GitHub issues/releases, and ONNX Runtime EP docs,
then parses the returned Markdown-ish content for headings, bullet
candidate-quirks, and deprecation warnings. Writes a deterministic report,
candidate quirks file, and refresh metadata for CI/PR workflows.

Refresh metadata and report fields are content-addressed: identical upstream
sources produce identical outputs (no wall-clock timestamps in written files).
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

# Add the MCP server package to the path when running the script directly.
_MCP_DIR = Path(__file__).resolve().parent.parent
import sys  # noqa: E402

if str(_MCP_DIR) not in sys.path:
    sys.path.insert(0, str(_MCP_DIR))

from olive_mcp_server.fetchers import (  # noqa: E402
    fetch_github_issues,
    fetch_official_docs,
    fetch_onnx_runtime_docs,
)

HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.+)$", re.MULTILINE)
BULLET_RE = re.compile(r"^\s*[-*]\s+(.+)$", re.MULTILINE)
DEPRECATION_RE = re.compile(
    r"\b(deprecated?|deprecat(?:ion|ed|e|ing)|no longer supported)\b",
    re.IGNORECASE,
)
# ISO-8601 timestamps that may appear in fetched source text (e.g. GitHub bodies).
ISO_TIMESTAMP_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)?)\b")

GENERATOR_NAME = "update_kb"
REFRESH_METADATA_NAME = "refresh_metadata.json"
KB_REL_PREFIX = "olive_mcp_server/knowledge_base"


def _generator_version() -> str:
    """Return the package/generator version used in refresh metadata."""
    try:
        from olive_mcp_server import __version__

        return str(__version__)
    except Exception:  # noqa: BLE001
        return "0.1.0"


def _canonical_json(data: Any) -> str:
    """Serialize data to a stable JSON string for hashing and comparisons."""
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _content_fingerprint(data: Any) -> str:
    """SHA-256 hex digest of canonical JSON for content-addressed identity."""
    return hashlib.sha256(_canonical_json(data).encode("utf-8")).hexdigest()


def _dump_json_text(data: Any) -> str:
    """Pretty-print JSON with a trailing newline (stable formatting)."""
    return json.dumps(data, indent=2, ensure_ascii=False) + "\n"


def _write_json_if_changed(path: Path, data: Any) -> bool:
    """Write JSON only when content differs. Returns True if the file changed."""
    new_text = _dump_json_text(data)
    if path.exists():
        try:
            if path.read_text(encoding="utf-8") == new_text:
                return False
        except OSError:
            pass
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(new_text, encoding="utf-8")
    return True


def _kb_rel(name: str) -> str:
    """Repo-relative path for a knowledge_base file (workflow-friendly)."""
    return f"{KB_REL_PREFIX}/{name}"


def _coalesce_text(data: Any) -> str:
    """Flatten fetcher output into a single string for parsing."""
    if isinstance(data, str):
        return data
    if isinstance(data, dict):
        if isinstance(data.get("content"), str):
            return data["content"]
        parts: list[str] = []
        pages = data.get("pages", {})
        if isinstance(pages, dict):
            parts.extend(v for v in pages.values() if isinstance(v, str))
        overview = data.get("overview")
        if isinstance(overview, str) and overview:
            parts.append(overview)
        return "\n\n".join(parts)
    return ""


def _extract_headings(text: str) -> list[str]:
    return [match.group(2).strip() for match in HEADING_RE.finditer(text)]


def _extract_bullets(text: str) -> list[str]:
    return [match.group(1).strip() for match in BULLET_RE.finditer(text)]


def _extract_deprecations(text: str) -> list[str]:
    """Find sentences mentioning deprecation or removal."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if DEPRECATION_RE.search(s)]


def _parse_source(data: Any) -> dict[str, Any]:
    """Parse fetcher output into structured sections and deprecation flags."""
    text = _coalesce_text(data)
    return {
        "headings": _extract_headings(text)[:30],
        "bullets": _extract_bullets(text)[:30],
        "deprecations": _extract_deprecations(text)[:20],
        "word_count": len(text.split()),
    }


def _build_candidate_quirks(parsed: dict[str, dict[str, Any]]) -> list[dict[str, str]]:
    """Turn bullet points from each source into candidate quirks for review."""
    candidates: list[dict[str, str]] = []
    for source_name, info in parsed.items():
        for bullet in info.get("bullets", [])[:20]:
            title = bullet[:80] + ("..." if len(bullet) > 80 else "")
            candidates.append(
                {
                    "category": source_name,
                    "title": title,
                    "description": bullet,
                    "source": "update_kb",
                }
            )
    return candidates


def _sanitize_source(data: Any) -> Any:
    """Replace large raw text fields with a small preview for the report."""
    if isinstance(data, str):
        return {"word_count": len(data.split()), "preview": data[:500]}
    if isinstance(data, dict):
        sanitized: dict[str, Any] = {}
        for key, value in data.items():
            if key == "content" and isinstance(value, str):
                sanitized[key] = {"word_count": len(value.split()), "preview": value[:500]}
            elif key == "pages" and isinstance(value, dict):
                sanitized[key] = {
                    k: {"word_count": len(v.split()), "preview": v[:500]} if isinstance(v, str) else v
                    for k, v in value.items()
                }
            elif key == "overview" and isinstance(value, str):
                sanitized[key] = {"word_count": len(value.split()), "preview": value[:500]}
            else:
                sanitized[key] = value
        return sanitized
    return data


def _collect_source_timestamps(raw_sources: dict[str, Any]) -> list[str]:
    """Collect ISO-like timestamps embedded in fetched source payloads."""
    found: list[str] = []
    for data in raw_sources.values():
        text = _coalesce_text(data)
        found.extend(match.group(1) for match in ISO_TIMESTAMP_RE.finditer(text))
        if isinstance(data, dict):
            for key in ("updated_at", "published_at", "created_at", "last_modified", "source_timestamp"):
                value = data.get(key)
                if isinstance(value, str) and ISO_TIMESTAMP_RE.fullmatch(value.strip()):
                    found.append(value.strip())
    return found


def _source_timestamp(raw_sources: dict[str, Any], fingerprint: str) -> str:
    """Deterministic source stamp: max upstream ISO time, else content digest.

    Never uses wall-clock generation time so no-op refreshes stay diff-stable.
    """
    timestamps = _collect_source_timestamps(raw_sources)
    if timestamps:
        return max(timestamps)
    return f"content:{fingerprint}"


def _load_refresh_metadata(path: Path) -> dict[str, Any]:
    """Load existing refresh metadata or return an empty scaffold."""
    if not path.exists():
        return {"schema_version": 1, "runs": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"schema_version": 1, "runs": {}}
    if not isinstance(data, dict):
        return {"schema_version": 1, "runs": {}}
    runs = data.get("runs")
    if not isinstance(runs, dict):
        data["runs"] = {}
    data.setdefault("schema_version", 1)
    return data


def _merge_refresh_metadata(
    existing: dict[str, Any],
    *,
    run_meta: dict[str, Any],
    generator_version: str,
    source_timestamp: str,
    changed_files: list[str],
) -> dict[str, Any]:
    """Merge this generator's run into the shared refresh_metadata document."""
    runs = dict(existing.get("runs") or {})
    runs[GENERATOR_NAME] = run_meta

    # Union changed files across generators for workflow consumption.
    all_changed: list[str] = []
    seen: set[str] = set()
    for name in sorted(runs):
        run = runs[name]
        if not isinstance(run, dict):
            continue
        for path in run.get("changed_files") or []:
            if isinstance(path, str) and path not in seen:
                seen.add(path)
                all_changed.append(path)
    # Prefer this run's files first if they were just computed.
    for path in changed_files:
        if path not in seen:
            seen.add(path)
            all_changed.append(path)

    # Aggregate source_timestamp: prefer max ISO-like, else this run's stamp.
    stamps = [source_timestamp]
    for run in runs.values():
        if isinstance(run, dict) and isinstance(run.get("source_timestamp"), str):
            stamps.append(run["source_timestamp"])
    iso_stamps = [s for s in stamps if not s.startswith("content:")]
    aggregate_ts = max(iso_stamps) if iso_stamps else source_timestamp

    return {
        "schema_version": 1,
        "generator_version": generator_version,
        "source_timestamp": aggregate_ts,
        "changed_files": all_changed,
        "runs": runs,
    }


def main(kb_dir: Path | None = None) -> None:
    """Fetch external sources, parse them, and write freshness reports."""
    if kb_dir is None:
        kb_dir = Path(__file__).parent.parent / "olive_mcp_server" / "knowledge_base"
    kb_dir.mkdir(parents=True, exist_ok=True)

    generator_version = _generator_version()

    raw_sources: dict[str, Any] = {
        "official_docs": fetch_official_docs(),
        "github": fetch_github_issues(),
        "onnx_runtime": fetch_onnx_runtime_docs(),
    }

    parsed = {name: _parse_source(data) for name, data in raw_sources.items()}
    source_fingerprint = _content_fingerprint(
        {
            "sources": {name: _sanitize_source(data) for name, data in raw_sources.items()},
            "parsed": parsed,
        }
    )
    source_ts = _source_timestamp(raw_sources, source_fingerprint)

    source_statuses: list[str] = []
    for data in raw_sources.values():
        if isinstance(data, dict) and isinstance(data.get("status"), str):
            source_statuses.append(data["status"])
        else:
            source_statuses.append("ok" if data else "error")

    success = bool(source_statuses) and all(status != "error" for status in source_statuses)

    # Deterministic report: no wall-clock fields.
    report: dict[str, Any] = {
        "source_timestamp": source_ts,
        "generator": GENERATOR_NAME,
        "generator_version": generator_version,
        "source_fingerprint": source_fingerprint,
        "success": success,
        "sources": {name: _sanitize_source(data) for name, data in raw_sources.items()},
        "parsed": parsed,
        "deprecations": [deprecation for section in parsed.values() for deprecation in section.get("deprecations", [])],
    }

    changed_files: list[str] = []
    metadata_path = kb_dir / REFRESH_METADATA_NAME
    existing_meta = _load_refresh_metadata(metadata_path)

    update_report_path = kb_dir / "update_report.json"
    if _write_json_if_changed(update_report_path, report):
        changed_files.append(_kb_rel("update_report.json"))
    print(f"Update report written to {update_report_path}")

    candidate_quirks = _build_candidate_quirks(parsed)
    candidate_path = kb_dir / "candidate_quirks.json"
    if _write_json_if_changed(candidate_path, candidate_quirks):
        changed_files.append(_kb_rel("candidate_quirks.json"))
    print(f"Candidate quirks written to {candidate_path}")

    run_meta: dict[str, Any] = {
        "generator": GENERATOR_NAME,
        "generator_version": generator_version,
        "source_timestamp": source_ts,
        "source_fingerprint": source_fingerprint,
        "changed_files": list(changed_files),
        "success": success,
    }

    metadata = _merge_refresh_metadata(
        existing_meta,
        run_meta=run_meta,
        generator_version=generator_version,
        source_timestamp=source_ts,
        changed_files=changed_files,
    )
    # Sidecar only — never list refresh_metadata.json in changed_files (avoids
    # self-referential churn). Workflows read this file for the file list.
    _write_json_if_changed(metadata_path, metadata)
    print(f"Refresh metadata written to {metadata_path}")
    print(
        f"update_kb metadata: generator_version={generator_version} "
        f"source_timestamp={source_ts} changed_files={changed_files}"
    )

    if not success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
