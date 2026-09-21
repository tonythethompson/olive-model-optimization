"""Determinism and failure tests for the KB refresh script."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def _load_script():
    path = Path(__file__).parents[1] / "scripts" / "update_kb.py"
    spec = importlib.util.spec_from_file_location("update_kb_test_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_refresh_is_deterministic_and_preserves_candidates(monkeypatch, tmp_path):
    update_kb = _load_script()
    payloads = {
        "docs": {"status": "ok", "pages": {"index": "# Docs\n\n- documented option."}},
        "github": {
            "status": "ok",
            "content": "# Open Issues\n\n- deprecated API.",
            "source_timestamp": "2024-01-01T00:00:00Z",
        },
        "ort": {"status": "ok", "pages": {"CUDA": "## CUDA\n\n- use it."}},
    }
    monkeypatch.setattr(update_kb, "fetch_official_docs", lambda: payloads["docs"])
    monkeypatch.setattr(update_kb, "fetch_github_issues", lambda: payloads["github"])
    monkeypatch.setattr(update_kb, "fetch_onnx_runtime_docs", lambda: payloads["ort"])
    update_kb.main(tmp_path)
    first = {p.name: p.read_bytes() for p in tmp_path.glob("*.json")}
    update_kb.main(tmp_path)
    second = {p.name: p.read_bytes() for p in tmp_path.glob("*.json")}
    assert first["update_report.json"] == second["update_report.json"]
    assert first["candidate_quirks.json"] == second["candidate_quirks.json"]
    first_report = json.loads(first["update_report.json"])
    second_report = json.loads(second["update_report.json"])
    assert first_report["source_fingerprint"] == second_report["source_fingerprint"]
    assert first_report["source_timestamp"] == second_report["source_timestamp"]
    second_meta = json.loads(second["refresh_metadata.json"])
    assert second_meta["runs"]["update_kb"]["changed_files"] == []
    candidates = json.loads((tmp_path / "candidate_quirks.json").read_text())
    assert candidates and {"category", "title", "description", "source"} <= candidates[0].keys()
    report = json.loads((tmp_path / "update_report.json").read_text())
    assert report["deprecations"]


def test_ort_overview_is_parsed_alongside_pages(monkeypatch, tmp_path):
    update_kb = _load_script()
    monkeypatch.setattr(update_kb, "fetch_official_docs", lambda: {"status": "ok", "pages": {}})
    monkeypatch.setattr(update_kb, "fetch_github_issues", lambda: {"status": "ok", "content": ""})
    monkeypatch.setattr(
        update_kb,
        "fetch_onnx_runtime_docs",
        lambda: {
            "status": "ok",
            "pages": {"CUDA": "## CUDA\n\n- use it."},
            "overview": "## Overview\n\n- CPU runs everywhere.",
        },
    )
    update_kb.main(tmp_path)
    parsed = json.loads((tmp_path / "update_report.json").read_text())["parsed"]
    headings = parsed["onnx_runtime"]["headings"]
    assert "CUDA" in headings
    assert "Overview" in headings


def test_refresh_error_exits(tmp_path, monkeypatch):
    update_kb = _load_script()
    monkeypatch.setattr(update_kb, "fetch_official_docs", lambda: {"status": "error"})
    monkeypatch.setattr(update_kb, "fetch_github_issues", lambda: {"status": "ok", "content": ""})
    monkeypatch.setattr(update_kb, "fetch_onnx_runtime_docs", lambda: {"status": "ok", "pages": {}})
    with pytest.raises(SystemExit) as exc:
        update_kb.main(tmp_path)
    assert exc.value.code == 1
    assert json.loads((tmp_path / "update_report.json").read_text())["success"] is False
