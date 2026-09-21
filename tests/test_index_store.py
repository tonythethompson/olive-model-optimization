"""Phase 1: shipped embedding indexes and warm path."""

from __future__ import annotations

import pytest

from olive_mcp_server.tools import docs_search, load_troubleshooting
from olive_mcp_server.tools import troubleshooting as ts
from olive_mcp_server.tools.capabilities import get_mcp_capabilities
from olive_mcp_server.tools.index_store import (
    content_hash_pairs,
    load_entry_embeddings,
    load_pair_index,
    read_manifest,
    shipped_index_status,
)


def test_manifest_present_after_build():
    manifest = read_manifest()
    assert manifest is not None
    assert "docs_kb" in (manifest.get("indexes") or {})
    assert "ts_olive" in (manifest.get("indexes") or {})
    status = shipped_index_status()
    assert status["shipped"] is True
    assert status["version"]


def test_docs_shipped_index_loads():
    texts = docs_search._load_kb_text()
    expected = content_hash_pairs(texts)
    loaded = load_pair_index("docs_kb", expected)
    assert loaded is not None
    pairs, emb = loaded
    assert len(pairs) == emb.shape[0]
    assert emb.shape[1] == 384


def test_get_or_build_uses_shipped_without_encode(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(docs_search, "_KB_TEXTS", [])
    monkeypatch.setattr(docs_search, "_KB_EMBEDDINGS", None)
    monkeypatch.setattr(docs_search, "_KB_INDEX_MTIME", (-1.0, -1))

    def boom(*_a, **_k):
        """Raise an assertion error to ensure embedding encoding is not invoked."""
        raise AssertionError("should not encode when shipped index matches")

    monkeypatch.setattr(docs_search, "build_kb_index", boom)
    pairs, emb = docs_search.get_or_build_kb_index()
    assert len(pairs) == emb.shape[0]
    assert emb.shape[0] > 0


def test_ts_olive_shipped_index():
    """Verify that the shipped troubleshooting index contains one embedding for each troubleshooting entry."""
    entries = load_troubleshooting()
    fp = ts._entries_fingerprint(entries)
    emb = load_entry_embeddings("ts_olive", fp)
    assert emb is not None
    assert emb.shape[0] == len(entries)


def test_ts_index_uses_shipped(monkeypatch: pytest.MonkeyPatch):
    ts._ts_index_cache = {}
    entries = load_troubleshooting()

    def boom(*_a, **_k):
        raise AssertionError("should not rebuild ts embeddings when shipped")

    monkeypatch.setattr(ts, "build_kb_index", boom)
    out_entries, emb = ts._get_troubleshooting_index(entries)
    assert len(out_entries) == emb.shape[0]
    assert emb.shape[0] == len(entries)


def test_capabilities_reports_shipped_index():
    caps = get_mcp_capabilities()
    assert caps["kb"]["index"]["shipped"] is True
    assert caps["kb"]["index"]["version"]


def test_hash_mismatch_skips_shipped(monkeypatch: pytest.MonkeyPatch):
    loaded = load_pair_index("docs_kb", "not-the-real-hash")
    assert loaded is None


def test_slo_keyword_troubleshoot_fast():
    """Keyword path should stay well under a second of tool time."""
    import os
    import time

    from olive_mcp_server.mcp_server import call_tool

    t0 = time.perf_counter()
    r = call_tool(
        "troubleshoot_olive_error",
        {"error_message": "CUDA out of memory", "mode": "keyword"},
    )
    ms = (time.perf_counter() - t0) * 1000
    assert r.get("matched_entry") or r.get("title")
    # Wall-clock SLO is opt-in: shared CI runners make hard latency asserts flaky.
    if os.environ.get("OLIVE_MCP_ENFORCE_SLO") == "1":
        assert ms < 2000, f"keyword troubleshoot too slow: {ms:.0f}ms"


def test_slo_catalog_fast():
    import os
    import time

    from olive_mcp_server.mcp_server import call_tool

    t0 = time.perf_counter()
    r = call_tool("get_olive_passes", {"filter": "quantization"})
    ms = (time.perf_counter() - t0) * 1000
    assert r.get("passes")
    if os.environ.get("OLIVE_MCP_ENFORCE_SLO") == "1":
        assert ms < 2000, f"catalog too slow: {ms:.0f}ms"
