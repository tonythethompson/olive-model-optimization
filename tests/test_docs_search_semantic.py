"""Tests for semantic upgrade of search_olive_documentation."""

from __future__ import annotations

import numpy as np
import pytest

from olive_mcp_server.tools import docs_search
from olive_mcp_server.tools.docs_search import search_olive_documentation
from olive_mcp_server.tools.index_store import content_hash_pairs


@pytest.fixture(autouse=True)
def _reset_kb_index_cache():
    """Ensure each test starts with a clean KB embedding cache."""
    docs_search._KB_TEXTS = []
    docs_search._KB_EMBEDDINGS = None
    docs_search._KB_INDEX_MTIME = (-1.0, -1)
    docs_search._LIVE_CACHE = {}
    docs_search._LAST_FETCH_TIME = 0.0
    docs_search._LIVE_FETCH_GENERATION = 0
    docs_search._LIVE_SNIPPETS = []
    docs_search._LIVE_EMBEDDINGS = None
    docs_search._LIVE_EMBED_CACHE_TIME = -1.0
    yield
    docs_search._KB_TEXTS = []
    docs_search._KB_EMBEDDINGS = None
    docs_search._KB_INDEX_MTIME = (-1.0, -1)
    docs_search._LIVE_CACHE = {}
    docs_search._LAST_FETCH_TIME = 0.0
    docs_search._LIVE_FETCH_GENERATION = 0
    docs_search._LIVE_SNIPPETS = []
    docs_search._LIVE_EMBEDDINGS = None
    docs_search._LIVE_EMBED_CACHE_TIME = -1.0


def _install_fake_semantic(
    monkeypatch: pytest.MonkeyPatch,
    *,
    results_for_query: dict[str, list] | None = None,
    default_results: list | None = None,
):
    """Mock semantic_search and index build so tests need no real model."""

    def fake_build(texts):
        n = len(list(texts)) if texts else 0
        return np.zeros((n, 384), dtype=np.float32)

    def fake_semantic(query, kb_texts, kb_embeddings, top_k, threshold=0.30):
        if results_for_query and query in results_for_query:
            return results_for_query[query][:top_k]
        if default_results is not None:
            return default_results[:top_k]
        return []

    monkeypatch.setattr(docs_search, "build_kb_index", fake_build)
    monkeypatch.setattr(docs_search, "semantic_search", fake_semantic)


def test_fuzzy_query_finds_quantization_related(monkeypatch: pytest.MonkeyPatch):
    """Semantic hits for a fuzzy quantization query should surface."""
    fake_hits = [
        {
            "source": "passes.OnnxQuantization.description",
            "snippet": "Static quantization with calibration data.",
            "relevance": 0.72,
        },
        {
            "source": "quirks.quantization[0].title",
            "snippet": "Use representative calibration samples.",
            "relevance": 0.61,
        },
    ]
    _install_fake_semantic(
        monkeypatch,
        default_results=fake_hits,
    )

    result = search_olive_documentation(
        query="how do I reduce model size with int8",
        top_k=3,
        live=False,
    )
    assert result["count"] > 0
    assert result["query"] == "how do I reduce model size with int8"
    sources = " ".join(r["source"] for r in result["results"]).lower()
    snippets = " ".join(r["snippet"] for r in result["results"]).lower()
    assert "quant" in sources or "quant" in snippets or "calibration" in snippets


_FIXED_KB_TEXTS = [
    ("passes.foo", "quantization calibration data"),
    ("passes.bar", "onnx conversion"),
]


def test_keyword_fallback_when_semantic_empty(monkeypatch: pytest.MonkeyPatch):
    """When semantic returns nothing above threshold, keyword fallback runs."""
    monkeypatch.setattr(docs_search, "_load_kb_text", lambda: list(_FIXED_KB_TEXTS))
    _install_fake_semantic(monkeypatch, default_results=[])

    result = search_olive_documentation(
        query="calibration data",
        top_k=3,
        live=False,
    )
    assert result["count"] > 0
    assert len(result["results"]) <= 3
    # Keyword relevance is normalized to [0, 1]
    assert all(isinstance(r["relevance"], (int, float)) for r in result["results"])
    assert all(0.0 < float(r["relevance"]) <= 1.0 for r in result["results"])


def test_keyword_fallback_when_semantic_raises(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(docs_search, "_load_kb_text", lambda: list(_FIXED_KB_TEXTS))

    def boom(*_a, **_k):
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(docs_search, "get_or_build_kb_index", boom)

    result = search_olive_documentation(
        query="calibration data",
        top_k=3,
        live=False,
    )
    assert result["count"] > 0


def test_kb_mtime_invalidation_rebuilds_index(monkeypatch: pytest.MonkeyPatch):
    """Changing the tracked mtime must force a rebuild of embeddings."""
    build_calls: list[int] = []

    sample_texts = [
        ("passes.foo", "quantization calibration"),
        ("passes.bar", "onnx conversion"),
    ]

    def fake_load():
        return list(sample_texts)

    def fake_build(texts):
        build_calls.append(len(list(texts)))
        return np.ones((len(list(texts)), 384), dtype=np.float32)

    def fake_semantic(query, kb_texts, kb_embeddings, top_k, threshold=0.30):
        return [
            {
                "source": kb_texts[0][0],
                "snippet": kb_texts[0][1][:300],
                "relevance": 0.9,
            }
        ][:top_k]

    monkeypatch.setattr(docs_search, "_load_kb_text", fake_load)
    monkeypatch.setattr(docs_search, "build_kb_index", fake_build)
    monkeypatch.setattr(docs_search, "semantic_search", fake_semantic)

    # First build
    monkeypatch.setattr(docs_search, "_kb_max_mtime", lambda: (100.0, 2))
    docs_search._search_local("quantization", 3)
    assert len(build_calls) == 1

    # Same mtime → cache hit
    docs_search._search_local("quantization", 3)
    assert len(build_calls) == 1

    # mtime change → rebuild
    monkeypatch.setattr(docs_search, "_kb_max_mtime", lambda: (200.0, 2))
    docs_search._search_local("quantization", 3)
    assert len(build_calls) == 2


def test_weak_live_result_does_not_displace_strong_local_top1(monkeypatch: pytest.MonkeyPatch):
    """A weak live hit must not evict the single strongest local result at top_k=1."""
    strong_local = [{"source": "passes.foo", "snippet": "strong", "relevance": 0.85}]
    weak_live = [{"source": "live:index", "snippet": "weak", "relevance": 0.31}]

    def fake_local(query, top_k, mode=None, budget_ms=None):
        return strong_local[:top_k], {
            "mode": "auto",
            "effective": "hybrid",
            "degraded": False,
        }

    def fake_live(query, top_k, *, mode=None, budget_ms=None):
        return weak_live[:top_k], {
            "mode": "auto",
            "effective": "hybrid",
            "degraded": False,
        }

    monkeypatch.setattr(docs_search, "_search_local", fake_local)
    monkeypatch.setattr(docs_search, "_search_live", fake_live)

    result = search_olive_documentation(query="quantization", top_k=1, live=True)
    assert result["results"] == [strong_local[0]]


def test_keyword_mode_with_live_does_not_start_semantic_loading(
    monkeypatch: pytest.MonkeyPatch,
):
    """mode=keyword + live=True must never build live embeddings or call semantic_search."""
    fetch_calls = {"n": 0}
    monkeypatch.setattr(docs_search, "_load_kb_text", lambda: list(_FIXED_KB_TEXTS))

    def fake_fetch():
        fetch_calls["n"] += 1
        return ({"index": "live calibration data for static quantization"}, 1.0)

    monkeypatch.setattr(docs_search, "_fetch_live_docs", fake_fetch)
    # Env default must not override an explicit keyword mode on either path.
    monkeypatch.setenv("OLIVE_MCP_RETRIEVAL_MODE", "semantic")

    def boom_semantic(*_a, **_k):
        raise AssertionError("semantic_search must not run under mode=keyword")

    def boom_live_index(*_a, **_k):
        raise AssertionError("_get_live_index must not run under mode=keyword")

    def boom_build(*_a, **_k):
        raise AssertionError("build_kb_index must not run under mode=keyword")

    def boom_kb_index(*_a, **_k):
        raise AssertionError("get_or_build_kb_index must not run under mode=keyword")

    monkeypatch.setattr(docs_search, "semantic_search", boom_semantic)
    monkeypatch.setattr(docs_search, "_get_live_index", boom_live_index)
    monkeypatch.setattr(docs_search, "build_kb_index", boom_build)
    monkeypatch.setattr(docs_search, "get_or_build_kb_index", boom_kb_index)

    result = search_olive_documentation(
        query="calibration",
        top_k=3,
        live=True,
        mode="keyword",
    )
    assert result["retrieval"]["mode"] == "keyword"
    assert result["retrieval"]["effective"] == "keyword"
    assert result["count"] > 0
    assert fetch_calls["n"] == 1
    assert any(r["source"].startswith("live:") for r in result["results"])
    sources = " ".join(r["source"] for r in result["results"])
    # Local keyword and/or live keyword hits expected; no semantic path taken.
    assert "calibration" in sources.lower() or any("calibration" in r["snippet"].lower() for r in result["results"])

    # Direct live helper: keyword path is fetch → split → _keyword_search only.
    live_hits, live_meta = docs_search._search_live("calibration", 3, mode="keyword")
    assert live_meta["mode"] == "keyword"
    assert live_meta["effective"] == "keyword"
    assert live_hits
    assert all(r["source"].startswith("live:") for r in live_hits)
    assert fetch_calls["n"] == 2


def test_empty_query_unchanged():
    result = search_olive_documentation(query="   ", top_k=5, live=False)
    assert result["count"] == 0
    assert result["results"] == []
    assert "Empty query" in result["note"]


def test_return_shape_preserved(monkeypatch: pytest.MonkeyPatch):
    _install_fake_semantic(
        monkeypatch,
        default_results=[
            {"source": "x", "snippet": "y", "relevance": 0.5},
        ],
    )
    result = search_olive_documentation(query="quantization", top_k=2, live=False)
    assert set(result.keys()) >= {"query", "count", "results", "note", "retrieval"}
    for r in result["results"]:
        assert set(r.keys()) >= {"source", "snippet", "relevance"}


def test_kb_stale_build_does_not_poison_cache(monkeypatch: pytest.MonkeyPatch):
    """If mtime advances mid-build, do not stamp stale texts with the new mtime."""
    sample_old = [("old.path", "old calibration text")]
    sample_new = [("new.path", "new calibration text")]
    mtime_box = {"v": 100.0}

    def fake_mtime():
        return (mtime_box["v"], 1)

    def fake_load():
        if mtime_box["v"] >= 200.0:
            return list(sample_new)
        return list(sample_old)

    def fake_build(texts):
        texts = list(texts)
        # Simulate hot-reload while encoding.
        mtime_box["v"] = 200.0
        return np.ones((len(texts), 384), dtype=np.float32)

    monkeypatch.setattr(docs_search, "_kb_max_mtime", fake_mtime)
    monkeypatch.setattr(docs_search, "_load_kb_text", fake_load)
    monkeypatch.setattr(docs_search, "build_kb_index", fake_build)

    texts1, _emb1 = docs_search.get_or_build_kb_index()
    # Stale build returned locally but must not poison global cache at mtime 200.
    assert texts1[0][0] == "old.path"
    # Cache must remain unpublished (still the pre-test sentinel) rather than
    # stamped with the newer mtime for stale content.
    assert docs_search._KB_INDEX_MTIME == (-1.0, -1)
    assert docs_search._KB_TEXTS == []

    # Next call with mtime 200 should load fresh content and cache it.
    texts2, _emb2 = docs_search.get_or_build_kb_index()
    assert texts2[0][0] == "new.path"
    assert docs_search._KB_INDEX_MTIME == (200.0, 1)
    assert docs_search._KB_TEXTS[0][0] == "new.path"


def test_live_fetch_generation_ignores_stale_completion(monkeypatch: pytest.MonkeyPatch):
    """An older in-flight fetch completing after a newer one must not win.

    Drives the real ``_fetch_live_docs`` through two threads so the
    generation-token guard inside the function itself is exercised, not
    re-implemented in the test.
    """
    import threading as _threading

    started_gen1 = _threading.Event()
    release_gen1 = _threading.Event()
    call_i = {"n": 0}

    def fake_fetch(pages=None):
        i = call_i["n"]
        call_i["n"] += 1
        if i == 0:
            # gen1: signal it has started, then block until gen2 finishes.
            started_gen1.set()
            release_gen1.wait(timeout=5)
            return {"status": "ok", "pages": {"index": "OLDER live page content calibration"}}
        return {"status": "ok", "pages": {"index": "NEWER live page content calibration"}}

    import olive_mcp_server.fetchers.official_docs_fetcher as fetcher

    monkeypatch.setattr(fetcher, "fetch_official_docs", fake_fetch)

    result_gen1: dict = {}
    t1 = _threading.Thread(target=lambda: result_gen1.update(pages=docs_search._fetch_live_docs()[0]))
    t1.start()
    assert started_gen1.wait(timeout=5)

    # gen2 runs to completion while gen1 is still blocked.
    pages_gen2, _ = docs_search._fetch_live_docs()
    assert "NEWER" in next(iter(pages_gen2.values()))

    # Now let the stale gen1 completion land; it must not clobber gen2's cache.
    release_gen1.set()
    t1.join(timeout=5)
    assert not t1.is_alive(), "stale generation-1 fetch did not complete"
    assert result_gen1.get("pages"), "generation-1 thread raised before returning"

    assert "NEWER" in next(iter(docs_search._LIVE_CACHE.values()))


def test_live_index_does_not_overwrite_newer_generation(monkeypatch: pytest.MonkeyPatch):
    """A stale (older fetch_time) live-index build must not clobber a newer one.

    Simulates: build for fetch_time=100 starts, a newer fetch_time=200 build
    completes and publishes first, then the stale build's publish attempt
    must be discarded rather than roll the cache back to fetch_time=100.
    """

    def fake_build(texts):
        return np.ones((len(list(texts)), 384), dtype=np.float32)

    monkeypatch.setattr(docs_search, "build_kb_index", fake_build)

    # Newer generation publishes first.
    monkeypatch.setattr(docs_search, "_fetch_live_docs", lambda: ({"index": "newer content"}, 200.0))
    docs_search._get_live_index()
    assert docs_search._LIVE_EMBED_CACHE_TIME == 200.0

    # Stale generation (older fetch_time) attempts to publish after.
    monkeypatch.setattr(docs_search, "_fetch_live_docs", lambda: ({"index": "older content"}, 100.0))
    docs_search._get_live_index()
    assert docs_search._LIVE_EMBED_CACHE_TIME == 200.0
    assert docs_search._LIVE_SNIPPETS[0][1] == "newer content"


def test_load_kb_text_skips_invalid_utf8_and_keeps_valid_files(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """One undecodable KB file must not abort loading of sibling valid JSON."""
    bad = tmp_path / "bad_utf8.json"
    good = tmp_path / "good.json"
    bad.write_bytes(b'{"oops": "\xff\xfe invalid"}')
    good.write_text('{"pass": "OnnxQuantization works"}', encoding="utf-8")

    monkeypatch.setattr(docs_search, "KB_DIR", tmp_path)
    loaded = docs_search._load_kb_text()
    sources = " ".join(path for path, _ in loaded)
    snippets = " ".join(text for _, text in loaded)
    assert "good" in sources
    assert "OnnxQuantization" in snippets
    assert "bad_utf8" not in sources


def test_generator_sidecars_excluded_from_kb_hash(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """Generator-written sidecars must not change the searchable KB content.

    update_kb.py / expand_kb.py write ``refresh_metadata.json`` (plus the
    already-excluded report files) into the KB directory at refresh time.
    If any of them leak into ``_load_kb_text()``, the content hash drifts
    from the shipped index and semantic search rebuilds on every run.
    """
    (tmp_path / "passes.json").write_text('{"pass": "OnnxQuantization"}', encoding="utf-8")

    monkeypatch.setattr(docs_search, "KB_DIR", tmp_path)
    baseline_pairs = docs_search._load_kb_text()
    baseline_hash = content_hash_pairs(baseline_pairs)

    # Simulate a KB refresh run dropping its sidecars next to the KB files.
    # String leaves mirror the real refresh_metadata.json (generator, timestamps,
    # fingerprints) so the content hash would drift if the file were indexed.
    (tmp_path / "refresh_metadata.json").write_text(
        '{"schema_version": 1, "generator_version": "0.5.0",'
        ' "source_timestamp": "2026-08-12T21:00:51Z",'
        ' "runs": {"update_kb": {"generator": "update_kb", "success": true}}}',
        encoding="utf-8",
    )
    (tmp_path / "update_report.json").write_text('{"generator": "update_kb"}', encoding="utf-8")
    (tmp_path / "candidate_quirks.json").write_text('[{"title": "candidate"}]', encoding="utf-8")

    after_pairs = docs_search._load_kb_text()
    assert content_hash_pairs(after_pairs) == baseline_hash
    sources = " ".join(path for path, _ in after_pairs)
    assert "refresh_metadata" not in sources
    assert "update_report" not in sources
    assert "candidate_quirks" not in sources


def test_iter_kb_json_files_rejects_symlinks_outside_kb_dir(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """Path traversal guard: symlinks resolving outside KB_DIR are rejected."""
    import os

    kb_dir = tmp_path / "kb"
    kb_dir.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    secret = outside / "secret.json"
    secret.write_text('{"leaked": true}', encoding="utf-8")

    # Create a symlink inside kb_dir that points outside
    link = kb_dir / "evil.json"
    try:
        os.symlink(str(secret), str(link))
    except OSError:
        pytest.skip("Symlink creation not permitted on this OS/config")

    # Also add a legitimate file
    legit = kb_dir / "passes.json"
    legit.write_text('{"pass": "safe"}', encoding="utf-8")

    monkeypatch.setattr(docs_search, "KB_DIR", kb_dir)
    loaded = docs_search._load_kb_text()
    sources = " ".join(path for path, _ in loaded)
    assert "passes" in sources
    assert "evil" not in sources
    assert "leaked" not in " ".join(text for _, text in loaded)


def test_search_query_sanitization(monkeypatch: pytest.MonkeyPatch):
    """Null bytes and excessively long queries are sanitized."""
    monkeypatch.setattr(docs_search, "_load_kb_text", lambda: [("p", "hello world")])
    monkeypatch.setattr(docs_search, "_KB_TEXTS", [])
    monkeypatch.setattr(docs_search, "_KB_EMBEDDINGS", None)
    monkeypatch.setattr(docs_search, "_KB_INDEX_MTIME", (-1.0, -1))

    # Query with null bytes should not crash and should be cleaned
    result = docs_search.search_olive_documentation(query="quant\x00ization", top_k=1, live=False, mode="keyword")
    assert result["query"] == "quantization"

    # Whitespace controls should become spaces (preserve token boundaries)
    for control in ("\n", "\t", "\r", "\x0b", "\x0c"):
        result_ws = docs_search.search_olive_documentation(
            query=f"quantization{control}calibration",
            top_k=1,
            live=False,
            mode="keyword",
        )
        assert result_ws["query"] == "quantization calibration"

    # Very long query should be truncated to 2000 chars (not crash)
    long_query = "x" * 5000
    result2 = docs_search.search_olive_documentation(query=long_query, top_k=1, live=False, mode="keyword")
    assert len(result2["query"]) == 2000

    # Truncation applies before control normalization ends; length stays <= 2000
    long_with_controls = ("a\tb\nc\rd\x0be\x0cf") * 400
    result3 = docs_search.search_olive_documentation(query=long_with_controls, top_k=1, live=False, mode="keyword")
    assert len(result3["query"]) == 2000
    assert "\t" not in result3["query"]
    assert "\n" not in result3["query"]
    assert "\r" not in result3["query"]
    assert "\x0b" not in result3["query"]
    assert "\x0c" not in result3["query"]


def test_live_auto_budgets_even_when_model_is_warm(monkeypatch: pytest.MonkeyPatch, wait_inflight_semantic_clear):
    """Warm model must not skip the auto budget: live fetch/index can still be cold."""
    import time

    from olive_mcp_server.tools.retrieval import retrieval_meta

    monkeypatch.setenv("OLIVE_MCP_SEMANTIC_BUDGET_MS", "50")

    # Keep local out of the shared budget pool so live owns the single-flight slot.
    monkeypatch.setattr(
        docs_search,
        "_search_local",
        lambda *_a, **_k: (
            [{"source": "local.kb", "snippet": "local calibration note", "relevance": 0.4}],
            retrieval_meta(
                mode="auto",
                effective="keyword",
                degraded=True,
                reason="semantic_budget_exceeded",
            ),
        ),
    )

    live_index_calls = {"n": 0}

    def slow_live_index():
        live_index_calls["n"] += 1
        time.sleep(0.5)
        return (
            [("live:index", "live calibration for static quantization")],
            np.zeros((1, 384), dtype=np.float32),
        )

    monkeypatch.setattr(docs_search, "_get_live_index", slow_live_index)
    monkeypatch.setattr(
        docs_search,
        "_fetch_live_docs",
        lambda: ({"index": "live calibration for static quantization"}, 1.0),
    )
    # The background worker keeps running past the 50ms budget (it isn't
    # cancelled, only raced against). Once slow_live_index() returns it would
    # otherwise call the real semantic_search(), which lazily loads the actual
    # embedding model on first use — a real network/disk-bound operation that
    # can outlive the shared single-flight slot well past this test (and the
    # conftest inflight-drain fixture's ceiling) on a cold cache. Stub it so
    # the budgeted background work finishes deterministically and fast.
    monkeypatch.setattr(docs_search, "semantic_search", lambda *_a, **_k: [])

    try:
        result = search_olive_documentation(
            query="calibration",
            top_k=3,
            live=True,
            mode="auto",
        )

        assert result["retrieval"]["mode"] == "auto"
        assert result["retrieval"]["effective"] == "keyword"
        assert result["retrieval"]["degraded"] is True
        assert live_index_calls["n"] == 1
        assert any(r["source"].startswith("live:") for r in result["results"])
    finally:
        # The abandoned background worker (still executing slow_live_index()
        # past the 50ms budget) must finish while the monkeypatches above are
        # still in scope — otherwise pytest's monkeypatch teardown restores
        # the real semantic_search() before the worker gets to it, racing a
        # genuine embedding-model load against test teardown. Drain it here
        # even if an assertion above failed, so a bug in this test can't also
        # leak a stuck worker into whichever test runs next.
        wait_inflight_semantic_clear()

    # After timeout, further live keyword search must avoid embeddings.
    def boom_if_keyword_hits_index(*_a, **_k):
        raise AssertionError("keyword fallback must not call _get_live_index")

    monkeypatch.setattr(docs_search, "_get_live_index", boom_if_keyword_hits_index)
    again, again_meta = docs_search._search_live("calibration", 3, mode="keyword")
    assert again
    assert all(r["source"].startswith("live:") for r in again)
    assert again_meta.get("effective") == "keyword"


def test_shared_semantic_budget_zero_remaining_forces_live_keyword(
    monkeypatch: pytest.MonkeyPatch,
):
    """When local consumes the shared deadline, live gets budget_ms=0 → keyword only."""
    import time

    from olive_mcp_server.tools.retrieval import retrieval_meta

    monkeypatch.setenv("OLIVE_MCP_SEMANTIC_BUDGET_MS", "80")

    def slow_local(query, top_k, *, mode=None, budget_ms=None):
        assert budget_ms is not None and budget_ms > 0
        time.sleep(0.12)  # exceed shared 80ms deadline
        return (
            [{"source": "local.kb", "snippet": "local calibration", "relevance": 0.5}],
            retrieval_meta(mode="auto", effective="hybrid"),
        )

    live_budgets: list[int | None] = []
    live_index_calls = {"n": 0}
    real_live = docs_search._search_live

    def tracking_real_live(query, top_k, *, mode=None, budget_ms=None):
        live_budgets.append(budget_ms)
        return real_live(query, top_k, mode=mode, budget_ms=budget_ms)

    def boom_live_index(*_a, **_k):
        live_index_calls["n"] += 1
        raise AssertionError("budget_ms=0 must not build live embeddings")

    monkeypatch.setattr(docs_search, "_search_local", slow_local)
    monkeypatch.setattr(docs_search, "_search_live", tracking_real_live)
    monkeypatch.setattr(docs_search, "_get_live_index", boom_live_index)
    monkeypatch.setattr(
        docs_search,
        "_fetch_live_docs",
        lambda: ({"index": "live calibration for static quantization"}, 1.0),
    )

    result = search_olive_documentation(
        query="calibration",
        top_k=3,
        live=True,
        mode="auto",
    )

    assert live_budgets == [0]
    assert live_index_calls["n"] == 0
    assert result["retrieval"]["degraded"] is True
    assert result["retrieval"]["reason"] == "semantic_budget_exceeded"
    assert any(r["source"].startswith("live:") for r in result["results"])


def test_explicit_budget_ms_zero_selects_keyword_not_unlimited():
    """Explicit budget_ms=0 must not call run_with_budget(0) (unlimited)."""
    hits, meta = docs_search._search_live(
        "calibration",
        3,
        mode="auto",
        budget_ms=0,
    )
    assert meta["effective"] == "keyword"
    assert meta["degraded"] is True
    assert meta["reason"] == "semantic_budget_exceeded"

    hits_l, meta_l = docs_search._search_local(
        "calibration",
        3,
        mode="auto",
        budget_ms=0,
    )
    assert meta_l["effective"] == "keyword"
    assert meta_l["degraded"] is True
    assert meta_l["reason"] == "semantic_budget_exceeded"
    assert isinstance(hits_l, list)
    assert isinstance(hits, list)
