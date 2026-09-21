"""Staleness behavior of the shipped-index builder (scripts/build_kb_index.py)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from olive_mcp_server.tools.index_store import read_manifest

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "build_kb_index.py"


class _EncodeAttempted(Exception):
    """Sentinel raised by the fake encoder to prove the build path ran."""


def _load_builder():
    spec = importlib.util.spec_from_file_location("build_kb_index_under_test", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_build_kb_index(_texts):
    raise _EncodeAttempted


def test_skips_build_when_indexes_up_to_date(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("olive_mcp_server.tools.embeddings.build_kb_index", _fake_build_kb_index)
    assert _load_builder().main() == 0


def test_rebuilds_when_manifest_hash_is_stale(monkeypatch: pytest.MonkeyPatch):
    manifest = read_manifest()
    assert manifest is not None
    stale = {
        **manifest,
        "indexes": {stem: {**meta, "content_hash": "stale"} for stem, meta in (manifest.get("indexes") or {}).items()},
    }
    monkeypatch.setattr("olive_mcp_server.tools.index_store.read_manifest", lambda: stale)
    monkeypatch.setattr("olive_mcp_server.tools.embeddings.build_kb_index", _fake_build_kb_index)
    with pytest.raises(_EncodeAttempted):
        _load_builder().main()


def test_force_rebuild_env_overrides_fresh_indexes(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OLIVE_MCP_REBUILD_INDEX", "1")
    monkeypatch.setattr("olive_mcp_server.tools.embeddings.build_kb_index", _fake_build_kb_index)
    with pytest.raises(_EncodeAttempted):
        _load_builder().main()
