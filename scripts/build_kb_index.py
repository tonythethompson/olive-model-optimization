#!/usr/bin/env python3
"""Build shipped KB embedding indexes (Phase 1).

Run from repo root or olive-mcp-server with the project venv::

  olive-mcp-server/.venv/Scripts/python olive-mcp-server/scripts/build_kb_index.py

Requires sentence-transformers. Writes under olive_mcp_server/knowledge_base/indexes/.

Skips the build when the shipped indexes already match the current KB content
(same model, content hashes, and index files present); set
OLIVE_MCP_REBUILD_INDEX=1 to force a rebuild.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parent.parent
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))


def main() -> int:
    """
    Build and save embedding indexes for documentation and troubleshooting knowledge bases.

    Returns:
        int: Exit code `0` after all indexes and the manifest are written successfully.
    """
    from olive_mcp_server.tools import load_studio_troubleshooting, load_troubleshooting
    from olive_mcp_server.tools.docs_search import _load_kb_text
    from olive_mcp_server.tools.embeddings import MODEL_NAME, build_kb_index
    from olive_mcp_server.tools.index_store import (
        _meta_path,
        _npz_path,
        content_hash_pairs,
        force_rebuild,
        read_manifest,
        save_entry_index,
        save_pair_index,
        write_manifest,
    )
    from olive_mcp_server.tools.troubleshooting import _entries_fingerprint, _entry_embed_text

    pairs = _load_kb_text()
    expected_hashes = {"docs_kb": content_hash_pairs(pairs)}
    for stem, loader in (
        ("ts_olive", load_troubleshooting),
        ("ts_studio", load_studio_troubleshooting),
    ):
        expected_hashes[stem] = _entries_fingerprint(list(loader()))

    if not force_rebuild():
        manifest = read_manifest()
        indexes = (manifest or {}).get("indexes") or {}
        up_to_date = bool(manifest) and manifest.get("model") == MODEL_NAME
        for stem, chash in expected_hashes.items():
            entry = indexes.get(stem) or {}
            if entry.get("content_hash") != chash:
                up_to_date = False
                break
            if not _meta_path(stem).is_file() or not _npz_path(stem).is_file():
                up_to_date = False
                break
        if up_to_date:
            print("Indexes already up to date (content hashes match); nothing to build.")
            return 0

    t0 = time.perf_counter()
    print(f"Building indexes with model={MODEL_NAME} …")

    print(f"  docs_kb: {len(pairs)} snippets")
    docs_emb = build_kb_index([t for _, t in pairs])
    docs_meta = save_pair_index("docs_kb", pairs, docs_emb)
    print(f"  docs_kb hash={docs_meta['content_hash'][:12]}…")

    manifest_indexes: dict = {
        "docs_kb": {
            "content_hash": docs_meta["content_hash"],
            "count": docs_meta["count"],
        }
    }

    for stem, loader in (
        ("ts_olive", load_troubleshooting),
        ("ts_studio", load_studio_troubleshooting),
    ):
        entries = list(loader())
        texts = [_entry_embed_text(e) for e in entries]
        ids = [str(e.get("id") or "") for e in entries]
        chash = _entries_fingerprint(entries)
        print(f"  {stem}: {len(entries)} entries")
        emb = build_kb_index(texts) if texts else build_kb_index([])
        meta = save_entry_index(
            stem,
            ids,
            texts,
            emb,
            content_hash=chash,
        )
        manifest_indexes[stem] = {
            "content_hash": meta["content_hash"],
            "count": meta["count"],
        }
        print(f"  {stem} hash={meta['content_hash'][:12]}…")

    path = write_manifest(manifest_indexes)
    elapsed = time.perf_counter() - t0
    print(f"Wrote {path} in {elapsed:.1f}s")
    assert content_hash_pairs(pairs) == docs_meta["content_hash"]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
