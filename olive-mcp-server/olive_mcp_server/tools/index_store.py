"""Shipped precomputed embedding indexes for the Olive MCP knowledge base.

Phase 1: document embeddings are built at package/release time so cold
processes only load arrays + embed the query (when semantic is used).

Layout under ``knowledge_base/indexes/``::

    manifest.json
    docs_kb.npz          # embeddings (N, dim) float32
    docs_kb.meta.json    # sources, texts, content_hash, model
    ts_olive.npz / ts_olive.meta.json
    ts_studio.npz / ts_studio.meta.json
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

from olive_mcp_server.tools import KB_DIR
from olive_mcp_server.tools.embeddings import EMBEDDING_DIM, MODEL_NAME

logger = logging.getLogger(__name__)

INDEX_DIR = KB_DIR / "indexes"
MANIFEST_NAME = "manifest.json"


def _truthy(name: str) -> bool:
    """Determine whether an environment variable represents an enabled value.

    Parameters:
        name (str): Name of the environment variable.

    Returns:
        bool: `true` if the value is `1`, `true`, `yes`, or `on`, ignoring surrounding
        whitespace and letter case; `false` otherwise.
    """
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def force_rebuild() -> bool:
    """
    Determine whether runtime index rebuilding is enabled.

    Returns:
        bool: `True` if shipped indexes should be ignored and rebuilt, `False` otherwise.
    """
    return _truthy("OLIVE_MCP_REBUILD_INDEX")


def _normalize_text(text: str) -> str:
    """Normalize line endings so Windows-built indexes match Linux hashes."""
    return str(text).replace("\r\n", "\n").replace("\r", "\n")


def content_hash_pairs(pairs: list[tuple[str, str]]) -> str:
    """SHA-256 of ordered (source, text) pairs (LF-normalized text)."""
    h = hashlib.sha256()
    for source, text in pairs:
        h.update(str(source).encode("utf-8"))
        h.update(b"\0")
        h.update(_normalize_text(text).encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def content_hash_texts(texts: list[str]) -> str:
    """
    Compute a SHA-256 hash for an ordered sequence of text values.

    Parameters:
        texts (list[str]): Text values whose normalized contents are hashed.

    Returns:
        str: The hexadecimal SHA-256 digest.
    """
    h = hashlib.sha256()
    for t in texts:
        h.update(_normalize_text(t).encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def ensure_index_dir() -> Path:
    """
    Ensure the directory used for embedding indexes exists.

    Returns:
        Path: The embedding index directory.
    """
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    return INDEX_DIR


def _meta_path(stem: str) -> Path:
    """Return the metadata file path for an index stem.

    Parameters:
        stem (str): Index name used to construct the metadata filename.

    Returns:
        Path: Path to the index metadata file.
    """
    return INDEX_DIR / f"{stem}.meta.json"


def _npz_path(stem: str) -> Path:
    """Build the path to an index's compressed NumPy data file.

    Parameters:
        stem (str): Index filename stem.

    Returns:
        Path: Path to the index's `.npz` file.
    """
    return INDEX_DIR / f"{stem}.npz"


def save_pair_index(
    stem: str,
    pairs: list[tuple[str, str]],
    embeddings: np.ndarray,
    *,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Persist source-text pairs and their embeddings as a document index.

    Parameters:
        pairs (list[tuple[str, str]]): Source and text values aligned with the embedding rows.
        embeddings (np.ndarray): Embedding vectors corresponding to the pairs.
        extra (dict[str, Any] | None): Additional metadata to include in the index metadata.

    Returns:
        dict[str, Any]: Metadata describing the saved index.
    """
    ensure_index_dir()
    emb = np.asarray(embeddings, dtype=np.float32)
    if emb.ndim == 1:
        emb = emb.reshape(1, -1)
    if len(pairs) != emb.shape[0]:
        raise ValueError(f"pairs/embeddings length mismatch: {len(pairs)} vs {emb.shape[0]}")

    chash = content_hash_pairs(pairs)
    meta: dict[str, Any] = {
        "stem": stem,
        "kind": "pairs",
        "model": MODEL_NAME,
        "dim": int(emb.shape[1]) if emb.size else EMBEDDING_DIM,
        "count": len(pairs),
        "content_hash": chash,
        "sources": [p[0] for p in pairs],
        "texts": [p[1] for p in pairs],
    }
    if extra:
        meta.update(extra)

    np.savez_compressed(_npz_path(stem), embeddings=emb)
    _meta_path(stem).write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return meta


def save_entry_index(
    stem: str,
    entry_ids: list[str],
    embed_texts: list[str],
    embeddings: np.ndarray,
    *,
    content_hash: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Persist aligned entry identifiers, embedding texts, and embeddings for a troubleshooting index.

    Parameters:
        stem (str): Index name used to derive the stored metadata and embedding paths.
        entry_ids (list[str]): Identifiers corresponding to each embedding row.
        embed_texts (list[str]): Texts corresponding to each embedding row.
        embeddings (np.ndarray): Embedding vectors aligned with `entry_ids` and `embed_texts`.
        content_hash (str): Hash identifying the source content represented by the index.
        extra (dict[str, Any] | None): Additional metadata to include in the stored index metadata.

    Returns:
        dict[str, Any]: Metadata written for the index.
    """
    ensure_index_dir()
    emb = np.asarray(embeddings, dtype=np.float32)
    if emb.ndim == 1:
        emb = emb.reshape(1, -1)
    if not (len(entry_ids) == len(embed_texts) == emb.shape[0]):
        raise ValueError("entry_ids, embed_texts, embeddings must share length")

    meta: dict[str, Any] = {
        "stem": stem,
        "kind": "entries",
        "model": MODEL_NAME,
        "dim": int(emb.shape[1]) if emb.size else EMBEDDING_DIM,
        "count": len(entry_ids),
        "content_hash": content_hash,
        "entry_ids": entry_ids,
        "embed_texts": embed_texts,
    }
    if extra:
        meta.update(extra)

    np.savez_compressed(_npz_path(stem), embeddings=emb)
    _meta_path(stem).write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return meta


def load_pair_index(stem: str, expected_hash: str) -> tuple[list[tuple[str, str]], np.ndarray] | None:
    """
    Load a document index when its model, content hash, and embedding data are valid.

    Parameters:
        stem (str): Index name used to locate the stored files.
        expected_hash (str): Content hash required for the index to be accepted.

    Returns:
        tuple[list[tuple[str, str]], np.ndarray] | None: The source-text pairs and
        embeddings, or `None` if the index is unavailable or invalid.
    """
    if force_rebuild():
        return None
    meta_p, npz_p = _meta_path(stem), _npz_path(stem)
    if not meta_p.is_file() or not npz_p.is_file():
        return None
    try:
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        if meta.get("model") != MODEL_NAME:
            logger.info("Shipped index %s model mismatch; rebuild", stem)
            return None
        if meta.get("content_hash") != expected_hash:
            logger.info("Shipped index %s content hash mismatch; rebuild", stem)
            return None
        data = np.load(npz_p, allow_pickle=False)
        emb = np.asarray(data["embeddings"], dtype=np.float32)
        if emb.ndim != 2 or emb.shape[1] != EMBEDDING_DIM:
            logger.warning(
                "Shipped index %s bad embedding shape %s (want rank-2, dim=%s); rebuild",
                stem,
                emb.shape,
                EMBEDDING_DIM,
            )
            return None
        sources = meta.get("sources") or []
        texts = meta.get("texts") or []
        if len(sources) != len(texts) or len(sources) != emb.shape[0]:
            logger.warning("Shipped index %s corrupted lengths; rebuild", stem)
            return None
        pairs = list(zip(sources, texts, strict=True))
        return pairs, emb
    except Exception:
        logger.warning("Failed to load shipped index %s", stem, exc_info=True)
        return None


def load_entry_embeddings(stem: str, expected_hash: str) -> np.ndarray | None:
    """
    Load embeddings from an entry index when its metadata matches the configured model and content hash.

    Parameters:
        stem (str): Index name used to locate the stored files.
        expected_hash (str): Content hash required for the index to be valid.

    Returns:
        np.ndarray | None: A two-dimensional float32 embedding array, or `None` when
        the index is unavailable, incompatible, invalid, or rebuilding is forced.
    """
    if force_rebuild():
        return None
    meta_p, npz_p = _meta_path(stem), _npz_path(stem)
    if not meta_p.is_file() or not npz_p.is_file():
        return None
    try:
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        if meta.get("model") != MODEL_NAME:
            return None
        if meta.get("content_hash") != expected_hash:
            return None
        data = np.load(npz_p, allow_pickle=False)
        emb = np.asarray(data["embeddings"], dtype=np.float32)
        if emb.ndim != 2 or emb.shape[1] != EMBEDDING_DIM:
            logger.warning(
                "Shipped entry index %s bad embedding shape %s (want rank-2, dim=%s); rebuild",
                stem,
                emb.shape,
                EMBEDDING_DIM,
            )
            return None
        if int(meta.get("count", -1)) != emb.shape[0]:
            return None
        return emb
    except Exception:
        logger.warning("Failed to load shipped entry index %s", stem, exc_info=True)
        return None


def write_manifest(entries: dict[str, dict[str, Any]]) -> Path:
    """
    Write index metadata and embedding configuration to the manifest file.

    Parameters:
        entries (dict[str, dict[str, Any]]): Metadata for the stored indexes.

    Returns:
        Path: Path to the written manifest file.
    """
    ensure_index_dir()
    path = INDEX_DIR / MANIFEST_NAME
    payload = {
        "model": MODEL_NAME,
        "dim": EMBEDDING_DIM,
        "indexes": entries,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def read_manifest() -> dict[str, Any] | None:
    """
    Read the index manifest from its configured location.

    Returns:
        dict[str, Any] | None: The parsed manifest, or `None` if the file is unavailable or invalid.
    """
    path = INDEX_DIR / MANIFEST_NAME
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def shipped_index_status() -> dict[str, Any]:
    """
    Summarize the available precomputed embedding indexes.

    Returns:
        dict[str, Any]: A status dictionary containing the index metadata version,
        whether indexes are available, their sorted stems, and the configured model.
    """
    manifest = read_manifest()
    if not manifest:
        return {"version": None, "shipped": False, "stems": []}
    indexes = manifest.get("indexes") or {}
    stems = sorted(indexes.keys())
    version = hashlib.sha256(json.dumps(indexes, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return {
        "version": version,
        "shipped": bool(stems),
        "stems": stems,
        "model": manifest.get("model"),
    }
