"""Shared embedding utilities for semantic knowledge-base search.

Uses BAAI/bge-small-en-v1.5 (384-dim, ~130MB, CPU-only, retrieval-tuned).
The model is loaded lazily on first encode call, not at import time.

BGE models require a query instruction prefix for retrieval (s2p) tasks:
  "Represent this sentence for searching relevant passages: "
This prefix is applied to queries in ``encode_query`` but NOT to indexed
passages in ``encode_texts`` / ``build_kb_index``.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384
DEFAULT_THRESHOLD = 0.30
# BGE retrieval query instruction (s2p). Prepended to queries only, not passages.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

_model = None
_model_lock = threading.Lock()


def _get_model() -> SentenceTransformer | None:
    """Thread-safe lazy singleton for SentenceTransformer (CPU-only)."""
    global _model
    if _model is not None:
        return _model
    with _model_lock:
        if _model is None:
            from sentence_transformers import SentenceTransformer

            _model = SentenceTransformer(MODEL_NAME, device="cpu")
        return _model


def is_model_loaded() -> bool:
    """Return True if the embedding model has already been loaded."""
    return _model is not None


def encode_texts(texts: Iterable[str]) -> np.ndarray:
    """Encode a batch of texts into an (N, 384) float32 matrix.

    Rows are L2-normalized (``normalize_embeddings=True``). Callers may still
    pass the matrix through ``cosine_similarity_scores``, which re-normalizes
    safely for zero rows and non-normalized inputs.
    """
    texts_list = list(texts) if texts is not None else []
    if not texts_list:
        return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
    model = _get_model()
    embeddings = model.encode(
        texts_list,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    return np.asarray(embeddings, dtype=np.float32)


def encode_query(query: str) -> np.ndarray:
    """Encode a single query string into a (384,) float32 vector.

    Prepends the BGE retrieval instruction prefix so the query embedding
    aligns with passage embeddings from ``encode_texts`` / ``build_kb_index``
    (which do NOT receive the prefix).
    """
    if not query or not str(query).strip():
        return np.zeros((EMBEDDING_DIM,), dtype=np.float32)
    return encode_texts([QUERY_INSTRUCTION + str(query)])[0]


def cosine_similarity_scores(
    query_vec: np.ndarray,
    index_matrix: np.ndarray,
) -> np.ndarray:
    """L2-normalized cosine similarity of query_vec against each index row.

    Returns a 1-D float32 array of length equal to the number of index rows.
    Empty query or empty index yields an empty or zero-length array.
    """
    if index_matrix is None or getattr(index_matrix, "size", 0) == 0:
        return np.zeros((0,), dtype=np.float32)
    if query_vec is None or getattr(query_vec, "size", 0) == 0:
        n = int(index_matrix.shape[0]) if getattr(index_matrix, "ndim", 0) >= 1 else 0
        return np.zeros((n,), dtype=np.float32)

    q = np.asarray(query_vec, dtype=np.float32).reshape(-1)
    mat = np.asarray(index_matrix, dtype=np.float32)
    if mat.ndim == 1:
        mat = mat.reshape(1, -1)

    q_norm = float(np.linalg.norm(q))
    if q_norm == 0.0:
        return np.zeros((mat.shape[0],), dtype=np.float32)
    q = q / q_norm

    row_norms = np.linalg.norm(mat, axis=1, keepdims=True)
    # Avoid division by zero for zero rows.
    safe_norms = np.where(row_norms == 0.0, 1.0, row_norms)
    mat_normed = mat / safe_norms
    # Zero rows stay zero after this (they had norm 0 and were divided by 1,
    # but original zeros remain zeros).

    scores = mat_normed @ q
    return np.asarray(scores, dtype=np.float32)


def build_kb_index(kb_texts: Sequence[str]) -> np.ndarray:
    """Batch-encode all KB entry texts into an L2-normalized embedding matrix.

    Contract: returned rows are unit-length (except all-zero empty index).
    Compatible with ``cosine_similarity_scores`` which also L2-normalizes.
    """
    return encode_texts(list(kb_texts) if kb_texts else [])


def _entry_source_and_text(entry: Any) -> tuple[str, str]:
    """Normalize a KB entry into (source, text).

    Accepts (source, text) pairs or dicts with source/text/snippet keys.
    """
    if isinstance(entry, dict):
        source = str(entry.get("source", ""))
        text = str(entry.get("text") or entry.get("snippet") or "")
        return source, text
    if isinstance(entry, (tuple, list)) and len(entry) >= 2:
        return str(entry[0]), str(entry[1])
    return "", str(entry)


def semantic_search(
    query: str,
    kb_texts: Sequence[Any],
    kb_embeddings: np.ndarray,
    top_k: int,
    threshold: float = DEFAULT_THRESHOLD,
) -> list[dict[str, Any]]:
    """Rank KB entries by cosine similarity to the query.

    Returns ``[{source, snippet, relevance}, ...]`` filtered by *threshold*,
    sorted by relevance descending, limited to *top_k*.
    """
    if top_k <= 0 or not kb_texts:
        return []
    if kb_embeddings is None or getattr(kb_embeddings, "size", 0) == 0:
        return []
    if not query or not str(query).strip():
        return []

    query_vec = encode_query(str(query))
    scores = cosine_similarity_scores(query_vec, kb_embeddings)
    if scores.size == 0:
        return []

    n = min(len(kb_texts), int(scores.shape[0]))
    results: list[dict[str, Any]] = []
    for i in range(n):
        score = float(scores[i])
        if score < threshold:
            continue
        source, text = _entry_source_and_text(kb_texts[i])
        results.append(
            {
                "source": source,
                "snippet": text[:300],
                "relevance": score,
            }
        )

    results.sort(key=lambda x: x["relevance"], reverse=True)
    return results[:top_k]
