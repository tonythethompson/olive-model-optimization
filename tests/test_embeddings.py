"""Unit tests for olive_mcp_server.tools.embeddings."""

from __future__ import annotations

import numpy as np
import pytest

from olive_mcp_server.tools import embeddings as emb


def test_lazy_loading_model_not_loaded_at_import():
    """Importing the module must not load the SentenceTransformer model."""
    import subprocess
    import sys

    # Fresh interpreter: import package graph without calling encode.
    code = (
        "from olive_mcp_server.tools import embeddings as e; "
        "from olive_mcp_server.tools import docs_search as d; "  # noqa: F841
        "assert e.is_model_loaded() is False, 'model loaded at import'"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert hasattr(emb, "_model")
    assert callable(emb.is_model_loaded)


def test_cosine_identity():
    """Identical vectors should have cosine similarity ~1.0."""
    v = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    mat = np.stack([v, v * 2.0])  # second row is collinear → also ~1.0
    scores = emb.cosine_similarity_scores(v, mat)
    assert scores.shape == (2,)
    assert scores[0] == pytest.approx(1.0, abs=1e-5)
    assert scores[1] == pytest.approx(1.0, abs=1e-5)


def test_cosine_orthogonal():
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    scores = emb.cosine_similarity_scores(a, b)
    assert scores[0] == pytest.approx(0.0, abs=1e-5)
    assert scores[1] == pytest.approx(1.0, abs=1e-5)


def test_cosine_empty_index():
    q = np.ones((4,), dtype=np.float32)
    empty = np.zeros((0, 4), dtype=np.float32)
    scores = emb.cosine_similarity_scores(q, empty)
    assert scores.shape == (0,)


def test_cosine_zero_query():
    q = np.zeros((3,), dtype=np.float32)
    mat = np.ones((2, 3), dtype=np.float32)
    scores = emb.cosine_similarity_scores(q, mat)
    assert scores.shape == (2,)
    assert np.allclose(scores, 0.0)


def test_threshold_filtering():
    """semantic_search must drop entries below threshold."""
    kb_texts = [
        ("src_a", "alpha document about apples"),
        ("src_b", "beta document about bananas"),
        ("src_c", "gamma document about grapes"),
    ]
    # Hand-crafted embeddings: query aligns strongly with row 0 only.
    # Note: L2-normalization is applied before cosine, so collinear rows
    # (e.g. [0.2,0,0] vs [1,0,0]) would both score 1.0 — use true angles.
    index = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.3, 0.9539, 0.0],  # cosine with [1,0,0] ≈ 0.30 < 0.50 threshold
        ],
        dtype=np.float32,
    )

    def fake_encode_query(_query: str) -> np.ndarray:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)

    original = emb.encode_query
    emb.encode_query = fake_encode_query  # type: ignore[assignment]
    try:
        results = emb.semantic_search(
            "apples",
            kb_texts,
            index,
            top_k=5,
            threshold=0.50,
        )
    finally:
        emb.encode_query = original  # type: ignore[assignment]

    assert len(results) == 1
    assert results[0]["source"] == "src_a"
    assert results[0]["relevance"] == pytest.approx(1.0, abs=1e-5)
    assert "snippet" in results[0]


def test_semantic_search_empty_index():
    results = emb.semantic_search(
        "anything",
        [],
        np.zeros((0, 384), dtype=np.float32),
        top_k=5,
    )
    assert results == []


def test_semantic_search_empty_query():
    kb_texts = [("s", "text")]
    index = np.ones((1, 4), dtype=np.float32)
    assert emb.semantic_search("", kb_texts, index, top_k=3) == []
    assert emb.semantic_search("   ", kb_texts, index, top_k=3) == []


def test_encode_texts_empty():
    out = emb.encode_texts([])
    assert out.shape == (0, emb.EMBEDDING_DIM)
    assert out.dtype == np.float32


def test_encode_shape_mocked(monkeypatch: pytest.MonkeyPatch):
    """encode_texts returns (N, 384) when the model is mocked."""

    class FakeModel:
        def encode(self, texts, **_kwargs):
            return np.random.randn(len(texts), emb.EMBEDDING_DIM).astype(np.float32)

    monkeypatch.setattr(emb, "_model", FakeModel())
    out = emb.encode_texts(["hello", "world"])
    assert out.shape == (2, emb.EMBEDDING_DIM)
    assert out.dtype == np.float32

    q = emb.encode_query("hello")
    assert q.shape == (emb.EMBEDDING_DIM,)


def test_build_kb_index_mocked(monkeypatch: pytest.MonkeyPatch):
    class FakeModel:
        def encode(self, texts, **_kwargs):
            return np.ones((len(texts), emb.EMBEDDING_DIM), dtype=np.float32)

    monkeypatch.setattr(emb, "_model", FakeModel())
    mat = emb.build_kb_index(["a", "b", "c"])
    assert mat.shape == (3, emb.EMBEDDING_DIM)
