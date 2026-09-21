"""Tool: get_context_for_pipeline.

Passive context retrieval for injecting KB snippets into an AI assistant
system prompt based on the current pipeline configuration.
"""

from __future__ import annotations

import logging
from typing import Any

from .docs_search import _keyword_search, _load_kb_text, get_or_build_kb_index
from .embeddings import DEFAULT_THRESHOLD, semantic_search
from .retrieval import get_retrieval_mode, retrieval_meta

logger = logging.getLogger(__name__)


def _pass_descriptor_text(item: Any) -> str:
    """Normalize a pass descriptor (str or dict) into searchable text."""
    if isinstance(item, str):
        return item.strip()
    if isinstance(item, dict):
        parts: list[str] = []
        for key in ("name", "type", "pass_name", "pass_type", "id"):
            val = item.get(key)
            if val:
                parts.append(str(val))
        # Fall back to any remaining string values.
        if not parts:
            parts = [str(v) for v in item.values() if isinstance(v, str) and v.strip()]
        return " ".join(parts).strip()
    return str(item).strip() if item is not None else ""


def _build_query(
    pipeline_passes: list[Any],
    model_name: str,
    target_hardware: str,
) -> tuple[str, str]:
    """Build semantic query text and a human-readable pipeline summary."""
    pass_texts = [_pass_descriptor_text(p) for p in (pipeline_passes or [])]
    pass_texts = [t for t in pass_texts if t]

    summary_parts: list[str] = []
    if pass_texts:
        summary_parts.append("passes: " + ", ".join(pass_texts))
    if model_name:
        summary_parts.append(f"model: {model_name}")
    if target_hardware:
        summary_parts.append(f"hardware: {target_hardware}")
    pipeline_summary = "; ".join(summary_parts) if summary_parts else "empty pipeline"

    query_bits = list(pass_texts)
    if model_name:
        query_bits.append(str(model_name))
    if target_hardware:
        query_bits.append(str(target_hardware))
    query = " ".join(query_bits).strip()
    return query, pipeline_summary


def get_context_for_pipeline(
    pipeline_passes: list[Any] | None = None,
    model_name: str = "",
    target_hardware: str = "",
    top_k: int = 5,
) -> dict[str, Any]:
    """Retrieve KB context snippets relevant to a planned Olive pipeline.

    Intended for frontend injection into an AI assistant system prompt.

    Args:
        pipeline_passes: List of pass names (strings) and/or dicts with
            name/type fields describing the pipeline.
        model_name: Optional model identifier.
        target_hardware: Optional hardware target string.
        top_k: Maximum number of snippets to return.

    Returns:
        context_snippets, pipeline_summary, confidence (avg relevance), snippet_count.
    """
    if top_k < 0:
        raise ValueError(f"top_k must be >= 0, got {top_k}")

    query, pipeline_summary = _build_query(
        list(pipeline_passes or []),
        model_name or "",
        target_hardware or "",
    )
    mode = get_retrieval_mode()

    if not query or top_k == 0:
        return {
            "context_snippets": [],
            "pipeline_summary": pipeline_summary,
            "confidence": 0.0,
            "snippet_count": 0,
            "status": "ok",
            "retrieval": retrieval_meta(mode=mode, effective="none"),
        }

    status = "ok"
    retrieval: dict[str, Any] = retrieval_meta(mode=mode, effective="semantic")

    def _keyword_results() -> list[dict[str, Any]]:
        terms = [t.lower() for t in query.split() if t]
        return _keyword_search(_load_kb_text(), terms, top_k)

    if mode == "keyword":
        try:
            results = _keyword_results()
            retrieval = retrieval_meta(mode=mode, effective="keyword")
        except Exception:
            logger.warning(
                "Keyword retrieval failed for pipeline context",
                exc_info=True,
            )
            results = []
            status = "retrieval_failed"
            retrieval = retrieval_meta(
                mode=mode,
                effective="none",
                degraded=True,
                reason="keyword_error",
            )
    else:
        try:
            kb_texts, embeddings = get_or_build_kb_index()
            results = semantic_search(
                query,
                kb_texts,
                embeddings,
                top_k,
                threshold=DEFAULT_THRESHOLD,
            )
        except Exception:
            logger.warning(
                "Semantic KB retrieval failed for pipeline context; falling back to keyword search",
                exc_info=True,
            )
            try:
                results = _keyword_results()
                status = "degraded"
                retrieval = retrieval_meta(
                    mode=mode,
                    effective="keyword",
                    degraded=True,
                    reason="semantic_error",
                )
            except Exception:
                logger.warning(
                    "Keyword fallback also failed for pipeline context",
                    exc_info=True,
                )
                results = []
                status = "retrieval_failed"
                retrieval = retrieval_meta(
                    mode=mode,
                    effective="none",
                    degraded=True,
                    reason="keyword_and_semantic_error",
                )

    confidences = [float(r.get("relevance", 0.0)) for r in results]
    confidence = sum(confidences) / len(confidences) if confidences else 0.0
    # Bound to [0, 1] for API stability (cosine is already ~[-1,1] but we threshold).
    confidence = max(0.0, min(1.0, confidence))

    return {
        "context_snippets": results,
        "pipeline_summary": pipeline_summary,
        "confidence": confidence,
        "snippet_count": len(results),
        # Distinguishes "genuinely no relevant KB entries" (ok, empty
        # results) from "semantic failed, keyword fallback used" (degraded)
        # from "both retrieval paths failed" (retrieval_failed).
        "status": status,
        "retrieval": retrieval,
    }
