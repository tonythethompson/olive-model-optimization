"""Tool: troubleshoot_olive_error.

Diagnoses Olive runtime and Olive Studio errors using domain-tagged KB
entries with hybrid semantic + keyword scoring. Tracks error frequency
with a module-level store.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from typing import Any, Literal

import numpy as np

from . import KB_DIR, load_quirks, load_studio_troubleshooting, load_troubleshooting
from .embeddings import (
    DEFAULT_THRESHOLD,
    EMBEDDING_DIM,
    build_kb_index,
    cosine_similarity_scores,
    encode_query,
    is_model_loaded,
)
from .feedback import FEEDBACK_MAX_ADJUSTMENT, feedback_score_delta
from .retrieval import (
    budget_degraded_reason,
    get_retrieval_mode,
    get_semantic_budget_ms,
    merge_retrieval_meta,
    retrieval_meta,
    run_with_budget,
)

logger = logging.getLogger(__name__)

DomainName = Literal["auto", "olive", "studio"]

# ---------------------------------------------------------------------------
# Frequency label thresholds (tuneable constants)
# ---------------------------------------------------------------------------
RECURRING_MAX = 3
FREQUENT_MAX = 10

# Hybrid scoring weights: semantic cosine + keyword pattern hits.
_SEMANTIC_WEIGHT = 0.6
_KEYWORD_WEIGHT = 0.4
# Extra ranking boost per pattern alternative hit (OR list multi-evidence).
_HIT_RANK_BONUS = 0.05
_HIT_RANK_BONUS_CAP = 5
# Aggregate feedback may nudge rank by at most FEEDBACK_MAX_ADJUSTMENT (0.05):
# enough to break close ties, never enough to invent a match or overturn a
# clear keyword hit (_KEYWORD_WEIGHT = 0.4).

# ---------------------------------------------------------------------------
# Error frequency tracker (module-level, lives for the process lifetime)
# ---------------------------------------------------------------------------
_lock = threading.Lock()
_frequency_store: dict[str, dict[str, Any]] = {}

# ---------------------------------------------------------------------------
# Troubleshooting embedding indexes (keyed by content fingerprint)
# ---------------------------------------------------------------------------
_ts_index_lock = threading.Lock()
# fingerprint -> (entries, embeddings, kb_mtime_at_build)
_ts_index_cache: dict[str, tuple[list[dict[str, Any]], np.ndarray, float]] = {}
_TS_INDEX_CACHE_MAX = 8


def _get_frequency_key(matched_entry: str | None, error_message: str) -> str:
    """Derive a stable key for frequency tracking."""
    if matched_entry:
        return f"entry:{matched_entry}"
    prefix = error_message[:80].lower().strip()
    return f"msg:{prefix}"


def _record_occurrence(key: str) -> dict[str, Any]:
    """Record one occurrence and return the updated frequency metadata."""
    now = time.time()
    with _lock:
        entry = _frequency_store.get(key)
        if entry is None:
            entry = {
                "occurrence_count": 1,
                "first_seen": now,
                "last_seen": now,
            }
            _frequency_store[key] = entry
        else:
            entry["occurrence_count"] += 1
            entry["last_seen"] = now
        return dict(entry)


def _frequency_label(count: int) -> str:
    """Human-readable frequency label."""
    if count == 1:
        return "first_occurrence"
    if count <= RECURRING_MAX:
        return "recurring"
    if count <= FREQUENT_MAX:
        return "frequent"
    return "persistent"


def _format_ts(ts: float) -> str:
    """Format an epoch timestamp as an ISO-8601 string."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))


# ---------------------------------------------------------------------------
# Hybrid scoring (semantic + keyword)
# ---------------------------------------------------------------------------


def _entry_embed_text(entry: dict[str, Any]) -> str:
    """Build the text used to embed a troubleshooting entry."""
    parts = [
        str(entry.get("title") or ""),
        str(entry.get("root_cause") or ""),
        str(entry.get("solution") or ""),
    ]
    patterns = entry.get("patterns") or []
    parts.extend(str(p) for p in patterns)
    return " ".join(p for p in parts if p).strip()


def _entries_fingerprint(entries: list[dict[str, Any]]) -> str:
    """Stable fingerprint of entry ids + embed texts (order-sensitive)."""
    h = hashlib.sha256()
    for entry in entries:
        eid = str(entry.get("id") or "")
        h.update(eid.encode("utf-8"))
        h.update(b"\0")
        # LF-normalize so Windows/Linux fingerprint the same text content.
        text = _entry_embed_text(entry).replace("\r\n", "\n").replace("\r", "\n")
        h.update(text.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def _troubleshooting_kb_mtime() -> float:
    """Max mtime of olive + studio troubleshooting JSON files."""
    mtimes: list[float] = []
    for name in ("troubleshooting.json", "studio_troubleshooting.json"):
        path = KB_DIR / name
        try:
            mtimes.append(path.stat().st_mtime)
        except OSError:
            continue
    return max(mtimes) if mtimes else 0.0


def get_troubleshooting_index(
    entries: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    """
    Prepare embeddings for troubleshooting entries and reuse compatible cached indexes.

    Parameters:
        entries (list[dict[str, Any]] | None): Troubleshooting entries to index.
        When omitted, loads the default troubleshooting entries.

    Returns:
        tuple[list[dict[str, Any]], np.ndarray]: The entries used to build the index
        and their position-aligned embedding matrix.
    """
    if entries is None:
        entries = load_troubleshooting()
    entries_list = list(entries)
    fingerprint = _entries_fingerprint(entries_list)
    file_mtime = _troubleshooting_kb_mtime()

    with _ts_index_lock:
        cached = _ts_index_cache.get(fingerprint)
        if cached is not None and cached[2] == file_mtime:
            return cached[0], cached[1]

    from .index_store import load_entry_embeddings

    # Prefer shipped precomputed embeddings when content hash matches.
    stem = (
        "ts_studio"
        if (entries_list and all(str(e.get("domain") or "") == "studio" for e in entries_list))
        else "ts_olive"
    )
    # Auto-domain pool mixes olive+studio — never use a single-domain shipped index.
    mixed = (
        bool(entries_list)
        and any(str(e.get("domain") or "olive") == "studio" for e in entries_list)
        and any(str(e.get("domain") or "olive") != "studio" for e in entries_list)
    )
    shipped = None if mixed else load_entry_embeddings(stem, fingerprint)

    texts = [_entry_embed_text(e) for e in entries_list]
    if shipped is not None:
        embeddings = shipped
    elif not texts:
        embeddings = np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
    else:
        embeddings = build_kb_index(texts)

    with _ts_index_lock:
        cached = _ts_index_cache.get(fingerprint)
        if cached is not None and cached[2] == file_mtime:
            return cached[0], cached[1]
        # Only publish if mtime is still the build-time value.
        current_mtime = _troubleshooting_kb_mtime()
        if current_mtime != file_mtime:
            return entries_list, embeddings
        _ts_index_cache[fingerprint] = (entries_list, embeddings, file_mtime)
        # Bound cache size: domain pools are few (olive/studio/auto), so evict
        # the oldest entry rather than let stale fingerprints from a
        # hot-reloading KB accumulate forever.
        if len(_ts_index_cache) > _TS_INDEX_CACHE_MAX:
            oldest_key = next(iter(_ts_index_cache))
            if oldest_key != fingerprint:
                del _ts_index_cache[oldest_key]
        return entries_list, embeddings


# Private alias retained for older tests that patch the historical name.
_get_troubleshooting_index = get_troubleshooting_index


def _pattern_hit_count(entry: dict[str, Any], text: str) -> int:
    """Count how many pattern alternatives appear in text."""
    patterns = entry.get("patterns") or []
    return sum(1 for p in patterns if str(p).lower() in text)


def _keyword_normalized(entry: dict[str, Any], text: str) -> float:
    """Pattern OR evidence in {0.0, 1.0}."""
    return 1.0 if _pattern_hit_count(entry, text) > 0 else 0.0


def _score(
    entry: dict[str, Any],
    error_message: str,
    pass_name: str,
    config_context: str,
    *,
    semantic_score: float = 0.0,
) -> float:
    """Hybrid score: 0.6 * semantic + 0.4 * keyword_OR + small multi-hit bonus.

    Keyword text includes pass_name/config_context for pattern matching.
    Semantic scores should be computed from error_message alone.
    Empty error_message must not be scored via pass_name alone (callers
    short-circuit before match); if called anyway, keyword on empty body
    alone yields 0 when pass/context are excluded.
    """
    error_only = (error_message or "").strip()
    if not error_only:
        return 0.0

    text = f"{error_message} {pass_name or ''} {config_context or ''}".lower()
    hits = _pattern_hit_count(entry, text)
    keyword = 1.0 if hits > 0 else 0.0
    semantic = float(semantic_score) if float(semantic_score) >= DEFAULT_THRESHOLD else 0.0
    bonus = _HIT_RANK_BONUS * min(hits, _HIT_RANK_BONUS_CAP)
    return _SEMANTIC_WEIGHT * semantic + _KEYWORD_WEIGHT * keyword + bonus


MAX_RELEVANT_QUIRKS = 20

_ENTRY_QUIRK_CATEGORIES: dict[str, list[str]] = {
    "onnx-export-shape": ["onnx_export", "pass_ordering"],
    "onnx-export-external-data": ["onnx_export"],
    "quant-accuracy-collapse": ["quantization", "pass_ordering"],
    "ep-fallback-cpu": ["hardware"],
    "oom-quantization": ["quantization", "onnx_export"],
    "calibration-data-mismatch": ["quantization"],
    "lora-target-modules": ["lora"],
    "tensorrt-build-slow": ["quantization", "hardware", "onnx_export"],
    "awq-slow-calibration": ["quantization"],
    "qnn-layer-not-supported": ["hardware", "quantization", "pass_ordering"],
    "coreml-dynamic-shape": ["onnx_export"],
    "lora-merge-fail": ["lora", "quantization"],
    "openvino-fallback": ["hardware"],
    "transformer-fusion-missing-dims": ["pass_ordering", "onnx_export"],
    "int4-perplexity": ["quantization", "pass_ordering"],
    "onnx-fp16-nan": ["onnx_export", "pass_ordering"],
    "calibration-distribution-mismatch": ["quantization"],
    "multi-pass-cache-overwrite": ["pass_ordering", "quantization", "onnx_export"],
    "search-local-optima": ["pass_ordering"],
    "torchscript-export-fail": ["onnx_export"],
    "olive-module-not-found": ["pass_ordering"],
    "olive-ort-cuda-ep-missing": ["hardware"],
    "olive-hf-auth-401": ["pass_ordering"],
    "olive-model-path-missing": ["pass_ordering"],
    "olive-disk-full": ["onnx_export"],
    "olive-cudnn-cuda-mismatch": ["hardware"],
    "olive-safetensors-missing": ["pass_ordering"],
    "olive-bitsandbytes-cuda": ["lora", "hardware"],
    "olive-triton-missing": ["quantization", "hardware"],
    "olive-ssl-huggingface": ["pass_ordering"],
    "olive-pass-config-validation": ["pass_ordering"],
    "olive-accelerator-device-busy": ["hardware", "quantization"],
    "studio-pytorch-hf-config": ["studio", "onnx_export"],
    "studio-apply-fix-empty": ["studio"],
    "studio-venv-mcp-pin": ["studio"],
    "studio-ai-provider-inactive": ["studio"],
    "studio-diagnose-wiring": ["studio"],
    "studio-tensorrt-pip-invalid-requirement": ["studio", "hardware"],
    "studio-recipe-not-parsed": ["studio"],
    "studio-unique-cache-dir": ["studio", "pass_ordering"],
    "studio-tensorrt-rtx-missing": ["studio", "hardware"],
    "studio-directml-missing": ["studio", "hardware"],
    "studio-webgpu-local-blocked": ["studio", "hardware"],
    "studio-openvino-npu-device": ["studio", "hardware"],
    "olive-directml-ep-missing": ["hardware"],
    "olive-webgpu-not-server-ort": ["hardware"],
}

_QUIRK_CATEGORY_ORDER: tuple[str, ...] = (
    "pass_ordering",
    "quantization",
    "onnx_export",
    "lora",
    "hardware",
    "studio",
)


def _infer_quirk_categories(entry_id: str | None, pass_name: str, domain: str | None) -> set[str]:
    """Infer relevant quirk categories from the matched entry, domain, and pass name."""
    categories: set[str] = set()

    if entry_id:
        categories.update(_ENTRY_QUIRK_CATEGORIES.get(entry_id, []))

    if domain == "studio":
        categories.add("studio")

    p = (pass_name or "").lower()
    if any(k in p for k in ("quant", "awq", "qat", "int4", "int8", "nvfp4", "hqq", "gptq")):
        categories.add("quantization")
        categories.add("pass_ordering")
    if any(k in p for k in ("onnx", "conversion", "export", "coreml", "float16", "fp16")):
        categories.add("onnx_export")
    if any(k in p for k in ("lora", "peft", "qlora")):
        categories.add("lora")
    if any(k in p for k in ("tensorrt", "qnn", "openvino", "execution", "provider", "cuda", "rocm")):
        categories.add("hardware")
    if any(k in p for k in ("transform", "optimize", "order", "cache", "search", "fusion", "split")):
        categories.add("pass_ordering")
    if any(k in p for k in ("studio", "venv", "diagnose", "apply", "sidebar", "recipe builder")):
        categories.add("studio")

    if not categories:
        categories = (
            {"pass_ordering", "quantization", "studio"} if domain == "studio" else {"pass_ordering", "quantization"}
        )

    return categories


def _build_relevant_quirks(
    entry_id: str | None,
    pass_name: str,
    domain: str | None = None,
) -> list[str]:
    """Build a deduplicated list of relevant quirk titles for an error context."""
    categories = _infer_quirk_categories(entry_id, pass_name, domain)
    quirks_db = load_quirks()
    titles: list[str] = []
    seen: set[str] = set()

    ordered_cats = [c for c in _QUIRK_CATEGORY_ORDER if c in categories]
    ordered_cats.extend(sorted(c for c in categories if c not in _QUIRK_CATEGORY_ORDER))

    for category in ordered_cats:
        for quirk in quirks_db.get(category, []):
            if len(titles) >= MAX_RELEVANT_QUIRKS:
                return titles
            title = quirk.get("title") if isinstance(quirk, dict) else None
            if not title or title in seen:
                continue
            seen.add(title)
            titles.append(title)
    return titles


def _pool_for_domain(domain: DomainName) -> list[dict[str, Any]]:
    """Load troubleshooting entries for the requested domain only."""
    if domain == "olive":
        return load_troubleshooting()
    if domain == "studio":
        return load_studio_troubleshooting()
    return load_troubleshooting() + load_studio_troubleshooting()


def _semantic_scores_for_entries(
    entries: list[dict[str, Any]],
    error_only: str,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    """Build index + cosine scores (may load the embedding model)."""
    # Use the private alias so unit tests can monkeypatch index construction
    # without loading sentence-transformers.
    index_entries, embeddings = _get_troubleshooting_index(entries)
    query_vec = encode_query(error_only)
    semantic_scores = cosine_similarity_scores(query_vec, embeddings)
    return index_entries, semantic_scores


def _best_match(
    entries: list[dict[str, Any]],
    error_message: str,
    pass_name: str,
    config_context: str,
    *,
    require_keyword: bool = False,
    mode: str | None = None,
) -> tuple[dict[str, Any] | None, float, dict[str, Any]]:
    """
    Select the highest-scoring troubleshooting entry for an error.

    Parameters:
        require_keyword (bool): Require at least one matching keyword or pattern.
        mode (str | None): Retrieval mode, such as ``"auto"``, ``"keyword"``, or
            ``"semantic"``.

    Returns:
        tuple: The selected entry, its score, and retrieval metadata. The entry is
        ``None`` with a score of ``0.0`` when no qualifying match exists.
    """
    resolved_mode = get_retrieval_mode(mode)
    empty_meta = retrieval_meta(mode=resolved_mode, effective="none")

    if not entries:
        return None, 0.0, empty_meta

    error_only = (error_message or "").strip()
    if not error_only:
        return None, 0.0, empty_meta

    score_entries = list(entries)
    semantic_scores = np.zeros((len(score_entries),), dtype=np.float32)
    effective = "keyword"
    degraded = False
    reason: str | None = None

    if resolved_mode == "keyword":
        effective = "keyword"
    else:
        use_budget = resolved_mode == "auto" and not is_model_loaded()
        budget_ms = get_semantic_budget_ms() if use_budget else 0

        def _load() -> tuple[list[dict[str, Any]], np.ndarray]:
            """
            Load semantic scores for the candidate entries.

            Returns:
                A tuple containing the candidate entries and their semantic scores.
            """
            return _semantic_scores_for_entries(entries, error_only)

        try:
            result, outcome = run_with_budget(_load, budget_ms)
            if outcome != "ok":
                degraded = True
                reason = budget_degraded_reason(outcome)
                effective = "keyword"
                if outcome == "busy":
                    logger.warning(
                        "Semantic scoring busy (single-flight in progress); keyword-only fallback",
                    )
                else:
                    logger.warning(
                        "Semantic scoring exceeded budget (%sms); keyword-only fallback",
                        budget_ms,
                    )
            else:
                score_entries, semantic_scores = result
                effective = "hybrid"
        except Exception:
            logger.warning(
                "Semantic scoring failed; falling back to keyword-only matching",
                exc_info=True,
            )
            score_entries = list(entries)
            semantic_scores = np.zeros((len(score_entries),), dtype=np.float32)
            degraded = True
            reason = "semantic_error"
            effective = "keyword"

    meta = retrieval_meta(
        mode=resolved_mode,
        effective=effective,
        degraded=degraded,
        reason=reason,
    )

    keyword_text = f"{error_message} {pass_name or ''} {config_context or ''}".lower()
    scored: list[tuple[dict[str, Any], float, int]] = []
    for i, entry in enumerate(score_entries):
        sem = float(semantic_scores[i]) if i < len(semantic_scores) else 0.0
        hybrid = _score(entry, error_message, pass_name, config_context, semantic_score=sem)
        # Bounded feedback adjustment after semantic/keyword hybrid score.
        # Cap: |delta| <= FEEDBACK_MAX_ADJUSTMENT (0.05). Positive net votes
        # slightly boost / break close ties; negative slightly demote.
        # Only applied when hybrid > 0 so zero-score (no keyword/semantic
        # evidence) stays a no-match and cannot be promoted by thumbs alone.
        if hybrid > 0:
            eid = entry.get("id")
            if isinstance(eid, str) and eid:
                delta = feedback_score_delta(eid)
                # feedback_score_delta already caps; clamp again so the local
                # rank bound stays obvious at the call site.
                if delta > FEEDBACK_MAX_ADJUSTMENT:
                    delta = FEEDBACK_MAX_ADJUSTMENT
                elif delta < -FEEDBACK_MAX_ADJUSTMENT:
                    delta = -FEEDBACK_MAX_ADJUSTMENT
                hybrid = hybrid + delta
        hits = _pattern_hit_count(entry, keyword_text)
        scored.append((entry, hybrid, hits))
    scored.sort(key=lambda x: (x[1], x[2]), reverse=True)
    candidates = scored
    # Auto/keyword never accept a cosine-only hit: BGE-small scores
    # unrelated "error" strings at ~0.55–0.65 against this index.
    if resolved_mode != "semantic" and (require_keyword or resolved_mode in ("auto", "keyword")):
        candidates = [row for row in scored if row[2] > 0]
        if not candidates:
            return None, 0.0, meta
    best_entry, best_score, _hits = candidates[0]
    if best_score <= 0:
        return None, 0.0, meta
    return best_entry, best_score, meta


def _resolve_domain(domain: str | None) -> DomainName:
    """Resolve a requested troubleshooting domain to a supported domain."""
    if domain in ("olive", "studio", "auto"):
        return domain  # type: ignore[return-value]
    return "auto"


def _no_match_payload() -> dict[str, Any]:
    """Provide generic troubleshooting guidance when no knowledge-base entry matches."""
    return {
        "matched_entry": None,
        "domain": None,
        "applyable": False,
        "title": "No exact match found",
        "root_cause": "The error does not match a known entry in the Olive or Olive Studio knowledge base.",
        "solution": (
            "Check official Olive docs and GitHub issues; reduce to a minimal repro and verify "
            "input model, pass order, and data config. For Olive Studio UI/builder issues, confirm "
            "recipe rebuild, provider settings, and MCP install (mcp<2)."
        ),
        "updated_config": {},
    }


def _build_diagnosis_payload(
    best: dict[str, Any],
    matched_entry: str | None,
    matched_domain: str | None,
    applyable: bool,
    pass_name: str,
    freq: dict[str, Any],
    retrieval: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Assemble a diagnosis response containing match details, troubleshooting
    guidance, frequency metadata, and optional retrieval metadata.

    Parameters:
        best (dict[str, Any]): Diagnosis details used to populate the response.
        matched_entry (str | None): Identifier of the matched knowledge-base entry, or `None` for no match.
        matched_domain (str | None): Domain associated with the matched entry.
        applyable (bool): Whether the matched diagnosis can be applied.
        pass_name (str): Name of the pass associated with the diagnosis.
        freq (dict[str, Any]): Frequency information for the diagnosed error.
        retrieval (dict[str, Any] | None): Optional metadata describing how the diagnosis was retrieved.

    Returns:
        dict[str, Any]: A structured diagnosis payload.
    """
    updated_config = best.get("updated_config", {}) or {}
    payload: dict[str, Any] = {
        "matched_entry": matched_entry,
        "domain": matched_domain,
        "applyable": applyable if matched_entry else False,
        "title": best.get("title", ""),
        "root_cause": best.get("root_cause", ""),
        "workaround": best.get("solution", ""),
        "updated_config": updated_config if isinstance(updated_config, dict) else {},
        "relevant_quirks": _build_relevant_quirks(matched_entry, pass_name, matched_domain),
        "related_olive_entry": best.get("related_olive_entry"),
        "frequency": {
            "occurrence_count": freq["occurrence_count"],
            "first_seen": _format_ts(freq["first_seen"]),
            "last_seen": _format_ts(freq["last_seen"]),
            "label": _frequency_label(freq["occurrence_count"]),
        },
    }
    if retrieval is not None:
        payload["retrieval"] = retrieval
    return payload


def _merge_retrieval_meta(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Merge retrieval metadata from two search pools (thin alias for tests)."""
    return merge_retrieval_meta(a, b)


def troubleshoot_olive_error(
    error_message: str,
    pass_name: str = "",
    config_context: str = "",
    domain: str = "auto",
    mode: str = "",
) -> dict[str, Any]:
    """Diagnose an Olive or Olive Studio error using the selected retrieval mode and knowledge base.

    Args:
        error_message: Error message or traceback snippet to diagnose.
        pass_name: Name of the pass where the error occurred, if known.
        config_context: Additional configuration context used for matching.
        domain: Knowledge-base domain to search: ``"auto"``, ``"olive"``, or
            ``"studio"``. Invalid values use automatic domain selection.
        mode: Retrieval mode: ``"auto"``, ``"keyword"``, or ``"semantic"``.
            An empty value uses the configured default.

    Returns:
        A diagnosis containing the matched entry or generic guidance, along with
        domain, applicability, troubleshooting details, frequency metadata, and
        retrieval metadata.
    """
    mode_arg = mode if (mode or "").strip() else None
    # Empty body: never match from pass_name/config_context alone (TensorRT/AWQ/…).
    if not (error_message or "").strip():
        best = _no_match_payload()
        freq_key = _get_frequency_key(None, error_message)
        freq = _record_occurrence(freq_key)
        return _build_diagnosis_payload(
            best,
            None,
            None,
            False,
            pass_name,
            freq,
            retrieval=retrieval_meta(
                mode=get_retrieval_mode(mode_arg),
                effective="none",
            ),
        )

    resolved = _resolve_domain(domain)

    best: dict[str, Any] | None = None
    matched_domain: str | None = None
    retrieval: dict[str, Any] = retrieval_meta(
        mode=get_retrieval_mode(mode_arg),
        effective="keyword",
    )

    if resolved == "auto":
        olive_best, olive_score, olive_meta = _best_match(
            load_troubleshooting(),
            error_message,
            pass_name,
            config_context,
            mode=mode_arg,
        )
        studio_best, studio_score, studio_meta = _best_match(
            load_studio_troubleshooting(),
            error_message,
            pass_name,
            config_context,
            mode=mode_arg,
        )
        retrieval = _merge_retrieval_meta(olive_meta, studio_meta)
        # Score both pools; Olive wins ties so generic Olive guidance stays stable.
        if studio_score > olive_score and studio_best is not None and studio_score > 0:
            best = studio_best
            matched_domain = "studio"
        elif olive_best is not None and olive_score > 0:
            best = olive_best
            matched_domain = "olive"
        elif studio_best is not None and studio_score > 0:
            best = studio_best
            matched_domain = "studio"
    else:
        pool = _pool_for_domain(resolved)
        hit, score, retrieval = _best_match(
            pool,
            error_message,
            pass_name,
            config_context,
            require_keyword=(get_retrieval_mode(mode_arg) != "semantic"),
            mode=mode_arg,
        )
        if hit is not None and score > 0:
            best = hit
            matched_domain = str(hit.get("domain") or resolved)

    if best is None:
        best = _no_match_payload()
        matched_entry = None
        matched_domain = None
        applyable = False
    else:
        matched_entry = best.get("id")
        matched_domain = matched_domain or best.get("domain") or "olive"
        applyable = bool(best.get("applyable"))

    freq_key = _get_frequency_key(matched_entry, error_message)
    freq = _record_occurrence(freq_key)

    return _build_diagnosis_payload(
        best,
        matched_entry,
        matched_domain,
        applyable,
        pass_name,
        freq,
        retrieval=retrieval,
    )


def diagnose_error(
    error_message: str,
    pass_name: str = "",
    config_context: str = "",
    domain: str = "auto",
    mode: str = "",
) -> dict[str, Any]:
    """Alias for troubleshoot_olive_error."""
    return troubleshoot_olive_error(
        error_message=error_message,
        pass_name=pass_name,
        config_context=config_context,
        domain=domain,
        mode=mode,
    )


def reset_frequency_store() -> None:
    """Clear the in-memory frequency store (useful for tests)."""
    with _lock:
        _frequency_store.clear()


def get_error_frequency_summary(limit: int = 10) -> dict[str, Any]:
    """Summarize tracked errors by occurrence frequency."""
    if limit < 0:
        raise ValueError("limit must be non-negative")

    with _lock:
        items: list[tuple[str, dict[str, Any]]] = [(key, dict(data)) for key, data in _frequency_store.items()]

    items.sort(key=lambda kv: (kv[1]["occurrence_count"], kv[1]["last_seen"]), reverse=True)

    entries = []
    for key, data in items[:limit]:
        if key.startswith("entry:"):
            matched_entry = key.split(":", 1)[1]
            message_prefix = ""
        else:
            matched_entry = None
            message_prefix = key.split(":", 1)[1]

        entries.append(
            {
                "matched_entry": matched_entry,
                "message_prefix": message_prefix,
                "occurrence_count": data["occurrence_count"],
                "first_seen": _format_ts(data["first_seen"]),
                "last_seen": _format_ts(data["last_seen"]),
                "label": _frequency_label(data["occurrence_count"]),
            }
        )

    return {
        "total_tracked": len(items),
        "limit": limit,
        "entries": entries,
    }
