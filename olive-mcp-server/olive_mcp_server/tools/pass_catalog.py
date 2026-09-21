"""Tool: get_olive_passes."""

from typing import Any

from . import load_passes


def get_olive_passes(filter: str | None = None) -> dict[str, Any]:  # noqa: A002
    """List available Olive optimization passes, optionally filtered by category.

    Args:
        filter: Optional category filter, e.g. "quantization", "conversion",
            "graph_optimization", "pruning", "finetuning", "distillation",
            or "performance_tuning".

    Returns:
        A dict with a list of passes and optional filter info.
    """
    passes = load_passes()
    requested = (filter or "").strip().lower()

    if requested:
        passes = [p for p in passes if p.get("type", "").lower() == requested]

    return {
        "filter": requested or None,
        "count": len(passes),
        "passes": passes,
    }
