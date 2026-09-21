"""Tool: evaluate_optimization_tradeoff."""

import re
from typing import Any

from . import load_passes


def _parse_compression_factor(value: str | None) -> float | None:
    """Convert a typical_compression string (e.g. '70-80%', '4x', '4x-8x') into a size multiplier.

    Returns None when the metadata is not numeric or not applicable.
    """
    if not value:
        return None

    text = re.sub(r"\s*\(.*?\)", "", value).lower().strip()
    if text in ("", "n/a", "varies"):
        return None

    if "x" in text:
        # e.g. '4x' or '4x-8x': a 4x compression means ~25% of original size.
        parts = [p.strip() for p in text.replace("x", "").split("-") if p.strip()]
        try:
            ratios = [1.0 / float(p) for p in parts]
            return sum(ratios) / len(ratios)
        except ValueError:
            return None

    if "%" in text:
        # e.g. '70-80%' means the model is compressed by 70-80%, so remaining size is 20-30%.
        parts = [p.strip() for p in text.replace("%", "").split("-") if p.strip()]
        try:
            percentages = [float(p) for p in parts]
            return 1.0 - (sum(percentages) / len(percentages)) / 100.0
        except ValueError:
            return None

    return None


def evaluate_optimization_tradeoff(
    passes: list[str],
    model: str = "",
    evaluation_metrics: list[str] | None = None,
) -> dict[str, Any]:
    """Analyze quality vs. performance tradeoff for a pass sequence.

    Args:
        passes: Ordered list of pass names.
        model: Optional model name for context.
        evaluation_metrics: Metrics to predict, e.g. ["accuracy", "latency", "size"].

    Returns:
        Predicted outcomes for each metric, risk factors, and Pareto recommendations.
    """
    metrics = evaluation_metrics or ["accuracy", "latency", "size"]
    catalog = {p["name"]: p for p in load_passes()}

    accuracy = 100.0
    latency = 1.0
    size = 100.0
    risks = []
    unknown_passes = []

    for name in passes:
        meta = catalog.get(name)
        if not meta:
            unknown_passes.append(name)
            continue

        ptype = meta.get("type")
        compression = _parse_compression_factor(meta.get("typical_compression"))
        gotchas = meta.get("gotchas", [])

        if ptype == "quantization":
            size *= compression if compression is not None else 0.25
            latency *= 0.35
            accuracy -= 1.5
            risks.append(
                f"{name}: {gotchas[0] if gotchas else 'quantization can drop accuracy if calibration data is poor.'}"
            )
        elif ptype == "graph_optimization":
            size *= compression if compression is not None else 0.95
            latency *= 0.85
            risks.append(
                f"{name}: {gotchas[0] if gotchas else 'graph optimization is usually safe but can change numerics.'}"
            )
        elif ptype == "conversion":
            size *= compression if compression is not None else 1.0
            latency *= 1.05
        elif ptype == "pruning":
            size *= compression if compression is not None else 0.7
            latency *= 0.8
            accuracy -= 2.0
            risks.append(
                f"{name}: {gotchas[0] if gotchas else 'pruning accuracy loss often requires fine-tuning to recover.'}"
            )
        elif ptype == "finetuning":
            size *= compression if compression is not None else 1.05
            latency *= 1.05
            accuracy += 1.0
        elif ptype == "distillation":
            size *= compression if compression is not None else 0.9
            latency *= 0.9
            accuracy += 0.5
            risks.append(f"{name}: distillation is slow and depends on teacher-student compatibility.")
        else:
            size *= compression if compression is not None else 1.0

    predicted = {
        "accuracy": round(max(0.0, accuracy), 2),
        "latency": round(latency, 2),
        "size": round(max(0.0, size), 2),
    }

    requested = {m: predicted.get(m, 0.0) for m in metrics}

    recommendation = "This sequence is reasonable."
    if "accuracy" in metrics and predicted["accuracy"] < 95:
        recommendation = "Consider reducing quantization/pruning aggressiveness to recover accuracy."
    if "latency" in metrics and predicted["latency"] > 0.5:
        recommendation = "Latency improvement is modest; check pass ordering and EP selection."

    return {
        "passes": passes,
        "model": model,
        "evaluation_metrics": metrics,
        "predicted_outcomes": requested,
        "relative_to_baseline": {
            "accuracy_drop_percent": round(100 - predicted["accuracy"], 2),
            "latency_speedup": round(1 / predicted["latency"], 2) if predicted["latency"] > 0 else 0,
            "size_reduction_percent": round(100 - predicted["size"], 2),
        },
        "risk_factors": risks,
        "unknown_passes": unknown_passes,
        "pareto_recommendation": recommendation,
    }
