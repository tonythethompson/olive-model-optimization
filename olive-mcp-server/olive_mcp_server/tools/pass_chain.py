"""Tool: get_pass_chain."""

from typing import Any

from . import load_passes

# Canonical ordering constraints: each pass type should appear after its
# dependencies in this list.
_TYPE_ORDER = [
    "finetuning",
    "conversion",
    "graph_optimization",
    "quantization",
    "performance_tuning",
    "distillation",
    "pruning",
]


def _type_index(pass_type: str) -> int:
    try:
        return _TYPE_ORDER.index(pass_type)
    except ValueError:
        return len(_TYPE_ORDER)


# Known source-format aliases and the canonical tokens used by passes.json.
_SOURCE_FORMAT_ALIASES = {
    "pytorch": "torch",
    "torch": "torch",
    "pt": "torch",
    "huggingface": "hf",
    "hf": "hf",
    "transformers": "hf",
    "tensorflow": "tf",
    "tf": "tf",
    "onnx": "onnx",
    "openvino": "openvino",
    "ov": "openvino",
    "qnn": "qnn",
}


# Source formats for which an incompatibility with a pass's input_formats is an
# actionable error rather than a warning. Derived from the canonical alias
# values so new entries in _SOURCE_FORMAT_ALIASES are known automatically.
_KNOWN_SOURCE_FORMATS = set(_SOURCE_FORMAT_ALIASES.values())


def _normalize_source_format(fmt: str) -> str:
    """Map common source-format aliases to the tokens used in passes.json."""
    normalized = fmt.strip().lower()
    return _SOURCE_FORMAT_ALIASES.get(normalized, normalized)


def get_pass_chain(pass_names: list[str], source_format: str = "") -> dict[str, Any]:
    """Validate and explain an ordered pass chain.

    Args:
        pass_names: List of pass names in intended execution order.
        source_format: Optional source model format ("onnx", "torch", "hf", etc.).
                      If not specified, infers from chain or emits warnings.

    Returns:
        Validation result, explanation, and reordering suggestions.
    """
    passes = {p["name"]: p for p in load_passes()}
    errors = []
    warnings = []
    resolved = []
    seen_pass_names: list[str] = []
    seen_types: list[str] = []

    for name in pass_names:
        meta = passes.get(name)
        if not meta:
            errors.append(f"Unknown pass '{name}'.")
            resolved.append({"name": name, "known": False})
            seen_pass_names.append(name)
            continue

        ptype = meta.get("type", "unknown")
        resolved.append(
            {
                "name": name,
                "type": ptype,
                "input_formats": meta.get("input_formats", []),
                "output_formats": meta.get("output_formats", []),
                "known": True,
            }
        )

        # Quantization passes check: verify input format compatibility with preceding conversion pass or source format.
        if ptype == "quantization":
            has_compatible_conversion = False
            for prev in resolved:
                if prev.get("type") == "conversion" and any(
                    fmt in meta.get("input_formats", []) for fmt in prev.get("output_formats", [])
                ):
                    has_compatible_conversion = True
                    break
            if not has_compatible_conversion:
                normalized_source = _normalize_source_format(source_format)
                input_formats = meta.get("input_formats", [])
                if not normalized_source:
                    # No source format supplied: still surface the missing
                    # conversion so callers are not left without guidance.
                    warnings.append(
                        f"Pass '{name}' requires input format in {input_formats} and no source_format was provided. "
                        "Specify source_format (e.g. 'onnx', 'torch', 'hf') or add a compatible conversion pass."
                    )
                elif normalized_source not in input_formats:
                    if normalized_source in _KNOWN_SOURCE_FORMATS:
                        errors.append(
                            f"Pass '{name}' requires input format in {input_formats} "
                            f"but no compatible conversion pass precedes it in the chain."
                        )
                    else:
                        warnings.append(
                            f"Pass '{name}' requires input format in {input_formats}. "
                            f"If your source model is already compatible, this is fine; "
                            f"otherwise add a conversion pass."
                        )

        # Graph optimizations are more effective before quantization.
        if ptype == "graph_optimization" and "quantization" in seen_types:
            warnings.append(
                f"Pass '{name}' is a graph optimization placed after quantization. "
                "Consider moving graph optimizations earlier for a cleaner quantized graph."
            )

        # FP16 should generally be last; optimizing after FP16 can revert precision.
        if ptype == "graph_optimization" and "OnnxFloatToFloat16" in seen_pass_names:
            warnings.append(
                f"Pass '{name}' runs after OnnxFloatToFloat16; FP16 precision may be reverted by optimization."
            )

        seen_pass_names.append(name)
        seen_types.append(ptype)

    # Type ordering check.
    known_indices = []
    for n in pass_names:
        if n in passes:
            known_indices.append(_type_index(passes[n]["type"]))
        else:
            known_indices.append(-1)

    for i in range(len(known_indices) - 1):
        if known_indices[i] > known_indices[i + 1] and known_indices[i + 1] != -1:
            warnings.append(
                f"Pass '{pass_names[i]}' (type {passes[pass_names[i]]['type']}) "
                f"comes before '{pass_names[i + 1]}' (type {passes[pass_names[i + 1]]['type']}); "
                "consider reordering for the canonical pipeline."
            )

    valid = len(errors) == 0

    return {
        "valid": valid,
        "errors": errors,
        "warnings": warnings,
        "chain": resolved,
        "canonical_order": _TYPE_ORDER,
    }
