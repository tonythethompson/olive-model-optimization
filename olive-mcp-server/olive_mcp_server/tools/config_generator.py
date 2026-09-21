"""Tool: get_pass_config_template."""

from typing import Any

from . import load_passes

_TARGET_DEFAULTS = {
    "quality": {
        "calibration_sampling_size": 300,
        "per_channel": True,
        "reduce_range": False,
    },
    "latency": {
        "calibration_sampling_size": 100,
        "per_channel": False,
        "reduce_range": False,
    },
    "balanced": {
        "calibration_sampling_size": 128,
        "per_channel": False,
        "reduce_range": False,
    },
}


def _apply_target_defaults(params: dict[str, Any], target: str) -> dict[str, Any]:
    """Apply optimization-target defaults to a parameter set.

    Target-specific values take precedence over pass defaults.

    Args:
        params: Pass parameter defaults.
        target: Optimization target (quality/latency/balanced).

    Returns:
        Merged parameters with target defaults taking precedence.
    """
    target_defaults = _TARGET_DEFAULTS.get(target, _TARGET_DEFAULTS["balanced"]).copy()

    # Merge pass params first, then override with target defaults
    merged = params.copy()
    merged.update(target_defaults)
    return merged


def get_pass_config_template(
    pass_name: str,
    framework: str = "onnx",
    optimization_target: str = "balanced",
) -> dict[str, Any]:
    """Generate a scaffold Olive workflow configuration for a single pass.

    Args:
        pass_name: Name of the Olive pass, e.g. "OnnxQuantization".
        framework: Source framework: "torch", "onnx", or "tf".
        optimization_target: One of "quality", "latency", or "balanced".

    Returns:
        A JSON-ready Olive configuration snippet with commentary.
    """
    passes = {p["name"]: p for p in load_passes()}
    meta = passes.get(pass_name)
    if not meta:
        return {
            "error": f"Pass '{pass_name}' not found in catalog.",
            "available": sorted(passes.keys()),
        }

    target = optimization_target.lower()
    if target not in _TARGET_DEFAULTS:
        return {
            "error": f"Unknown optimization_target '{optimization_target}'. Choose: quality, latency, balanced.",
        }

    # Start with the pass default values.
    defaults = {k: v.get("default") for k, v in meta.get("optional_params", {}).items()}
    # Keep only non-null defaults.
    defaults = {k: v for k, v in defaults.items() if v is not None}

    # Apply target-specific defaults (target values take precedence)
    params = _apply_target_defaults(defaults, target)

    # Fill in placeholders for any required params not already covered.
    for req in meta.get("required_params", []):
        if req not in params:
            params[req] = f"<REQUIRED: set {req}>"

    # Framework-specific input model type.
    framework_map = {
        "torch": "PyTorchModel",
        "pytorch": "PyTorchModel",
        "hf": "HfModel",
        "huggingface": "HfModel",
        "onnx": "ONNXModel",
        "tf": "TensorFlowModel",
        "tensorflow": "TensorFlowModel",
    }
    input_model_type = framework_map.get(framework.lower(), "PyTorchModel")

    # Canonical engine setup.
    config = {
        "input_model": {
            "type": input_model_type,
            "config": {
                "model_path": "<path/to/model>",
                "task": "text-generation",
            },
        },
        "systems": {
            "local_system": {
                "type": "LocalSystem",
                "config": {"accelerators": [{"device": "cpu", "execution_providers": ["CPUExecutionProvider"]}]},
            }
        },
        "passes": {
            pass_name: {
                "type": meta["class"],
                "params": params,
            }
        },
        "engine": {
            "search_strategy": False,
            "host": "local_system",
            "target": "local_system",
            "cache_dir": "~/.cache/olive",
            "output_dir": "./models/optimized",
        },
    }

    return {
        "pass_name": pass_name,
        "framework": framework,
        "optimization_target": target,
        "description": meta.get("description"),
        "required_params": meta.get("required_params", []),
        "gotchas": meta.get("gotchas", []),
        "config": config,
    }
