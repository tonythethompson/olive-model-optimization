"""Tool: plan_optimization — NL intent to UIState patch.

Parses a natural-language optimization intent and produces a partial UIState
patch ready to apply to the Olive Studio frontend store. Uses existing
strategy_advisor, hardware_guide, and pass_chain modules internally.

No module-level network calls; no new pip dependencies.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

from .agent_model_info import _is_valid_model_id
from .normalization import parse_hardware_target
from .strategy_advisor import get_quantization_strategy, normalize_model_type
from .studio_loopback import err
from .studio_recipe import validate_ui_state_recipe

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Intent parsing keyword sets
# ---------------------------------------------------------------------------

_HARDWARE_KEYWORDS: list[tuple[str, str]] = [
    # (regex pattern, canonical hardware target string for downstream calls)
    (r"\bnvidia\b", "NVIDIA RTX 4090"),
    (r"\brtx\s*\d{4}", "NVIDIA RTX 4090"),
    (r"\bcuda\b", "NVIDIA RTX 4090"),
    (r"\btensorrt\b", "TensorRT"),
    (r"\bopenvino\b", "OpenVINO CPU"),
    (r"\bintel\b", "Intel Core i9 CPU"),
    (r"\bqualcomm\b", "Qualcomm Snapdragon NPU"),
    (r"\bqnn\b", "Qualcomm Snapdragon NPU"),
    (r"\bsnapdragon\b", "Qualcomm Snapdragon NPU"),
    (r"\bapple\b", "Apple M2/M3 (CoreML)"),
    (r"\bcoreml\b", "Apple M2/M3 (CoreML)"),
    (r"\bdirectml\b", "Windows DirectML GPU"),
    (r"\brocm\b", "AMD MI300X / ROCm"),
    (r"\bmi\d{3}", "AMD MI300X / ROCm"),
    (r"\bwebgpu\b", "WebGPU (Browser)"),
    (r"\bcpu\b", "Intel Core i9 CPU"),
    (r"\bnpu\b", "Qualcomm Snapdragon NPU"),
]

_MODEL_PATTERNS: list[re.Pattern[str]] = [
    # HuggingFace org/model style — require at least one letter on each side
    # of the slash and exclude common slash-separated prose (and/or, int8/int4,
    # input/output, path fragments).
    re.compile(r"\b(?=[A-Za-z0-9_-]*[A-Za-z])[A-Za-z0-9_-]+/(?=[A-Za-z0-9._-]*[A-Za-z])[A-Za-z0-9._-]+\b"),
    # Known model family names
    re.compile(
        r"\b(?:llama|phi|mistral|qwen|falcon|gpt|bert|resnet|mobilenet|"
        r"whisper|vit|yolo|stable[- ]?diffusion|deepseek|mixtral|"
        r"efficientnet|t5|codellama)\b",
        re.IGNORECASE,
    ),
]

# Short slash-separated tokens that should not be treated as model references.
_MODEL_REF_STOPWORDS = frozenset(
    {
        "and/or",
        "input/output",
        "int4/int8",
        "int8/int4",
        "fp16/fp32",
        "fp32/fp16",
        "cpu/gpu",
        "gpu/cpu",
        "train/eval",
        "eval/train",
        "onnx/pt",
        "pt/onnx",
        "pytorch/torch",
    }
)

_OPTIMIZATION_KEYWORDS = re.compile(
    r"\b(?:quantiz(?:e|ation)|compress|optimi[sz]e|speed|latency|smaller|"
    r"int4|int8|awq|gptq|hqq|prune|pruning|lora|qlora|fp16|float16|"
    r"convert|onnx|reduce|shrink|accelerate|faster)\b",
    re.IGNORECASE,
)

# Canonical ONNX Runtime execution-provider IDs accepted by the UI's
# ihvProvider field (see src/types.ts IHVProvider). Used to gate
# _normalize_provider so unknown values do not pass through.
_UI_PROVIDER_IDS = frozenset(
    {
        "CUDAExecutionProvider",
        "TensorrtExecutionProvider",
        "NvTensorRTRTXExecutionProvider",
        "DmlExecutionProvider",
        "OpenVINOExecutionProvider",
        "QNNExecutionProvider",
        "QnnAbiExecutionProvider",
        "ROCMExecutionProvider",
        "WebGpuExecutionProvider",
        "CoreMLExecutionProvider",
        "NNAPIExecutionProvider",
        "VitisAIExecutionProvider",
        "SNPEExecutionProvider",
        "TensorflowLiteExecutionProvider",
        "XnnpackExecutionProvider",
        "WasmExecutionProvider",
        "CPUExecutionProvider",
    }
)


# ---------------------------------------------------------------------------
# Intent parsing
# ---------------------------------------------------------------------------


def _parse_intent(intent: str) -> dict[str, Any]:
    """Extract hardware target, model reference, and optimization goal from intent.

    Returns a dict with keys 'hardware_target', 'model_ref', 'optimization_goal'
    — each is either a non-empty string or None.
    """
    lower = intent.lower()
    result: dict[str, Any] = {
        "hardware_target": None,
        "model_ref": None,
        "optimization_goal": None,
    }

    # Hardware target — first match wins (ordered by specificity)
    # Route structured OpenVINO intents through parse_hardware_target before
    # the order-sensitive _HARDWARE_KEYWORDS matches, so inputs like
    # "intel npu" and "openvino npu" resolve to the canonical Intel Core Ultra
    # NPU (OpenVINO) target rather than Qualcomm Snapdragon NPU.
    _OV_PHRASE_RE = re.compile(
        r"\b(openvino[\s:+-]*(?:cpu|gpu|npu)|intel\s+npu|core\s+ultra\s+npu)\b",
        re.IGNORECASE,
    )
    ov_match = _OV_PHRASE_RE.search(intent)
    if ov_match:
        parsed_hw = parse_hardware_target(ov_match.group(0))
        if parsed_hw.profile and not parsed_hw.error:
            result["hardware_target"] = parsed_hw.profile
            if parsed_hw.openvino_device:
                # Stash the OV device so _compose_ui_state_patch can use it.
                result["_openvino_device"] = parsed_hw.openvino_device

    if result["hardware_target"] is None:
        for pattern, target in _HARDWARE_KEYWORDS:
            if re.search(pattern, lower):
                result["hardware_target"] = target
                break

    # Model reference — first match wins, but exclude generic slash-separated
    # prose that the org/model regex would otherwise capture.
    for pat in _MODEL_PATTERNS:
        m = pat.search(intent)
        if m and m.group(0).lower() not in _MODEL_REF_STOPWORDS:
            result["model_ref"] = m.group(0)
            break

    # Optimization goal
    m = _OPTIMIZATION_KEYWORDS.search(intent)
    if m:
        result["optimization_goal"] = m.group(0)

    return result


def _infer_provider(hardware_target: str) -> str | None:
    """Map a hardware target string to a canonical ONNX Runtime provider ID.

    CPU-only intents (e.g. ``Intel Core i9 CPU``) resolve to
    ``CPUExecutionProvider`` before the OpenVINO/Intel branch so the default
    CPU target does not become an OpenVINO provider.
    """
    lower = hardware_target.lower()
    if "tensorrt" in lower:
        return "TensorrtExecutionProvider"
    if "nvidia" in lower or "rtx" in lower or "cuda" in lower:
        return "CUDAExecutionProvider"
    if "directml" in lower:
        return "DmlExecutionProvider"
    if "rocm" in lower or "mi300" in lower:
        return "ROCMExecutionProvider"
    if "apple" in lower or "coreml" in lower:
        return "CoreMLExecutionProvider"
    if "qualcomm" in lower or "qnn" in lower or "snapdragon" in lower:
        return "QNNExecutionProvider"
    # OpenVINO must be checked before CPU so "Intel Core i9 CPU" (the default
    # OpenVINO CPU target from _HARDWARE_KEYWORDS) resolves to
    # OpenVINOExecutionProvider. Pure CPU intents that don't mention OpenVINO
    # or Intel (e.g. "CPU") still fall through to CPUExecutionProvider.
    if "openvino" in lower:
        return "OpenVINOExecutionProvider"
    if "intel" in lower and "cpu" not in lower:
        return "OpenVINOExecutionProvider"
    if "cpu" in lower:
        return "CPUExecutionProvider"
    if "webgpu" in lower:
        return "WebGpuExecutionProvider"
    return None


def _normalize_provider(value: Any) -> str | None:
    """Normalize probe provider labels to canonical UI-supported provider IDs.

    Returns a canonical provider ID only when the input is a non-empty string
    matching the UI's accepted ihvProvider IDs (directly or via alias). Unknown
    values return None so they cannot pass through into the UIState patch.
    """
    if not isinstance(value, str) or not value:
        return None
    canonical = {
        "nvidia": "CUDAExecutionProvider",
        "cuda": "CUDAExecutionProvider",
        "tensorrt": "TensorrtExecutionProvider",
        "directml": "DmlExecutionProvider",
        "amd": "ROCMExecutionProvider",
        "rocm": "ROCMExecutionProvider",
        "apple": "CoreMLExecutionProvider",
        "coreml": "CoreMLExecutionProvider",
        "qualcomm": "QNNExecutionProvider",
        "qnn": "QNNExecutionProvider",
        "intel": "OpenVINOExecutionProvider",
        "openvino": "OpenVINOExecutionProvider",
        "webgpu": "WebGpuExecutionProvider",
    }
    resolved = canonical.get(value.lower(), value)
    return resolved if resolved in _UI_PROVIDER_IDS else None


def _infer_cuda_version(intent: str) -> str | None:
    """Extract CUDA version from intent if mentioned."""
    m = re.search(r"cuda\s*(\d{2}(?:\.\d)?)", intent, re.IGNORECASE)
    if m:
        return m.group(1)
    return None


def _build_strategy_passes(strategy: dict[str, Any]) -> dict[str, Any]:
    """Translate a strategy recommendation into a partial passes patch."""
    passes: dict[str, Any] = {}
    algo_lower = str(strategy.get("recommended_algorithm", "")).lower()

    if "int4" in algo_lower or "int8" in algo_lower:
        passes["quantization"] = True
        passes["quantPrecision"] = "int4" if "int4" in algo_lower else "int8"

        methods = (
            ("awq", "awq"),
            ("gptq", "gptq"),
            ("hqq", "hqq"),
            ("static", "static"),
            ("dynamic", "dynamic"),
            ("weight", "weight_only"),
        )
        quant_method = next((value for needle, value in methods if needle in algo_lower), None)
        if quant_method:
            passes["quantMethod"] = quant_method

    if "fp16" in algo_lower or "float16" in algo_lower:
        passes["fp16Conversion"] = True
    if strategy.get("pass_chain"):
        passes["passChain"] = strategy["pass_chain"]
    return passes


def _probe_provider(hardware_probe: dict[str, Any]) -> str | None:
    """Return a normalized provider from either supported probe key."""
    provider = hardware_probe.get("ihvProvider")
    if provider is None:
        provider = hardware_probe.get("provider")
    return _normalize_provider(provider)


def _apply_hardware_probe_overrides(
    patch: dict[str, Any],
    hardware_probe: dict[str, Any] | None,
) -> None:
    """Apply authoritative hardware-probe values to a UI-state patch."""
    if not hardware_probe:
        return

    provider = _probe_provider(hardware_probe)
    if provider:
        patch["ihvProvider"] = provider
    for field in ("cudaVersion", "openvinoTargetDevice"):
        value = hardware_probe.get(field)
        if isinstance(value, str) and value:
            patch[field] = value


def _compose_ui_state_patch(
    strategy: dict[str, Any],
    model_id: str,
    hardware_target: str,
    hardware_probe: dict[str, Any] | None,
    intent: str,
    openvino_device: str | None = None,
) -> dict[str, Any]:
    """Compose a UIState patch from strategy and guide results."""
    patch: dict[str, Any] = {}

    # Model source + ID
    if model_id:
        patch["hfModelId"] = model_id
        patch["modelSource"] = "huggingface"

    # Provider
    provider = _infer_provider(hardware_target)
    if provider:
        patch["ihvProvider"] = provider

    # CUDA version
    cuda_ver = _infer_cuda_version(intent)
    if cuda_ver:
        patch["cudaVersion"] = cuda_ver

    # OpenVINO device — prefer the parsed device from the intent, then the
    # strategy's resolved device.
    ov_device = openvino_device or strategy.get("openvino_device")
    if ov_device:
        patch["openvinoTargetDevice"] = ov_device

    passes = _build_strategy_passes(strategy)
    if passes:
        patch["passes"] = passes
    _apply_hardware_probe_overrides(patch, hardware_probe)
    return patch


def _generate_alternatives(
    strategy: dict[str, Any],
    model_type: str,
) -> list[dict[str, Any]]:
    """Generate 0-3 alternative optimization approaches."""
    alternatives: list[dict[str, Any]] = []
    algo = str(strategy.get("recommended_algorithm", "")).lower()

    # Alternative 1: Different precision
    if "int4" in algo:
        alt_passes: dict[str, Any] = {"quantization": True, "quantPrecision": "int8"}
        alternatives.append(
            {
                "description": "INT8 quantization (higher accuracy, larger model)",
                "ui_state_patch": {"passes": alt_passes},
            }
        )
    elif "int8" in algo:
        alt_passes = {"quantization": True, "quantPrecision": "int4"}
        alternatives.append(
            {
                "description": "INT4 quantization (smaller model, potentially lower accuracy)",
                "ui_state_patch": {"passes": alt_passes},
            }
        )

    # Alternative 2: FP16 only (if not already the primary)
    if "fp16" not in algo and "float16" not in algo:
        alternatives.append(
            {
                "description": "FP16 conversion only (minimal accuracy loss, ~50% size reduction)",
                "ui_state_patch": {
                    "passes": {
                        "fp16Conversion": True,
                        "passChain": ["OnnxConversion", "OnnxFloatToFloat16"],
                    },
                },
            }
        )

    # Alternative 3: Different quant method for LLMs
    if model_type == "llm" and "awq" in algo:
        alternatives.append(
            {
                "description": "GPTQ quantization (alternative to AWQ, better for some models)",
                "ui_state_patch": {
                    "passes": {
                        "quantization": True,
                        "quantPrecision": "int4",
                        "quantMethod": "gptq",
                        "passChain": ["OnnxConversion", "GptqQuantizer"],
                    },
                },
            }
        )
    elif model_type == "llm" and "gptq" in algo:
        alternatives.append(
            {
                "description": "AWQ quantization (alternative to GPTQ, faster inference)",
                "ui_state_patch": {
                    "passes": {
                        "quantization": True,
                        "quantPrecision": "int4",
                        "quantMethod": "awq",
                        "passChain": ["OnnxConversion", "NVModelOptQuantization"],
                    },
                },
            }
        )

    return alternatives[:3]


def _validate_patch(patch: dict[str, Any]) -> tuple[bool, str | None]:
    """Validate the UIState patch via the Studio bridge (best-effort).

    Calls ``validate_ui_state_recipe`` directly (same-process) instead of
    routing through the HTTP ``/api/mcp/tool`` bridge. Returns
    ``(validated, validation_note)``. ``validated`` is True only when the
    validation response carries ``is_runnable: True`` with no schema or
    pipeline errors. Non-dict or unexpected responses are treated as not
    validated rather than falling through to success.
    """
    response = validate_ui_state_recipe(ui_state=patch)
    if not isinstance(response, dict):
        return False, "Patch not validated by Studio (unexpected response type)."
    if response.get("error"):
        error_code = response["error"]
        if error_code == "studio_unavailable":
            return False, "Studio bridge unavailable; patch not validated"
        # Other errors (e.g. validation failure) — still mark as not validated
        return False, f"Validation failed: {response.get('message', 'unknown error')}"
    # Require an explicit positive signal: is_runnable True with no critical
    # schema or pipeline errors.
    schema_errors = response.get("schema_errors", [])
    pipeline_issues = response.get("pipeline_issues", [])
    critical = response.get("pipeline_critical_count")
    if response.get("is_runnable") is True and not schema_errors and not pipeline_issues and not critical:
        return True, None
    return False, "Patch not validated by Studio (validation issues detected)."


def _validate_plan_inputs(
    intent: Any,
    model_id: Any,
    hardware_probe: Any = None,
) -> dict[str, Any] | None:
    """Return a structured validation error, or None for valid inputs."""
    if not isinstance(intent, str) or not intent:
        return err("invalid_input", "Intent must be a non-empty string (1-2000 characters).")
    if len(intent) > 2000:
        return err(
            "invalid_input",
            "Intent exceeds maximum length of 2000 characters.",
            detail=f"length={len(intent)}",
        )
    if model_id is not None:
        if not isinstance(model_id, str):
            return err("invalid_input", "model_id must be a string.")
        if model_id and len(model_id) > 200:
            return err(
                "invalid_input",
                "model_id exceeds maximum length of 200 characters.",
                detail=f"length={len(model_id)}",
            )
        if model_id and not _is_valid_model_id(model_id):
            return err(
                "invalid_input",
                "model_id must be a valid HuggingFace repo ID (owner/name), "
                "without path traversal, query strings, or control characters.",
            )
    if hardware_probe is not None and not isinstance(hardware_probe, dict):
        return err("invalid_input", "hardware_probe must be a JSON object (dict).")
    return None


def _build_reasoning(
    strategy: dict[str, Any],
    hardware_target: str,
    model_type: str,
    optimization_goal: str | None,
) -> str:
    """Summarize how the planner interpreted and handled the request."""
    parts = [
        f"Detected intent: hardware={hardware_target}, model_type={model_type}, "
        f"goal={optimization_goal or 'general optimization'}."
    ]
    optional_parts = (
        ("recommended_algorithm", "Recommended algorithm: {}."),
        ("calibration_strategy", "Calibration: {}."),
    )
    for field, template in optional_parts:
        if strategy.get(field):
            parts.append(template.format(strategy[field]))
    if strategy.get("risks"):
        parts.append(f"Key risks: {'; '.join(strategy['risks'][:2])}.")
    return " ".join(parts)


def _build_plan_response(
    patch: dict[str, Any],
    reasoning: str,
    alternatives: list[dict[str, Any]],
) -> dict[str, Any]:
    """Validate a patch and package the public planner response."""
    validated, validation_note = _validate_patch(patch)
    result: dict[str, Any] = {
        "ui_state_patch": patch,
        "reasoning": reasoning,
        "alternatives": alternatives,
        "validated": validated,
        "side_effect": False,
    }
    if validation_note is not None:
        result["validation_note"] = validation_note
    return result


# ---------------------------------------------------------------------------
# Main tool function
# ---------------------------------------------------------------------------


def plan_optimization(
    intent: str,
    hardware_probe: dict[str, Any] | None = None,
    model_id: str = "",
    session_id: str | None = None,
) -> dict[str, Any]:
    """Convert a natural-language optimization intent into a UIState patch.

    Args:
        intent: Natural language description of the desired optimization
                (1-2000 characters).
        hardware_probe: Optional hardware context from the agent's environment
                       (overrides inferred provider/CUDA settings).
        model_id: Optional HuggingFace model ID for model-type inference
                  (1-200 characters).
        session_id: Optional Studio agent-loop session ID.

    Returns:
        A dict containing ui_state_patch, reasoning, alternatives, validated,
        and optionally validation_note. Returns error dict on invalid input
        or unparseable intent.
    """
    try:
        input_error = _validate_plan_inputs(intent, model_id, hardware_probe)
        if input_error:
            return input_error

        from .studio_loopback import ENV_API_URL, _ensure_session, _update_session

        active_session_id: str | None = None
        if session_id or os.environ.get(ENV_API_URL):
            active_session_id, session = _ensure_session(session_id)
            if active_session_id is None:
                return session
        else:
            session = {}

        parsed = _parse_intent(intent)
        model_ref = parsed["model_ref"]
        optimization_goal = parsed["optimization_goal"]
        if not any((parsed["hardware_target"], model_ref, optimization_goal)):
            return err(
                "unparseable_intent",
                "Could not identify a hardware target, model reference, or optimization goal in the provided intent.",
            )

        model_type = normalize_model_type(model_id or model_ref) if model_id or model_ref else "generic"
        hardware_target = parsed["hardware_target"] or "Intel Core i9 CPU"
        openvino_device = parsed.get("_openvino_device")
        strategy = get_quantization_strategy(
            model_type=model_type,
            target_hardware=hardware_target,
        )
        # Handle error dicts returned by get_quantization_strategy before
        # passing strategy to _compose_ui_state_patch.
        if isinstance(strategy.get("error"), str) and strategy["error"]:
            return err(
                "unsupported_hardware",
                f"Hardware target '{hardware_target}' is not supported.",
                detail=strategy["error"],
            )
        patch = _compose_ui_state_patch(
            strategy=strategy,
            model_id=model_id,
            hardware_target=hardware_target,
            hardware_probe=hardware_probe,
            intent=intent,
            openvino_device=openvino_device,
        )
        reasoning = _build_reasoning(strategy, hardware_target, model_type, optimization_goal)
        alternatives = _generate_alternatives(strategy, model_type)
        result = _build_plan_response(patch, reasoning, alternatives)
        if active_session_id:
            update = _update_session(
                active_session_id,
                diagnosticNotes=[
                    *session.get("diagnosticNotes", [])[-49:],
                    "Optimization plan created and validated.",
                ],
            )
            if isinstance(update.get("error"), str) and update["error"]:
                return update
            result["session_id"] = active_session_id
        return result

    except Exception as exc:
        logger.warning("plan_optimization unexpected error", exc_info=True)
        return err("internal_error", type(exc).__name__)
