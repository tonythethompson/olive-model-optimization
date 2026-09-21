"""Tool: get_quantization_strategy."""

import re
from typing import Any

from . import load_quirks
from .normalization import HardwareTarget, parse_hardware_target


def _hardware_category(profile: str) -> str:
    """Map a canonical profile name to a strategy category.

    Order matters: webgpu and directml must be checked before generic nvidia/intel
    so WebGPU does not collapse into nvidia and DirectML stays its own path.
    Apple/CoreML before intel: bare "core" must not steal "CoreML".
    """
    n = profile.lower()
    if "webgpu" in n or "web gpu" in n:
        return "webgpu"
    if "directml" in n or n == "dml" or "windows directml" in n:
        return "directml"
    # ROCm before generic nvidia/intel; do NOT match bare "amd" (EPYC CPU stays non-ROCm).
    if "rocm" in n or "mi300" in n or "mi250" in n or "mi210" in n:
        return "rocm"
    if "nvidia" in n or "rtx" in n or "tesla" in n or "t4" in n or "cuda" in n or "tensorrt" in n:
        return "nvidia"
    if "apple" in n or "coreml" in n or "m2" in n or "m3" in n:
        return "apple"
    if "intel" in n or "openvino" in n or "core ultra" in n or "core i" in n:
        return "intel"
    if "qualcomm" in n or "qnn" in n or "snapdragon" in n:
        return "qualcomm"
    if "android" in n or "nnapi" in n:
        return "android"
    if "xilinx" in n or "vitis" in n:
        return "xilinx"
    return "generic"


def normalize_model_type(model_type: str, architecture: str = "") -> str:
    """Normalize a model type string to a canonical category.

    This is the public alias of the internal ``_normalize_model_type`` helper.
    Callers outside ``strategy_advisor`` should use this public symbol so
    refactoring internals cannot cause an import-time failure.

    When ``architecture`` is provided (e.g. from the HuggingFace API), it is
    used as an additional classification signal so opaque model IDs can be
    classified from their architecture name (e.g.
    ``WhisperForConditionalGeneration`` → ``speech``).
    """
    return _normalize_model_type(model_type, architecture)


def _normalize_model_type(model_type: str, architecture: str = "") -> str:
    m = model_type.lower()
    if "llm" in m or any(x in m for x in ["llama", "mistral", "phi", "gpt", "qwen", "falcon"]):
        return "llm"
    if "cnn" in m or any(x in m for x in ["resnet", "mobilenet", "vit", "conv"]):
        return "cnn"
    if "vision" in m or "image" in m:
        return "vision"
    if "speech" in m or "whisper" in m or "audio" in m:
        return "speech"
    # Fall back to architecture-based classification for opaque model IDs.
    arch_lower = architecture.lower()
    if arch_lower:
        if any(x in arch_lower for x in ["llama", "mistral", "phi", "gpt", "qwen", "falcon", "causallm", "modelllama"]):
            return "llm"
        if any(x in arch_lower for x in ["resnet", "mobilenet", "vit", "conv", "cnn"]):
            return "cnn"
        if any(x in arch_lower for x in ["vision", "image", "clip"]):
            return "vision"
        if any(x in arch_lower for x in ["whisper", "speech", "audio"]):
            return "speech"
    return "generic"


def _latency_rank(latency: str) -> int:
    """
    Classify a latency description by urgency.

    Parameters:
        latency (str): Text describing the target latency.

    Returns:
        int: 0 for real-time latency, 1 for latency under 500 milliseconds, 2 for
        latency under one second, or 3 for unspecified latency.
    """
    lat_lower = latency.lower()
    if "<100" in lat_lower or "100ms" in lat_lower or "realtime" in lat_lower or "real-time" in lat_lower:
        return 0
    if "<500" in lat_lower or "500ms" in lat_lower:
        return 1
    if "<1" in lat_lower and "s" in lat_lower:
        return 2
    return 3


def get_quantization_strategy(
    model_type: str,
    target_hardware: str,
    latency_budget: str = "<500ms",
    accuracy_threshold: str = "<2% drop",
) -> dict[str, Any]:
    """Recommend a quantization approach for a model + hardware combo.

    Args:
        model_type: e.g. "LLM", "CNN", "Vision", or a model name like "llama-7b".
        target_hardware: e.g. "NVIDIA RTX 4090", "Qualcomm Snapdragon NPU".
        latency_budget: Human-readable latency target, e.g. "<100ms".
        accuracy_threshold: Human-readable accuracy constraint, e.g. "<2% drop".

    Returns:
        Recommended algorithm, calibration strategy, expected outcomes, and risks.
        ``target_hardware`` identifies the strategy bucket (e.g. ``nvidia``,
        ``webgpu``); ``resolved_profile`` preserves the resolved hardware
        description. ``openvino_device`` is unchanged when an OpenVINO path
        was selected. Returns ``{"error": ...}`` when the hardware target has
        an invalid OpenVINO device token.
    """
    parsed: HardwareTarget = parse_hardware_target(target_hardware)
    if parsed.error:
        return {"error": parsed.error}

    hw = _hardware_category(parsed.profile)
    # Prefer structured OV device; also honor the canonical NPU profile name
    # when callers pass it directly (no loose "npu" substring sniffing).
    is_openvino_npu = parsed.openvino_device == "NPU" or parsed.profile == "Intel Core Ultra NPU (OpenVINO)"
    mt = _normalize_model_type(model_type)
    latency_rank_val = _latency_rank(latency_budget)
    quirks = load_quirks()

    # Base recommendations.
    if mt == "llm":
        if hw == "nvidia":
            algorithm = "NVModelOptQuantization (AWQ, int4 weights / int8 activations)"
            calibration = "calibration-light (32-128 samples); AWQ is data-aware"
            expected = {
                "size_reduction": "75-85%",
                "latency_speedup": "8-15x on TensorRT",
                "accuracy_drop": "1-3% for well-calibrated LLMs",
            }
            risks = [
                "AWQ can hurt perplexity on small models; validate with perplexity metric.",
                "TensorRT engine build is slow and device-specific.",
            ]
            pass_chain = ["OnnxConversion", "NVModelOptQuantization"]
        elif hw == "directml":
            algorithm = "INT8 static / weight-only quantization for Windows DirectML"
            calibration = "100-200 representative samples; prefer static INT8 PTQ"
            expected = {
                "size_reduction": "65-75%",
                "latency_speedup": "2-5x on DirectML",
                "accuracy_drop": "1-4%",
            }
            risks = [
                "DirectML is Windows 10/11 + DirectX 12 only (onnxruntime-directml).",
                "Operator coverage differs from CUDA; validate the graph on target GPU.",
            ]
            pass_chain = ["OnnxConversion", "OnnxStaticQuantization"]
        elif hw == "webgpu":
            algorithm = "FP16 export for browser ORT Web WebGPU (not local Olive run)"
            calibration = "n/a for FP16 export; keep calibration light if probing INT8"
            expected = {
                "size_reduction": "40-50% (FP16)",
                "latency_speedup": "1-3x in-browser vs CPU WASM",
                "accuracy_drop": "<1% FP16",
            }
            risks = [
                "WebGPU is browser ORT Web only — Olive Studio blocks Execute Live (isRunnable: false).",
                "INT8 support varies by browser/GPU; prefer FP16 export recipes.",
            ]
            pass_chain = ["OnnxConversion", "OnnxFloatToFloat16"]
        elif hw == "rocm":
            algorithm = "GPTQ weight-only int4 for ROCm (prefer over AWQ)"
            calibration = "128 samples (GPTQ); AWQ has limited ROCm support"
            expected = {
                "size_reduction": "70-80%",
                "latency_speedup": "4-10x on MI300X-class HBM",
                "accuracy_drop": "2-4%",
            }
            risks = [
                "Operator coverage lags CUDA; verify the graph on ROCMExecutionProvider.",
                "Requires onnxruntime-rocm (or ROCm ORT build) and a working ROCm stack.",
                "MI300X has large HBM — prefer external data format for multi-GB weights.",
            ]
            pass_chain = ["OnnxConversion", "GptqQuantizer", "OnnxModelOptimizer"]
        elif hw == "apple":
            algorithm = "CoreML INT8 static quantization (QDQ)"
            calibration = "100-200 representative samples, symmetric per-channel"
            expected = {
                "size_reduction": "65-75%",
                "latency_speedup": "3-7x on Neural Engine",
                "accuracy_drop": "1-4%",
            }
            risks = [
                "CoreML has limited dynamic-shape support; fix sequence length.",
                "Mistral/GQA attention patterns may not map cleanly to CoreML.",
            ]
            pass_chain = ["OnnxConversion", "OnnxModelOptimizer", "OnnxStaticQuantization"]
        elif hw == "intel" and is_openvino_npu:
            algorithm = "OpenVINO weight compression / INT8 with device=NPU"
            calibration = "100 representative samples; OpenVINO NPU driver required"
            expected = {
                "size_reduction": "60-75%",
                "latency_speedup": "3-8x on Intel NPU",
                "accuracy_drop": "1-3%",
            }
            risks = [
                "Set openvinoTargetDevice to NPU in Olive Studio (EP id stays OpenVINOExecutionProvider).",
                "Unsupported ops fall back to CPU; verify Core Ultra / Meteor Lake+ NPU driver.",
                "OpenVINOConversion accepts torch or ONNX; prefer OpenVINOOptimumConversion "
                "only for HuggingFace/Torch sources.",
            ]
            # OpenVINOConversion accepts torch|onnx (unlike Optimum, which is torch-only).
            pass_chain = ["OpenVINOConversion", "OpenVINOWeightCompression"]
        else:
            algorithm = "GPTQ or HQQ weight-only int4 (CPU fallback)"
            calibration = "calibration-free (HQQ) or 128 samples (GPTQ)"
            expected = {
                "size_reduction": "70-80%",
                "latency_speedup": "2-5x (CPU), 4-8x (OpenVINO)",
                "accuracy_drop": "2-5%",
            }
            risks = [
                "Weight-only quantization does not speed up all operators on CPU.",
                "GPTQ can be slow to calibrate on large models.",
            ]
            pass_chain = ["OnnxConversion", "OnnxDynamicQuantization"]

    elif mt == "cnn" or mt == "vision":
        if hw == "nvidia":
            algorithm = "TensorRT INT8 static quantization"
            calibration = "200-500 ImageNet-like samples, per-channel symmetric"
            expected = {
                "size_reduction": "70-80%",
                "latency_speedup": "4-8x",
                "accuracy_drop": "<1%",
            }
            risks = [
                "Per-channel quantization improves accuracy but engine build is slower.",
            ]
            pass_chain = ["OnnxConversion", "OnnxModelOptimizer", "OnnxStaticQuantization", "OnnxFloatToFloat16"]
        elif hw == "directml":
            algorithm = "DirectML INT8 static PTQ"
            calibration = "100-300 ImageNet-like samples, per-channel when supported"
            expected = {
                "size_reduction": "65-75%",
                "latency_speedup": "2-5x",
                "accuracy_drop": "<2%",
            }
            risks = [
                "Windows + onnxruntime-directml only; do not mix CUDA/TRT packages into this EP path.",
                "Validate operator coverage on the target DirectX 12 GPU.",
            ]
            pass_chain = ["OnnxConversion", "OnnxModelOptimizer", "OnnxStaticQuantization"]
        elif hw == "webgpu":
            algorithm = "FP16 + model optimizer for browser WebGPU"
            calibration = "n/a for FP16; optional small set if probing quantized WebGPU"
            expected = {
                "size_reduction": "40-50% (FP16)",
                "latency_speedup": "1-3x in-browser",
                "accuracy_drop": "<1% FP16",
            }
            risks = [
                "Not a local Olive CLI execution target; export recipe for ORT Web WebGPU.",
                "INT8 support varies by browser/GPU.",
            ]
            pass_chain = ["OnnxConversion", "OnnxModelOptimizer", "OnnxFloatToFloat16"]
        elif hw == "rocm":
            algorithm = "INT8 static PTQ + optional FP16 for ROCm EP"
            calibration = "100-300 ImageNet-like samples; whitelist ops for ROCMExecutionProvider"
            expected = {
                "size_reduction": "65-75%",
                "latency_speedup": "2-6x on ROCm GPU",
                "accuracy_drop": "<2%",
            }
            risks = [
                "ROCm op coverage lags CUDA; verify unsupported ops on the target GPU.",
                "Requires onnxruntime-rocm / ROCm drivers matched to the host stack.",
            ]
            pass_chain = [
                "OnnxConversion",
                "OnnxStaticQuantization",
                "OnnxFloatToFloat16",
            ]
        elif hw == "qualcomm":
            algorithm = "QNN INT8 per-channel symmetric quantization"
            calibration = "128-256 representative samples, symmetric per-channel"
            expected = {
                "size_reduction": "65-75%",
                "latency_speedup": "5-10x on NPU",
                "accuracy_drop": "1-3%",
            }
            risks = [
                "QNN has a narrow operator whitelist; LayerNorm/GELU may fail.",
                "Must use symmetric per-channel quantization for best NPU accuracy.",
            ]
            pass_chain = ["OnnxConversion", "OnnxModelOptimizer", "QNNQuantization"]
        elif hw == "apple":
            algorithm = "CoreML per-channel INT8"
            calibration = "100-200 samples"
            expected = {
                "size_reduction": "60-70%",
                "latency_speedup": "3-6x",
                "accuracy_drop": "1-2%",
            }
            risks = [
                "CoreML requires fixed input shapes; dynamic batch is not supported.",
            ]
            pass_chain = ["OnnxConversion", "OnnxModelOptimizer", "OnnxStaticQuantization"]
        elif hw == "intel" and is_openvino_npu:
            algorithm = "OpenVINO INT8 / IR path with device=NPU"
            calibration = "100-200 samples; OpenVINO NPU plugin required"
            expected = {
                "size_reduction": "65-75%",
                "latency_speedup": "3-8x on Intel NPU",
                "accuracy_drop": "<2%",
            }
            risks = [
                "Olive Studio openvinoTargetDevice must be NPU; unsupported ops fall back to CPU.",
            ]
            pass_chain = ["OnnxConversion", "OpenVINOConversion", "OpenVINOQuantization"]
        else:
            algorithm = "OpenVINO INT8 or ONNX Runtime static INT8"
            calibration = "100-300 samples"
            expected = {
                "size_reduction": "65-75%",
                "latency_speedup": "2-4x",
                "accuracy_drop": "<2%",
            }
            risks = [
                "OpenVINO may silently fall back to CPUExecutionProvider for unsupported ops.",
            ]
            pass_chain = ["OnnxConversion", "OnnxModelOptimizer", "OnnxStaticQuantization"]

    else:
        # generic / speech
        if hw == "webgpu":
            algorithm = "FP16 export for browser WebGPU"
            calibration = "n/a for FP16 export"
            expected = {
                "size_reduction": "40-50% (FP16)",
                "latency_speedup": "1-3x in-browser",
                "accuracy_drop": "<1%",
            }
            risks = [
                "WebGPU is not a local Olive run target; use Export recipe only.",
            ]
            pass_chain = ["OnnxConversion", "OnnxFloatToFloat16"]
        elif hw == "directml":
            algorithm = "ONNX Runtime static INT8 (QDQ) for DirectML"
            calibration = "100-300 representative samples"
            expected = {
                "size_reduction": "65-75%",
                "latency_speedup": "2-5x",
                "accuracy_drop": "1-3%",
            }
            risks = [
                "Windows DirectML only; validate dynamic sequence lengths on target GPU.",
            ]
            pass_chain = ["OnnxConversion", "OnnxModelOptimizer", "OnnxStaticQuantization"]
        elif hw == "rocm":
            algorithm = "ONNX Runtime static INT8 (QDQ) for ROCm"
            calibration = "100-300 representative samples"
            expected = {
                "size_reduction": "65-75%",
                "latency_speedup": "2-5x on ROCm GPU",
                "accuracy_drop": "1-3%",
            }
            risks = [
                "ROCm op coverage lags CUDA; validate dynamic sequence lengths on target GPU.",
                "Requires onnxruntime-rocm / ROCm stack; prefer GPTQ for large LLMs instead of INT8 QDQ.",
            ]
            pass_chain = ["OnnxConversion", "OnnxModelOptimizer", "OnnxStaticQuantization"]
        else:
            algorithm = "ONNX Runtime static INT8 (QDQ)"
            calibration = "100-300 representative samples"
            expected = {
                "size_reduction": "65-75%",
                "latency_speedup": "2-4x",
                "accuracy_drop": "1-3%",
            }
            risks = [
                "Speech models often have dynamic sequence lengths; use dynamic_axes "
                "and a representative length range.",
            ]
            pass_chain = ["OnnxConversion", "OnnxModelOptimizer", "OnnxStaticQuantization"]

    # Latency / accuracy overrides only when the selected algorithm uses int4.
    uses_int4 = "int4" in algorithm.lower()
    if uses_int4:
        if latency_rank_val == 0 and mt == "llm":
            algorithm = algorithm.replace("int4", "int4 (aggressive)") + " + KV-cache quantization recommended"
            risks.append("Aggressive int4 can increase perplexity; evaluate with a held-out set.")
        elif latency_rank_val == 0 and (mt == "cnn" or mt == "vision"):
            algorithm += " + consider pruning 20-30% before quantization"
            risks.append("Pruning + quantization compound accuracy loss; fine-tune if possible.")

        match = re.search(r"(\d+(?:\.\d+)?)\s*%", accuracy_threshold)
        if match:
            try:
                threshold_value = float(match.group(1))
                if threshold_value <= 1.0:
                    algorithm = algorithm.replace("int4", "int8") + " (tight accuracy target)"
                    risks.append("Tight accuracy target requires larger calibration set and per-channel weights.")
            except ValueError:
                # Skip override if parsing fails
                pass

    # Pull the top quirks from the most relevant categories.
    candidates = quirks.get("quantization", [])[:2] + quirks.get("pass_ordering", [])[:1]
    relevant_quirks = [title for q in candidates if (title := q.get("title", ""))]

    result: dict[str, Any] = {
        "model_type": mt,
        "target_hardware": hw,
        "resolved_profile": parsed.profile,
        "latency_budget": latency_budget,
        "accuracy_threshold": accuracy_threshold,
        "recommended_algorithm": algorithm,
        "calibration_strategy": calibration,
        "expected_outcomes": expected,
        "risks": risks,
        "pass_chain": pass_chain,
        "relevant_quirks": relevant_quirks,
    }
    if parsed.openvino_device is not None:
        result["openvino_device"] = parsed.openvino_device
    return result
