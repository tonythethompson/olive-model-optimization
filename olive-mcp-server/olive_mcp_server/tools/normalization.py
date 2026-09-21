"""Shared normalization helpers for MCP tool inputs.

These map user-supplied strings (which vary in casing and phrasing) to the
canonical values used as keys in the knowledge base JSON files.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from . import load_hardware_profiles

_FRAMEWORK_ALIASES = {
    "torch": "PyTorch",
    "pytorch": "PyTorch",
    "hf": "PyTorch",
    "huggingface": "PyTorch",
    "onnx": "ONNX",
    "tf": "TensorFlow",
    "tensorflow": "TensorFlow",
}

_MODEL_ALIASES = {
    "mistral": "Mistral 7B",
    "phi-3": "Phi-3-mini",
    "phi3": "Phi-3-mini",
    "phi-4": "Phi-4",
    "phi4": "Phi-4",
    "resnet": "ResNet-50",
    "whisper": "Whisper",
    "meta-llama/Meta-Llama-3-8B": "Llama 3 8B",
    "llama-3-8b": "Llama 3 8B",
    "llama3-8b": "Llama 3 8B",
    "llama 3": "Llama 3 8B",
    "llama-3": "Llama 3 8B",
    "llama3": "Llama 3 8B",
    "meta-llama/Llama-2-7b-hf": "Llama 2 7B",
    "llama-2-7b": "Llama 2 7B",
    "llama2-7b": "Llama 2 7B",
    "llama 2": "Llama 2 7B",
    "llama-2": "Llama 2 7B",
    "llama2": "Llama 2 7B",
    "codellama/CodeLlama-7b-Instruct-hf": "CodeLlama 7B",
    "codellama-7b": "CodeLlama 7B",
    "code-llama": "CodeLlama 7B",
    "codellama": "CodeLlama 7B",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": "DeepSeek-R1-Distill-Llama 8B",
    "deepseek-r1-distill-llama-8b": "DeepSeek-R1-Distill-Llama 8B",
    "deepseek-r1": "DeepSeek-R1-Distill-Llama 8B",
    "deepseek": "DeepSeek-R1-Distill-Llama 8B",
    "compvis/stable-diffusion-v1-4": "Stable Diffusion v1.4",
    "stable-diffusion-v1-4": "Stable Diffusion v1.4",
    "stable diffusion": "Stable Diffusion v1.4",
    "stable-diffusion": "Stable Diffusion v1.4",
    "google-bert/bert-base-uncased": "BERT-base",
    "bert-base": "BERT-base",
    "bert": "BERT-base",
    "mobilenetv2": "MobileNetV2",
    "mobilenet v2": "MobileNetV2",
    "efficientnet-b0": "EfficientNet-B0",
    "efficientnet b0": "EfficientNet-B0",
    "google/vit-base-patch16-224": "ViT-base",
    "vit-base": "ViT-base",
    "vit base": "ViT-base",
    "t5-small": "T5-small",
    "t5 small": "T5-small",
    "openai-gpt/gpt2": "GPT-2",
    "gpt2": "GPT-2",
    "gpt-2": "GPT-2",
    "tiiuae/falcon-7b": "Falcon 7B",
    "falcon-7b": "Falcon 7B",
    "falcon 7b": "Falcon 7B",
    "ultralytics/yolov8": "YOLOv8",
    "yolov8": "YOLOv8",
    "yolo v8": "YOLOv8",
    "mistralai/Mixtral-8x7B-Instruct-v0.1": "Mixtral 8x7B",
    "mixtral-8x7b": "Mixtral 8x7B",
    "mixtral 8x7b": "Mixtral 8x7B",
    "azure openai whisper": "Azure OpenAI Whisper",
}

_EXECUTION_PROVIDER_TO_TARGET = {
    "CPUExecutionProvider": "Intel Core i9 CPU",
    "CUDAExecutionProvider": "NVIDIA RTX 4090",
    "TensorrtExecutionProvider": "NVIDIA RTX 4090",
    "NvTensorRTRTXExecutionProvider": "NVIDIA TensorRT RTX",
    # OpenVINO EP bare form is handled by parse_hardware_target structured path
    # (sets openvino_device=CPU). Kept here as a defensive fallback profile only.
    "OpenVINOExecutionProvider": "Intel Core i9 CPU",
    "QNNExecutionProvider": "Qualcomm Snapdragon NPU",
    "ROCMExecutionProvider": "AMD MI300X / ROCm",
    "DmlExecutionProvider": "Windows DirectML GPU",
    "DirectMLExecutionProvider": "Windows DirectML GPU",  # legacy alias only (canonical EP id is DmlExecutionProvider)
    "WebGpuExecutionProvider": "WebGPU (Browser)",
    "CoreMLExecutionProvider": "Apple M2/M3 (CoreML)",
    "NNAPIExecutionProvider": "Android NNAPI",
    "VitisAIExecutionProvider": "Xilinx Vitis AI DPU",
    "SNPEExecutionProvider": "Qualcomm SNPE (Legacy)",
    "TensorflowLiteExecutionProvider": "TensorFlow Lite Export",
    "XnnpackExecutionProvider": "XNNPACK (Mobile)",
    "WasmExecutionProvider": "WASM (Browser)",
}

_EXECUTION_PROVIDER_TO_TARGET_LOWER = {key.lower(): value for key, value in _EXECUTION_PROVIDER_TO_TARGET.items()}

# Exact lowercase keys only (full stripped input). Applied after EP map, before
# profile exact/substring match. Do not use loose substrings (e.g. bare "rtx").
# Bare "gpu" is handled as an explicit unresolved fallback (must not become
# OV/WebGPU/DML via reverse profile substring match).
# OpenVINO inputs are handled solely by _try_parse_openvino (not listed here).
_HARDWARE_ALIASES = {
    "tensorrt rtx": "NVIDIA TensorRT RTX",
    "trt rtx": "NVIDIA TensorRT RTX",
    "nvtensorrtrtx": "NVIDIA TensorRT RTX",
    "tensorrt-rtx": "NVIDIA TensorRT RTX",
    "directml": "Windows DirectML GPU",
    "dml": "Windows DirectML GPU",
    "webgpu": "WebGPU (Browser)",
    "web gpu": "WebGPU (Browser)",
    "ort web": "WebGPU (Browser)",
    "coreml": "Apple M2/M3 (CoreML)",
    "nnapi": "Android NNAPI",
    "vitisai": "Xilinx Vitis AI DPU",
    "vitis-ai": "Xilinx Vitis AI DPU",
    "snpe": "Qualcomm SNPE (Legacy)",
    "tflite": "TensorFlow Lite Export",
    "xnnpack": "XNNPACK (Mobile)",
    "wasm": "WASM (Browser)",
}

OpenVinoDevice = Literal["CPU", "GPU", "NPU"]

_OV_DEVICE_CANONICAL: dict[str, OpenVinoDevice] = {
    "cpu": "CPU",
    "gpu": "GPU",
    "npu": "NPU",
}

_OV_DEVICE_PROFILES: dict[OpenVinoDevice, str] = {
    "CPU": "Intel Core i9 CPU",
    "GPU": "Intel iGPU / OpenVINO",
    "NPU": "Intel Core Ultra NPU (OpenVINO)",
}

_ARC_PROFILE = "Intel Arc A770"

# OpenVINO EP / shorthand prefix + optional device separator (: + or whitespace).
_OV_PREFIX_DEVICE_RE = re.compile(r"^(openvinoexecutionprovider|openvino)(?:[:\+\s]+([a-z0-9]+))?$")

_hardware_profiles_cache: list[dict] | None = None


@dataclass(frozen=True)
class HardwareTarget:
    """Canonical hardware profile plus optional OpenVINO device selection.

    Attributes:
        profile: Canonical hardware profile name, or "" when invalid OV device.
        openvino_device: CPU|GPU|NPU when the input selected an OpenVINO path.
        error: Descriptive error when an OV device token is invalid (no CPU fallback).
    """

    profile: str
    openvino_device: OpenVinoDevice | None = None
    error: str | None = None


def clear_hardware_profiles_cache() -> None:
    """Drop the module-level hardware profile cache (tests / reload)."""
    global _hardware_profiles_cache
    _hardware_profiles_cache = None


def _get_hardware_profiles() -> list[dict]:
    """Return cached hardware profiles, loading them once."""
    global _hardware_profiles_cache
    if _hardware_profiles_cache is None:
        _hardware_profiles_cache = load_hardware_profiles()
    return _hardware_profiles_cache


def _is_word_boundary(text: str, index: int, length: int) -> bool:
    """Return True when text[index:index+length] is bounded by non-alphanumeric chars or string edges."""
    if index > 0 and text[index - 1].isalnum():
        return False
    end = index + length
    return not (end < len(text) and text[end].isalnum())


def _invalid_openvino_device(token: str) -> HardwareTarget:
    """Build an error target for an unrecognized OpenVINO device token."""
    return HardwareTarget(
        profile="",
        openvino_device=None,
        error=(f"Invalid OpenVINO device '{token}'. Expected one of: CPU, GPU, NPU."),
    )


def _openvino_target_for_device(device: OpenVinoDevice) -> HardwareTarget:
    """Map a canonical OpenVINO device to its profile + device pair."""
    return HardwareTarget(
        profile=_OV_DEVICE_PROFILES[device],
        openvino_device=device,
    )


def _try_parse_openvino(lower: str) -> HardwareTarget | None:
    """Parse OpenVINO-shaped inputs into a HardwareTarget.

    Returns None when the input is not an OpenVINO structured form so the
    caller can continue with EP map / aliases / profile matching.

    Bare ``gpu`` is intentionally not treated as OpenVINO.
    """
    # Arc discrete GPU forms → Arc profile + GPU device.
    if lower in {
        "openvino arc",
        "intel arc",
        "openvino:arc",
        "openvino+arc",
        "openvinoexecutionprovider:arc",
        "openvinoexecutionprovider+arc",
    }:
        return HardwareTarget(profile=_ARC_PROFILE, openvino_device="GPU")

    # Explicit iGPU / Intel GPU forms (not bare "gpu").
    if lower in {
        "igpu",
        "intel igpu",
        "openvino igpu",
        "intel gpu",
        "openvino gpu",
        "openvino:gpu",
        "openvino+gpu",
        "openvinoexecutionprovider:gpu",
        "openvinoexecutionprovider+gpu",
    }:
        return _openvino_target_for_device("GPU")

    # Intel / Core Ultra NPU shorthand (not bare "npu" — that stays Qualcomm).
    if lower in {"intel npu", "core ultra npu"}:
        return _openvino_target_for_device("NPU")

    match = _OV_PREFIX_DEVICE_RE.fullmatch(lower)
    if match is None:
        return None

    device_token = match.group(2)
    if device_token is None:
        # Bare openvino / OpenVINOExecutionProvider → CPU device.
        return _openvino_target_for_device("CPU")

    if device_token == "arc":
        return HardwareTarget(profile=_ARC_PROFILE, openvino_device="GPU")
    if device_token == "igpu":
        return _openvino_target_for_device("GPU")

    canonical = _OV_DEVICE_CANONICAL.get(device_token)
    if canonical is None:
        return _invalid_openvino_device(device_token)
    return _openvino_target_for_device(canonical)


def _match_hardware_profile(lower: str, fallback_name: str) -> str:
    """Resolve lowercased input against known hardware profiles."""
    # Bare "gpu" matches many profile names via reverse substring ("WebGPU",
    # "DirectML GPU", …). Keep it unresolved / non-OpenVINO instead.
    if lower == "gpu":
        return fallback_name

    profiles = _get_hardware_profiles()

    for profile in profiles:
        if profile["target"].lower() == lower:
            return profile["target"]

    # Forward substring: profile target is contained in the input
    # (e.g. "NVIDIA RTX 4090" in "NVIDIA RTX 4090 Super").
    forward = [p for p in profiles if p["target"].lower() in lower]
    if forward:
        forward.sort(key=lambda p: len(p["target"]), reverse=True)
        return forward[0]["target"]

    # Reverse substring: input is contained in a profile target
    # (e.g. "RTX 4090" in "NVIDIA RTX 4090").
    # Shortest match first so bare "npu" stays Qualcomm Snapdragon NPU
    # (shorter than "Intel Core Ultra NPU (OpenVINO)").
    reverse = [p for p in profiles if lower in p["target"].lower()]
    if reverse:
        reverse.sort(key=lambda p: len(p["target"]))
        return reverse[0]["target"]

    return fallback_name


def parse_hardware_target(raw: str) -> HardwareTarget:
    """Parse a hardware target string into a structured HardwareTarget.

    Pure regarding caller-visible state: no cache writes beyond the existing
    read-through profile cache, no global mutation of user data.

    Match order:
      1. strip
      2. OpenVINO / EP+device structured parse
      3. execution-provider map (case-insensitive)
      4. exact hardware aliases
      5. profile exact / forward / reverse (shortest-npu rule; bare gpu fallback)
    """
    name = raw.strip()
    if not name:
        return HardwareTarget(profile="")

    lower = name.lower()

    ov_target = _try_parse_openvino(lower)
    if ov_target is not None:
        return ov_target

    # Map ONNX Runtime execution-provider strings to canonical hardware targets.
    # Lookup is case-insensitive so mixed/lowercase EP ids resolve before aliases.
    ep_target = _EXECUTION_PROVIDER_TO_TARGET_LOWER.get(lower)
    if ep_target is not None:
        name = ep_target
        lower = name.lower()

    # Exact alias match on the full lowercased input (after EP map).
    alias_target = _HARDWARE_ALIASES.get(lower)
    if alias_target is not None:
        name = alias_target
        lower = name.lower()

    profile = _match_hardware_profile(lower, fallback_name=name)
    return HardwareTarget(profile=profile)


def normalize_framework(framework: str) -> str:
    """Canonicalize a framework name, e.g. 'torch' -> 'PyTorch'."""
    name = framework.strip()
    return _FRAMEWORK_ALIASES.get(name.lower(), name)


def normalize_model(model_name: str) -> str:
    """Map a model name or HuggingFace ID to its compatibility-matrix key.

    Falls back to the stripped input if no known alias matches, so callers
    can still detect "not found in the local matrix" cases.
    """
    name = model_name.strip()
    lower = name.lower()
    # Longer aliases first so overlapping names resolve to the most specific match.
    for alias in sorted(_MODEL_ALIASES, key=len, reverse=True):
        start = 0
        while True:
            idx = lower.find(alias, start)
            if idx == -1:
                break
            if _is_word_boundary(lower, idx, len(alias)):
                return _MODEL_ALIASES[alias]
            start = idx + 1
    return name


def normalize_hardware(target_hardware: str) -> str:
    """Match a hardware target to its canonical profile name.

    Thin adapter over :func:`parse_hardware_target`. On invalid OpenVINO device
    tokens returns the stripped input (no CPU fallback). Otherwise returns the
    canonical profile string for back-compat callers.
    """
    target = parse_hardware_target(target_hardware)
    if target.error:
        return target_hardware.strip()
    return target.profile
