"""Tests for normalization helpers."""

import pytest

from olive_mcp_server.tools.normalization import (
    normalize_framework,
    normalize_hardware,
    normalize_model,
    parse_hardware_target,
)


@pytest.mark.parametrize(
    ("input", "expected"),
    [
        ("torch", "PyTorch"),
        ("PyTorch", "PyTorch"),
        ("hf", "PyTorch"),
        ("HuggingFace", "PyTorch"),
        ("onnx", "ONNX"),
        ("tf", "TensorFlow"),
        ("TensorFlow", "TensorFlow"),
        ("tflite", "tflite"),
    ],
)
def test_normalize_framework(input: str, expected: str) -> None:
    assert normalize_framework(input) == expected


@pytest.mark.parametrize(
    ("input", "expected"),
    [
        ("mistralai/Mistral-7B-v0.1", "Mistral 7B"),
        ("Mistral-7B-Instruct", "Mistral 7B"),
        ("microsoft/phi-3-mini", "Phi-3-mini"),
        ("phi3-mini", "Phi-3-mini"),
        ("microsoft/Phi-4", "Phi-4"),
        ("phi4-mini", "Phi-4"),
        ("resnet-101", "ResNet-50"),
        ("openai/whisper-base", "Whisper"),
        ("meta-llama/Meta-Llama-3-8B", "Llama 3 8B"),
        ("llama-3-8b-instruct", "Llama 3 8B"),
        ("meta-llama/Llama-2-7b-hf", "Llama 2 7B"),
        ("llama2-7b", "Llama 2 7B"),
        ("codellama/CodeLlama-7b-Instruct-hf", "CodeLlama 7B"),
        ("code-llama-7b", "CodeLlama 7B"),
        ("deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "DeepSeek-R1-Distill-Llama 8B"),
        ("deepseek-r1-8b", "DeepSeek-R1-Distill-Llama 8B"),
        ("compvis/stable-diffusion-v1-4", "Stable Diffusion v1.4"),
        ("stable-diffusion", "Stable Diffusion v1.4"),
        ("google-bert/bert-base-uncased", "BERT-base"),
        ("bert-base", "BERT-base"),
        ("google/vit-base-patch16-224", "ViT-base"),
        ("t5-small", "T5-small"),
        ("gpt2", "GPT-2"),
        ("tiiuae/falcon-7b", "Falcon 7B"),
        ("ultralytics/yolov8", "YOLOv8"),
        ("mistralai/Mixtral-8x7B-Instruct-v0.1", "Mixtral 8x7B"),
        ("unrelated-model", "unrelated-model"),
    ],
)
def test_normalize_model(input: str, expected: str) -> None:
    assert normalize_model(input) == expected


def test_normalize_model_no_false_substring_match() -> None:
    """A name that merely contains an alias as part of a larger token should not match."""
    assert normalize_model("mymistralmodel") == "mymistralmodel"


@pytest.mark.parametrize(
    ("input", "expected"),
    [
        ("NVIDIA RTX 4090", "NVIDIA RTX 4090"),
        ("nvidia rtx 4090", "NVIDIA RTX 4090"),
        ("RTX 4090", "NVIDIA RTX 4090"),
        ("NVIDIA RTX 4090 Super", "NVIDIA RTX 4090"),
        ("T4", "NVIDIA T4"),
        ("Azure ML N-series", "Azure ML N-series"),
        ("azure ml", "Azure ML N-series"),
        ("NvTensorRTRTXExecutionProvider", "NVIDIA TensorRT RTX"),
        ("DmlExecutionProvider", "Windows DirectML GPU"),
        ("DirectMLExecutionProvider", "Windows DirectML GPU"),
        ("WebGpuExecutionProvider", "WebGPU (Browser)"),
        ("tensorrt rtx", "NVIDIA TensorRT RTX"),
        ("openvino npu", "Intel Core Ultra NPU (OpenVINO)"),
        ("intel npu", "Intel Core Ultra NPU (OpenVINO)"),
        ("directml", "Windows DirectML GPU"),
        ("webgpu", "WebGPU (Browser)"),
        ("npu", "Qualcomm Snapdragon NPU"),
        ("OpenVINOExecutionProvider", "Intel Core i9 CPU"),
        ("openvino", "Intel Core i9 CPU"),
        ("openvino gpu", "Intel iGPU / OpenVINO"),
        ("intel gpu", "Intel iGPU / OpenVINO"),
        ("igpu", "Intel iGPU / OpenVINO"),
        ("openvino arc", "Intel Arc A770"),
        ("intel arc", "Intel Arc A770"),
    ],
)
def test_normalize_hardware(input: str, expected: str) -> None:
    assert normalize_hardware(input) == expected


def test_normalize_hardware_unknown() -> None:
    assert normalize_hardware("MadeUpChip 9000") == "MadeUpChip 9000"


def test_normalize_hardware_invalid_ov_device_returns_stripped_input() -> None:
    """Invalid OV device must not CPU-fallback; adapter returns stripped input."""
    assert normalize_hardware("openvino:tpu") == "openvino:tpu"
    assert normalize_hardware("  openvino:tpu  ") == "openvino:tpu"


@pytest.mark.parametrize(
    ("raw", "profile", "openvino_device"),
    [
        ("OpenVINOExecutionProvider", "Intel Core i9 CPU", "CPU"),
        ("openvino", "Intel Core i9 CPU", "CPU"),
        ("openvino npu", "Intel Core Ultra NPU (OpenVINO)", "NPU"),
        ("openvino:npu", "Intel Core Ultra NPU (OpenVINO)", "NPU"),
        ("openvino+npu", "Intel Core Ultra NPU (OpenVINO)", "NPU"),
        ("OpenVINOExecutionProvider:NPU", "Intel Core Ultra NPU (OpenVINO)", "NPU"),
        ("intel npu", "Intel Core Ultra NPU (OpenVINO)", "NPU"),
        ("openvino gpu", "Intel iGPU / OpenVINO", "GPU"),
        ("openvino:gpu", "Intel iGPU / OpenVINO", "GPU"),
        ("intel gpu", "Intel iGPU / OpenVINO", "GPU"),
        ("igpu", "Intel iGPU / OpenVINO", "GPU"),
        ("openvino arc", "Intel Arc A770", "GPU"),
        ("intel arc", "Intel Arc A770", "GPU"),
        ("npu", "Qualcomm Snapdragon NPU", None),
        ("NvTensorRTRTXExecutionProvider", "NVIDIA TensorRT RTX", None),
        ("directml", "Windows DirectML GPU", None),
        ("webgpu", "WebGPU (Browser)", None),
        ("DmlExecutionProvider", "Windows DirectML GPU", None),
        ("WebGpuExecutionProvider", "WebGPU (Browser)", None),
    ],
)
def test_parse_hardware_target_acceptance(raw: str, profile: str, openvino_device: str | None) -> None:
    target = parse_hardware_target(raw)
    assert target.error is None
    assert target.profile == profile
    assert target.openvino_device == openvino_device


def test_parse_hardware_target_invalid_ov_device() -> None:
    target = parse_hardware_target("openvino:tpu")
    assert target.error is not None
    assert "tpu" in target.error.lower()
    assert target.profile == ""
    assert target.openvino_device is None


def test_parse_hardware_target_bare_gpu_not_ov() -> None:
    """Bare 'gpu' must not be claimed by OpenVINO or WebGPU reverse match."""
    target = parse_hardware_target("gpu")
    assert target.error is None
    assert target.openvino_device is None
    # Unresolved generic GPU fallback (not WebGPU / DirectML / OpenVINO).
    assert target.profile == "gpu"
    assert target.profile != "WebGPU (Browser)"
    assert target.profile != "Windows DirectML GPU"
    assert target.profile != "Intel iGPU / OpenVINO"
    assert target.profile != "Intel Arc A770"
    assert target.profile != "Intel Core Ultra NPU (OpenVINO)"
    assert target.profile != "Intel Core i9 CPU"


@pytest.mark.parametrize(
    "ep_id",
    [
        "dmlexecutionprovider",
        "DmlExecutionProvider",
        "webgpuexecutionprovider",
        "nvtensorrtrtxexecutionprovider",
    ],
)
def test_parse_hardware_target_ep_map_is_case_insensitive(ep_id: str) -> None:
    target = parse_hardware_target(ep_id)
    assert target.error is None
    assert target.profile in {
        "Windows DirectML GPU",
        "WebGPU (Browser)",
        "NVIDIA TensorRT RTX",
    }
