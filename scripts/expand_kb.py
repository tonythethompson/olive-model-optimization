"""Expand knowledge base files to target counts for production coverage.

Emits deterministic refresh metadata (source timestamp, generator version,
changed files) for CI/PR workflows. Does not embed wall-clock timestamps in
KB content — no-op runs produce no file diffs.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

KB_DIR = Path(__file__).parent.parent / "olive_mcp_server" / "knowledge_base"
GENERATOR_NAME = "expand_kb"
REFRESH_METADATA_NAME = "refresh_metadata.json"
KB_REL_PREFIX = "olive_mcp_server/knowledge_base"

# Track files touched during this run for refresh metadata.
_CHANGED_FILES: list[str] = []


def _generator_version() -> str:
    """Return the package/generator version used in refresh metadata."""
    try:
        # Package root is on path when installed; fall back when run as script.
        import sys

        mcp_dir = Path(__file__).resolve().parent.parent
        if str(mcp_dir) not in sys.path:
            sys.path.insert(0, str(mcp_dir))
        from olive_mcp_server import __version__

        return str(__version__)
    except Exception:  # noqa: BLE001
        return "0.1.0"


def _canonical_json(data: Any) -> str:
    """Serialize data to a stable JSON string for hashing and comparisons."""
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _content_fingerprint(data: Any) -> str:
    """SHA-256 hex digest of canonical JSON for content-addressed identity."""
    return hashlib.sha256(_canonical_json(data).encode("utf-8")).hexdigest()


def _dump_json_text(data: Any) -> str:
    """Pretty-print JSON with a trailing newline (stable formatting)."""
    return json.dumps(data, indent=2, ensure_ascii=False) + "\n"


def _kb_rel(name: str) -> str:
    """Repo-relative path for a knowledge_base file (workflow-friendly)."""
    return f"{KB_REL_PREFIX}/{name}"


def load(name: str) -> dict:
    with open(KB_DIR / name, encoding="utf-8") as f:
        return json.load(f)


def save(name: str, data: dict) -> None:
    """Write KB JSON only when semantic content differs (no no-op diffs).

    Compares canonical JSON so formatting-only differences do not rewrite
    tracked KB files or inflate changed_files on a no-op refresh.
    """
    path = KB_DIR / name
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
            if _canonical_json(existing) == _canonical_json(data):
                return
        except (OSError, json.JSONDecodeError):
            pass
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_dump_json_text(data), encoding="utf-8")
    rel = _kb_rel(name)
    if rel not in _CHANGED_FILES:
        _CHANGED_FILES.append(rel)


def expand_passes():
    data = load("passes.json")
    existing = {p["name"]: p for p in data["passes"]}

    new_passes = [
        {
            "name": "IncDynamicQuantization",
            "type": "quantization",
            "class": "olive.passes.inc_quantization.IncDynamicQuantization",
            "description": "Intel Neural Compressor dynamic quantization.",
            "input_formats": ["torch", "onnx"],
            "output_formats": ["torch", "onnx"],
            "required_params": [],
            "optional_params": {
                "weight_type": {"type": "str", "default": "int8", "enum": ["int8", "uint8"]},
                "per_channel": {"type": "bool", "default": False},
                "reduce_range": {"type": "bool", "default": False},
            },
            "hardware_requirements": ["CPUExecutionProvider", "OpenVINOExecutionProvider"],
            "typical_compression": "60-70%",
            "gotchas": ["No calibration data required; accuracy may lag static quantization."],
        },
        {
            "name": "IncStaticQuantization",
            "type": "quantization",
            "class": "olive.passes.inc_quantization.IncStaticQuantization",
            "description": "Intel Neural Compressor static quantization.",
            "input_formats": ["torch", "onnx"],
            "output_formats": ["torch", "onnx"],
            "required_params": ["calibration_data_dir"],
            "optional_params": {
                "calibration_sampling_size": {"type": "int", "default": 100},
                "weight_type": {"type": "str", "default": "int8"},
                "scheme": {"type": "str", "default": "sym", "enum": ["sym", "asym"]},
                "per_channel": {"type": "bool", "default": False},
            },
            "hardware_requirements": ["CPUExecutionProvider", "OpenVINOExecutionProvider"],
            "typical_compression": "65-75%",
            "gotchas": ["Calibration data must be representative of inference distribution."],
        },
        {
            "name": "OnnxOpVersionConversion",
            "type": "conversion",
            "class": "olive.passes.onnx.conversion.OnnxOpVersionConversion",
            "description": "Convert an ONNX model to a newer opset version.",
            "input_formats": ["onnx"],
            "output_formats": ["onnx"],
            "required_params": ["target_opset"],
            "optional_params": {
                "target_opset": {"type": "int", "default": 14, "range": "7-21"},
            },
            "hardware_requirements": ["CPUExecutionProvider"],
            "typical_compression": "none",
            "gotchas": ["Higher opsets can break compatibility with older execution providers."],
        },
        {
            "name": "OpenVINOConversion",
            "type": "conversion",
            "class": "olive.passes.openvino.OpenVINOConversion",
            "description": "Convert a PyTorch or ONNX model to OpenVINO IR.",
            "input_formats": ["torch", "onnx"],
            "output_formats": ["openvino"],
            "required_params": [],
            "optional_params": {
                "input_shapes": {"type": "list[list[int]]", "default": []},
                "fp16": {"type": "bool", "default": False},
            },
            "hardware_requirements": ["OpenVINOExecutionProvider"],
            "typical_compression": "varies",
            "gotchas": ["OpenVINO plugin must be installed separately."],
        },
        {
            "name": "SNPEConversion",
            "type": "conversion",
            "class": "olive.passes.snpe.SNPEConversion",
            "description": "Convert to Qualcomm SNPE DLC format.",
            "input_formats": ["onnx"],
            "output_formats": ["snpe"],
            "required_params": [],
            "optional_params": {
                "input_list": {"type": "str", "default": ""},
            },
            "hardware_requirements": ["SNPEExecutionProvider"],
            "typical_compression": "varies",
            "gotchas": ["SNPE is Qualcomm's legacy runtime; prefer QNN for newer platforms."],
        },
        {
            "name": "TensorFlowConversion",
            "type": "conversion",
            "class": "olive.passes.tensorflow.TensorFlowConversion",
            "description": "Convert a model to TensorFlow SavedModel.",
            "input_formats": ["onnx", "torch"],
            "output_formats": ["tf"],
            "required_params": [],
            "optional_params": {},
            "hardware_requirements": ["TensorflowLiteExecutionProvider"],
            "typical_compression": "none",
            "gotchas": ["Not all ONNX ops map cleanly to TensorFlow."],
        },
        {
            "name": "OrtMixedPrecision",
            "type": "graph_optimization",
            "class": "olive.passes.onnx.mixed_precision.OrtMixedPrecision",
            "description": "Mixed int8/float32 precision conversion.",
            "input_formats": ["onnx"],
            "output_formats": ["onnx"],
            "required_params": [],
            "optional_params": {
                "op_block_list": {"type": "list[str]", "default": []},
            },
            "hardware_requirements": ["CUDAExecutionProvider"],
            "typical_compression": "30-40%",
            "gotchas": ["op_block_list prevents specific ops from being cast to FP16."],
        },
        {
            "name": "QNNPreprocess",
            "type": "graph_optimization",
            "class": "olive.passes.qnn.QNNPreprocess",
            "description": "Preprocess ONNX graph for Qualcomm QNN.",
            "input_formats": ["onnx"],
            "output_formats": ["onnx"],
            "required_params": [],
            "optional_params": {},
            "hardware_requirements": ["QNNExecutionProvider"],
            "typical_compression": "0-5%",
            "gotchas": ["May insert quantize/dequantize nodes and replace unsupported ops."],
        },
        {
            "name": "IncSparsityFineTuning",
            "type": "pruning",
            "class": "olive.passes.inc_pruning.IncSparsityFineTuning",
            "description": "Fine-tune after Intel Neural Compressor pruning.",
            "input_formats": ["torch"],
            "output_formats": ["torch"],
            "required_params": [],
            "optional_params": {
                "learning_rate": {"type": "float", "default": 1e-5},
                "train_epochs": {"type": "int", "default": 3},
            },
            "hardware_requirements": ["CPUExecutionProvider"],
            "typical_compression": "varies",
            "gotchas": ["Recovery accuracy depends on sparsity ratio."],
        },
        {
            "name": "SparsityFineTuning",
            "type": "pruning",
            "class": "olive.passes.sparsity.SparsityFineTuning",
            "description": "General sparsity fine-tuning pass.",
            "input_formats": ["torch"],
            "output_formats": ["torch"],
            "required_params": [],
            "optional_params": {
                "target_sparsity": {"type": "float", "default": 0.5, "range": "0.0-0.95"},
            },
            "hardware_requirements": ["CPUExecutionProvider"],
            "typical_compression": "40-60%",
            "gotchas": ["Sparse speedup requires sparse-aware execution provider."],
        },
        {
            "name": "DistillationPass",
            "type": "distillation",
            "class": "olive.passes.distillation.DistillationPass",
            "description": "General knowledge distillation pass.",
            "input_formats": ["torch"],
            "output_formats": ["torch"],
            "required_params": ["teacher_model"],
            "optional_params": {
                "temperature": {"type": "float", "default": 3.0},
                "loss_weights": {"type": "list[float]", "default": [1.0, 0.5]},
            },
            "hardware_requirements": ["CUDAExecutionProvider"],
            "typical_compression": "varies",
            "gotchas": ["Teacher and student logits must be compatible."],
        },
        {
            "name": "MultiLoRA",
            "type": "finetuning",
            "class": "olive.passes.lora.MultiLoRA",
            "description": "Multiple LoRA adapters for ONNX Runtime (experimental).",
            "input_formats": ["torch", "hf"],
            "output_formats": ["onnx"],
            "required_params": [],
            "optional_params": {
                "lora_rank": {"type": "int", "default": 8},
                "lora_alpha": {"type": "int", "default": 16},
            },
            "hardware_requirements": ["CUDAExecutionProvider"],
            "typical_compression": "n/a",
            "gotchas": ["Experimental; requires careful adapter routing."],
        },
        {
            "name": "ExtractLoRA",
            "type": "finetuning",
            "class": "olive.passes.lora.ExtractLoRA",
            "description": "Extract LoRA weights to separate adapter files.",
            "input_formats": ["torch"],
            "output_formats": ["torch"],
            "required_params": [],
            "optional_params": {},
            "hardware_requirements": ["CPUExecutionProvider"],
            "typical_compression": "n/a",
            "gotchas": ["Extracted adapters must be merged or loaded separately."],
        },
        {
            "name": "GenerateAdapterWeights",
            "type": "finetuning",
            "class": "olive.passes.lora.GenerateAdapterWeights",
            "description": "Generate adapter weight files for serving.",
            "input_formats": ["torch"],
            "output_formats": ["onnx"],
            "required_params": [],
            "optional_params": {
                "adapter_path": {"type": "str", "default": ""},
            },
            "hardware_requirements": ["CPUExecutionProvider"],
            "typical_compression": "n/a",
            "gotchas": ["Used for multi-adapter serving patterns."],
        },
        {
            "name": "FinetuningPass",
            "type": "finetuning",
            "class": "olive.passes.finetuning.FinetuningPass",
            "description": "General PyTorch fine-tuning pass.",
            "input_formats": ["torch", "hf"],
            "output_formats": ["torch"],
            "required_params": [],
            "optional_params": {
                "learning_rate": {"type": "float", "default": 1e-5},
                "num_train_epochs": {"type": "int", "default": 3},
                "batch_size": {"type": "int", "default": 8},
            },
            "hardware_requirements": ["CUDAExecutionProvider"],
            "typical_compression": "n/a",
            "gotchas": ["Full fine-tuning is memory intensive."],
        },
        {
            "name": "HuggingFaceFineTuning",
            "type": "finetuning",
            "class": "olive.passes.finetuning.HuggingFaceFineTuning",
            "description": "Hugging Face Trainer integration for fine-tuning.",
            "input_formats": ["hf"],
            "output_formats": ["torch"],
            "required_params": [],
            "optional_params": {
                "per_device_train_batch_size": {"type": "int", "default": 8},
                "num_train_epochs": {"type": "int", "default": 3},
            },
            "hardware_requirements": ["CUDAExecutionProvider"],
            "typical_compression": "n/a",
            "gotchas": ["Requires transformers, datasets, and accelerate."],
        },
        {
            "name": "ModelOptOptimizer",
            "type": "performance_tuning",
            "class": "olive.passes.nv_model_opt.ModelOptOptimizer",
            "description": "NVIDIA Model Optimizer integration.",
            "input_formats": ["torch", "onnx"],
            "output_formats": ["onnx"],
            "required_params": [],
            "optional_params": {
                "mode": {"type": "str", "default": "quantize", "enum": ["quantize", "prune", "export"]},
            },
            "hardware_requirements": ["CUDAExecutionProvider", "TensorrtExecutionProvider"],
            "typical_compression": "70-85%",
            "gotchas": ["Requires nvidia-modelopt package."],
        },
        {
            "name": "BatchSizeOptimization",
            "type": "performance_tuning",
            "class": "olive.passes.perf_tuning.BatchSizeOptimization",
            "description": "Search for optimal batch size.",
            "input_formats": ["onnx"],
            "output_formats": ["onnx"],
            "required_params": [],
            "optional_params": {
                "min_batch_size": {"type": "int", "default": 1},
                "max_batch_size": {"type": "int", "default": 64},
            },
            "hardware_requirements": ["CPUExecutionProvider", "CUDAExecutionProvider"],
            "typical_compression": "0%",
            "gotchas": ["Optimal batch depends on model size and memory."],
        },
        {
            "name": "OnnxGraphCapture",
            "type": "performance_tuning",
            "class": "olive.passes.onnx.graph_capture.OnnxGraphCapture",
            "description": "Capture ONNX computation graph for analysis.",
            "input_formats": ["onnx"],
            "output_formats": ["onnx"],
            "required_params": [],
            "optional_params": {
                "input_names": {"type": "list[str]", "default": []},
            },
            "hardware_requirements": ["CPUExecutionProvider"],
            "typical_compression": "0%",
            "gotchas": ["Used for debugging and graph inspection."],
        },
        {
            "name": "QNNConversion",
            "type": "conversion",
            "class": "olive.passes.qnn.QNNConversion",
            "description": "Convert ONNX model to QNN context binary.",
            "input_formats": ["onnx"],
            "output_formats": ["qnn"],
            "required_params": [],
            "optional_params": {
                "qnn_sdk_root": {"type": "str", "default": ""},
            },
            "hardware_requirements": ["QNNExecutionProvider"],
            "typical_compression": "varies",
            "gotchas": ["Requires Qualcomm QNN SDK."],
        },
        {
            "name": "AzureMLQuantization",
            "type": "quantization",
            "class": "olive.passes.azureml.AzureMLQuantization",
            "description": "Quantization pass running on Azure ML compute.",
            "input_formats": ["onnx"],
            "output_formats": ["onnx"],
            "required_params": ["compute_target"],
            "optional_params": {
                "compute_target": {"type": "str", "default": ""},
                "conda_environment": {"type": "str", "default": ""},
            },
            "hardware_requirements": ["CPUExecutionProvider"],
            "typical_compression": "70-80%",
            "gotchas": ["Requires Azure ML workspace and Olive AzureML extension."],
        },
        {
            "name": "OptimumQuantization",
            "type": "quantization",
            "class": "olive.passes.optimum.OptimumQuantization",
            "description": "Hugging Face Optimum quantization pass.",
            "input_formats": ["hf", "torch"],
            "output_formats": ["onnx"],
            "required_params": [],
            "optional_params": {
                "quantization_approach": {"type": "str", "default": "dynamic", "enum": ["dynamic", "static"]},
            },
            "hardware_requirements": ["CPUExecutionProvider"],
            "typical_compression": "65-75%",
            "gotchas": ["Optimum path is convenient but less tunable than Olive native passes."],
        },
        {
            "name": "OptimumConversion",
            "type": "conversion",
            "class": "olive.passes.optimum.OptimumConversion",
            "description": "Hugging Face Optimum ONNX conversion.",
            "input_formats": ["hf"],
            "output_formats": ["onnx"],
            "required_params": [],
            "optional_params": {
                "opset": {"type": "int", "default": 14},
            },
            "hardware_requirements": ["CPUExecutionProvider"],
            "typical_compression": "none",
            "gotchas": ["Uses Optimum exporters; check supported architectures."],
        },
        {
            "name": "OptimumGraphOptimization",
            "type": "graph_optimization",
            "class": "olive.passes.optimum.OptimumGraphOptimization",
            "description": "Hugging Face Optimum ONNX graph optimizations.",
            "input_formats": ["onnx"],
            "output_formats": ["onnx"],
            "required_params": [],
            "optional_params": {
                "optimization_level": {"type": "int", "default": 1, "range": "0-99"},
            },
            "hardware_requirements": ["CPUExecutionProvider"],
            "typical_compression": "0-10%",
            "gotchas": ["Wraps Optimum optimizer; less control than native Olive passes."],
        },
    ]

    for p in new_passes:
        if p["name"] not in existing:
            data["passes"].append(p)

    save("passes.json", data)
    print(f"passes.json now has {len(data['passes'])} passes")


def expand_hardware_profiles():
    data = load("hardware_profiles.json")
    existing = {p["target"]: p for p in data["profiles"]}

    new_profiles = [
        {
            "target": "NVIDIA A100",
            "accelerator": "gpu",
            "execution_providers": ["CUDAExecutionProvider", "TensorrtExecutionProvider"],
            "recommended_passes": ["OnnxConversion", "NVModelOptQuantization", "OnnxFloatToFloat16"],
            "typical_speedup": "10-20x",
            "calibration_size": 512,
            "optimal_batch_size": 64,
            "memory_gb": 80,
            "ops_supported": ["Conv", "Gemm", "Attention", "LayerNormalization", "GELU"],
            "known_issues": ["Large batch sizes may need gradient checkpointing during calibration."],
            "notes": "A100 supports FP16/INT8/TensorRT; best for large LLMs and batch inference.",
        },
        {
            "target": "AMD MI300X / ROCm",
            "accelerator": "gpu",
            "execution_providers": ["ROCMExecutionProvider"],
            "recommended_passes": ["OnnxConversion", "OnnxStaticQuantization", "OnnxModelOptimizer"],
            "typical_speedup": "3-6x",
            "calibration_size": 256,
            "optimal_batch_size": 16,
            "memory_gb": 192,
            "ops_supported": ["Conv", "Gemm", "Attention"],
            "known_issues": ["Operator coverage lags CUDA; verify with ROCm EP whitelist."],
            "notes": "Use ONNX Runtime ROCm EP; MI300X has large HBM for big models.",
        },
        {
            "target": "NVIDIA Jetson Orin",
            "accelerator": "gpu",
            "execution_providers": ["TensorrtExecutionProvider", "CUDAExecutionProvider"],
            "recommended_passes": ["OnnxConversion", "OnnxStaticQuantization", "OnnxFloatToFloat16"],
            "typical_speedup": "4-8x",
            "calibration_size": 128,
            "optimal_batch_size": 1,
            "memory_gb": 32,
            "ops_supported": ["Conv", "Gemm", "Attention"],
            "known_issues": ["Limited memory; use external data format for large models."],
            "notes": "Best for edge vision models with TensorRT INT8.",
        },
        {
            "target": "Intel Arc A770",
            "accelerator": "gpu",
            "execution_providers": ["OpenVINOExecutionProvider"],
            "recommended_passes": ["OnnxConversion", "OpenVINOConversion", "OnnxStaticQuantization"],
            "typical_speedup": "3-6x",
            "calibration_size": 128,
            "optimal_batch_size": 8,
            "memory_gb": 16,
            "ops_supported": ["Conv", "Gemm", "Attention", "LayerNormalization"],
            "known_issues": ["OpenVINO GPU plugin must be installed."],
            "notes": "Use OpenVINO FP16 or INT8 for Arc GPUs.",
        },
        {
            "target": "Raspberry Pi 5 (CPU)",
            "accelerator": "cpu",
            "execution_providers": ["CPUExecutionProvider"],
            "recommended_passes": ["OnnxConversion", "OnnxStaticQuantization", "OnnxModelOptimizer"],
            "typical_speedup": "2-3x",
            "calibration_size": 64,
            "optimal_batch_size": 1,
            "memory_gb": 8,
            "ops_supported": ["Conv", "Gemm", "MaxPool"],
            "known_issues": ["Very limited memory; small models and small calibration sets only."],
            "notes": "Quantize to INT8 and keep the model under 2GB.",
        },
        {
            "target": "NVIDIA TensorRT RTX",
            "accelerator": "gpu",
            "execution_providers": ["NvTensorRTRTXExecutionProvider"],
            "recommended_passes": ["OnnxConversion", "NVModelOptQuantization", "OnnxFloatToFloat16"],
            "typical_speedup": "5-12x",
            "calibration_size": 128,
            "optimal_batch_size": 8,
            "memory_gb": 12,
            "ops_supported": ["Conv", "Gemm", "Attention", "LayerNormalization", "GELU"],
            "known_issues": [
                "The tensorrt-rtx package installs the standalone TensorRT RTX EP-ABI provider library implementing NvTensorRTRTXExecutionProvider for ONNX Runtime 1.23+ (not the full TensorRT / nvinfer SDK). Requires SM ≥ 7.5 (Turing+).",
                "PTQ INT8/QDQ path is a poor fit for tensorrt-rtx; prefer AWQ INT4.",
            ],
            "notes": "Consumer RTX path via JIT TensorRT engines; distinct from full TensorrtExecutionProvider / nvinfer SDK. Align with Olive Studio provider catalog TRT RTX.",
        },
        {
            "target": "Intel Core Ultra NPU (OpenVINO)",
            "accelerator": "npu",
            "execution_providers": ["OpenVINOExecutionProvider"],
            "recommended_passes": [
                "OpenVINOConversion",
                "OpenVINOWeightCompression",
            ],
            "typical_speedup": "3-8x",
            "calibration_size": 100,
            "optimal_batch_size": 1,
            "memory_gb": 16,
            "ops_supported": ["Conv", "Gemm", "Attention", "LayerNormalization"],
            "known_issues": [
                "NPU plugin requires Intel Core Ultra / Meteor Lake+ with OpenVINO NPU driver.",
                "Unsupported ops fall back to CPU; verify OpenVINO device=NPU.",
                'Olive Studio sets openvinoTargetDevice: "NPU" separately from EP id.',
                "OpenVINOOptimumConversion is torch/HF-only; use OpenVINOConversion for ONNX sources.",
            ],
            "notes": "Maps app OpenVINO + NPU device target. Prefer INT8/weight-compression paths documented for OpenVINO. Default chain uses OpenVINOConversion (torch|onnx) before OpenVINOWeightCompression.",
        },
        {
            "target": "Windows DirectML GPU",
            "accelerator": "gpu",
            "execution_providers": ["DmlExecutionProvider"],
            "recommended_passes": ["OnnxConversion", "OnnxModelOptimizer", "OnnxStaticQuantization"],
            "typical_speedup": "2-5x",
            "calibration_size": 128,
            "optimal_batch_size": 8,
            "memory_gb": 8,
            "ops_supported": ["Conv", "Gemm", "Attention", "MaxPool"],
            "known_issues": [
                "Windows 10/11 + DirectX 12 only; needs onnxruntime-directml.",
                "Operator coverage differs from CUDA; validate graph.",
            ],
            "notes": "Olive Studio default-family venv path for DirectML. Prefer INT8 PTQ. Do not mix CUDA/TRT packages into this EP path.",
        },
        {
            "target": "WebGPU (Browser)",
            "accelerator": "gpu",
            "execution_providers": ["WebGpuExecutionProvider"],
            "recommended_passes": ["OnnxConversion", "OnnxModelOptimizer", "OnnxFloatToFloat16"],
            "typical_speedup": "1-3x",
            "calibration_size": 64,
            "optimal_batch_size": 1,
            "memory_gb": 4,
            "ops_supported": ["Conv", "Gemm", "Softmax"],
            "known_issues": [
                "Browser WebGPU only (ORT Web); not a local Olive CLI/EP execution target in Studio.",
                "Olive Studio blocks local run for WebGPU (isRunnable: false / localExecutionIssues).",
                "INT8 support varies by browser/GPU.",
            ],
            "notes": "Export/optimize ONNX for in-browser ORT Web WebGPU. Prefer FP16. Use Studio for recipe build/export, not Execute Live.",
        },
    ]

    for p in new_profiles:
        if p["target"] not in existing:
            data["profiles"].append(p)

    save("hardware_profiles.json", data)
    print(f"hardware_profiles.json now has {len(data['profiles'])} profiles")


def expand_troubleshooting():
    """Add predefined troubleshooting entries to the knowledge base and report the resulting entry count."""
    data = load("troubleshooting.json")
    entries = {e["id"]: e for e in data.get("entries", [])}

    new_entries = [
        {
            "id": "awq-slow-calibration",
            "patterns": ["AWQ", "nvidia-modelopt", "awq calibration", "activation-aware"],
            "title": "AWQ calibration is slow or hangs",
            "root_cause": "AWQ scans weights and activations; large models or large calibration sets take time.",
            "solution": "Reduce calibration iters to 32; use a small subset (128 samples); ensure GPU memory is available.",
            "updated_config": {
                "passes": {
                    "NVModelOptQuantization": {
                        "params": {"algorithm": "awq", "calibration_iters": 32, "weight_bits": 4}
                    }
                }
            },
        },
        {
            "id": "qnn-layer-not-supported",
            "patterns": ["QNN", "not supported", "unsupported op", "LayerNormalization", "GELU"],
            "title": "QNN reports unsupported operator",
            "root_cause": "Qualcomm QNN has a limited op whitelist; LayerNorm and GELU are often unsupported.",
            "solution": "Run QNNPreprocess before QNNQuantization; use OnnxModelOptimizer to fuse LayerNorm; consider replacing GELU with approximate ops if the model allows.",
            "updated_config": {
                "passes": {
                    "OnnxModelOptimizer": {"params": {"graph_optimization_level": "all"}},
                    "QNNPreprocess": {"params": {}},
                    "QNNQuantization": {"params": {"per_channel": True}},
                }
            },
        },
        {
            "id": "coreml-dynamic-shape",
            "patterns": ["CoreML", "dynamic shape", "flexible shape", "input shape"],
            "title": "CoreML conversion fails with dynamic shapes",
            "root_cause": "CoreML prefers fixed input shapes; dynamic batch or sequence length can fail.",
            "solution": "Fix input shapes in OnnxConversion or set min/max sequence lengths; use symbolic shape inference carefully.",
            "updated_config": {
                "passes": {
                    "OnnxConversion": {
                        "params": {
                            "input_shapes": [[1, 128]],
                            "dynamic_axes": {},
                        }
                    }
                }
            },
        },
        {
            "id": "lora-merge-fail",
            "patterns": ["LoRA", "merge", "adapter", "base model", "merge_weights"],
            "title": "LoRA adapter cannot be merged into quantized base",
            "root_cause": "Merging adapters into a 4-bit or quantized base model is not supported by Olive.",
            "solution": "Merge LoRA weights into the full-precision base model before ONNX conversion; then quantize the merged ONNX model.",
            "updated_config": {
                "passes": {
                    "LoRA": {"params": {"target_modules": ["q_proj", "v_proj"]}},
                    "ExtractLoRA": {"params": {}},
                    "OnnxConversion": {"params": {}},
                }
            },
        },
        {
            "id": "openvino-fallback",
            "patterns": ["OpenVINO", "fallback", "CPUExecutionProvider", "not supported"],
            "title": "OpenVINO silently falls back to CPU EP",
            "root_cause": "OpenVINO Execution Provider does not support some ops or data types.",
            "solution": "Check OpenVINO operator support; use OnnxModelOptimizer to fuse unsupported patterns; convert to OpenVINO IR for broader op support.",
            "updated_config": {
                "passes": {
                    "OnnxModelOptimizer": {"params": {"graph_optimization_level": "all"}},
                    "OpenVINOConversion": {"params": {"fp16": True}},
                }
            },
        },
        {
            "id": "transformer-fusion-missing-dims",
            "patterns": ["OrtTransformersOptimization", "num_heads", "hidden_size", "fusion"],
            "title": "Transformer optimization fails due to missing dims",
            "root_cause": "OrtTransformersOptimization needs num_heads and hidden_size for attention fusion.",
            "solution": "Set num_heads and hidden_size in pass params; verify against model config.",
            "updated_config": {
                "passes": {
                    "OrtTransformersOptimization": {"params": {"num_heads": 32, "hidden_size": 4096, "opt_level": 1}}
                }
            },
        },
        {
            "id": "int4-perplexity",
            "patterns": ["int4", "perplexity", "quality", "degradation"],
            "title": "INT4 quantization causes large perplexity increase",
            "root_cause": "INT4 weight quantization is aggressive; small models or sensitive layers suffer.",
            "solution": "Use group size 128 or 64; exclude lm_head and embedding layers from quantization; try AWQ or GPTQ with larger group size.",
            "updated_config": {
                "passes": {
                    "NVModelOptQuantization": {
                        "params": {"algorithm": "awq", "weight_bits": 4, "calibration_iters": 64}
                    }
                }
            },
        },
        {
            "id": "onnx-fp16-nan",
            "patterns": ["FP16", "NaN", "inf", "OnnxFloatToFloat16", "overflow"],
            "title": "FP16 conversion produces NaN/Inf",
            "root_cause": "Model activations exceed FP16 range or are not trained for FP16.",
            "solution": "Keep I/O types float32; cast int inputs to int32; exclude sensitive ops from FP16; consider mixed precision.",
            "updated_config": {
                "passes": {"OnnxFloatToFloat16": {"params": {"keep_io_types": True, "cast_int_inputs_to_int32": True}}}
            },
        },
        {
            "id": "calibration-distribution-mismatch",
            "patterns": ["calibration", "distribution", "domain shift", "accuracy"],
            "title": "Calibration data distribution mismatch",
            "root_cause": "Calibration samples do not match inference inputs; quantization scale/zp are wrong.",
            "solution": "Use production-like data; cover min/max sequence lengths and vocabulary; avoid synthetic or unrelated samples.",
            "updated_config": {
                "data_configs": {
                    "calibration_data": {
                        "type": "HuggingFaceContainer",
                        "load_dataset_config": {"path": "<production-like-dataset>", "split": "validation"},
                    }
                }
            },
        },
        {
            "id": "multi-pass-cache-overwrite",
            "patterns": ["cache", "output", "overwrite", "same name"],
            "title": "Multi-pass run overwrites intermediate outputs",
            "root_cause": "Pass output names collide or cache_dir is shared across runs.",
            "solution": "Set unique output_name per pass; use separate cache_dir per experiment; clean cache between unrelated runs.",
            "updated_config": {
                "engine": {"cache_dir": "~/.cache/olive/experiment_1"},
                "passes": {
                    "OnnxConversion": {"output_name": "onnx_model"},
                    "OnnxQuantization": {"output_name": "quant_model"},
                },
            },
        },
        {
            "id": "search-local-optima",
            "patterns": ["search", "objective", "local optima", "latency vs accuracy"],
            "title": "Search finds poor local optimum",
            "root_cause": "Objective weights or search space are not tuned; evaluator is noisy with small sample size.",
            "solution": "Fix seed for reproducibility; use larger calibration/evaluation sets; adjust objective weights; try bayesian or evolutionary search.",
            "updated_config": {
                "engine": {
                    "search_algorithm": "bayesian",
                    "search_strategy": {
                        "max_iterations": 20,
                        "seed": 42,
                    },
                }
            },
        },
        {
            "id": "torchscript-export-fail",
            "patterns": ["torchscript", "jit", "tracer", " scripting"],
            "title": "TorchScript export fails for a HuggingFace model",
            "root_cause": "Many HF models use control flow or data-dependent shapes that TorchScript cannot trace.",
            "solution": "Use OnnxConversion instead of TorchScript export; provide example inputs and dynamic_axes.",
            "updated_config": {
                "passes": {
                    "OnnxConversion": {
                        "params": {
                            "example_input": "<tensor or dict>",
                            "dynamic_axes": {"input": {"0": "batch_size"}},
                        }
                    }
                }
            },
        },
    ]

    for e in new_entries:
        if e["id"] not in entries:
            data["entries"].append(e)

    save("troubleshooting.json", data)
    print(f"troubleshooting.json now has {len(data['entries'])} entries")


def _load_refresh_metadata(path: Path) -> dict[str, Any]:
    """Load existing refresh metadata or return an empty scaffold."""
    if not path.exists():
        return {"schema_version": 1, "runs": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"schema_version": 1, "runs": {}}
    if not isinstance(data, dict):
        return {"schema_version": 1, "runs": {}}
    runs = data.get("runs")
    if not isinstance(runs, dict):
        data["runs"] = {}
    data.setdefault("schema_version", 1)
    return data


def _source_timestamp_from_kb() -> str:
    """Derive a deterministic source stamp from existing KB last_updated fields.

    expand_kb has no external fetch; the catalog embedded in this script plus
    current KB dates are the source. Prefer max YYYY-MM-DD from KB roots;
    fall back to a content fingerprint of the three target files' identity keys.
    """
    dates: list[str] = []
    for name in ("passes.json", "hardware_profiles.json", "troubleshooting.json"):
        path = KB_DIR / name
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        last = data.get("last_updated")
        if isinstance(last, str) and len(last) >= 10:
            dates.append(last[:10])
    if dates:
        return max(dates)
    return f"content:{_content_fingerprint({'generator': GENERATOR_NAME})}"


def _write_refresh_metadata(
    *,
    generator_version: str,
    source_timestamp: str,
    source_fingerprint: str,
    changed_files: list[str],
) -> None:
    """Merge expand_kb run metadata into knowledge_base/refresh_metadata.json."""
    metadata_path = KB_DIR / REFRESH_METADATA_NAME
    existing = _load_refresh_metadata(metadata_path)

    run_meta: dict[str, Any] = {
        "generator": GENERATOR_NAME,
        "generator_version": generator_version,
        "source_timestamp": source_timestamp,
        "source_fingerprint": source_fingerprint,
        "changed_files": list(changed_files),
        "success": True,
    }

    runs = dict(existing.get("runs") or {})
    runs[GENERATOR_NAME] = run_meta

    all_changed: list[str] = []
    seen: set[str] = set()
    for name in sorted(runs):
        run = runs[name]
        if not isinstance(run, dict):
            continue
        for path in run.get("changed_files") or []:
            if isinstance(path, str) and path not in seen:
                seen.add(path)
                all_changed.append(path)

    stamps = [source_timestamp]
    for run in runs.values():
        if isinstance(run, dict) and isinstance(run.get("source_timestamp"), str):
            stamps.append(run["source_timestamp"])
    iso_stamps = [s for s in stamps if not str(s).startswith("content:")]
    aggregate_ts = max(iso_stamps) if iso_stamps else source_timestamp

    metadata: dict[str, Any] = {
        "schema_version": 1,
        "generator_version": generator_version,
        "source_timestamp": aggregate_ts,
        "changed_files": all_changed,
        "runs": runs,
    }

    # Sidecar only — never list refresh_metadata.json in changed_files (avoids
    # self-referential churn). Workflows read this file for the file list.
    new_text = _dump_json_text(metadata)
    if metadata_path.exists():
        try:
            if metadata_path.read_text(encoding="utf-8") == new_text:
                print(f"Refresh metadata unchanged at {metadata_path}")
                return
        except OSError:
            pass

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(new_text, encoding="utf-8")
    print(f"Refresh metadata written to {metadata_path}")


def main():
    """Expand the knowledge base with pass, hardware profile, and troubleshooting entries."""
    global _CHANGED_FILES
    _CHANGED_FILES = []

    generator_version = _generator_version()
    # Fingerprint of expansion outcome targets (post-run content identity).
    # Computed after expansions so it reflects final KB state.
    expand_passes()
    expand_hardware_profiles()
    expand_troubleshooting()

    outcome: dict[str, Any] = {}
    for name in ("passes.json", "hardware_profiles.json", "troubleshooting.json"):
        path = KB_DIR / name
        if path.exists():
            try:
                outcome[name] = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                outcome[name] = None

    source_fingerprint = _content_fingerprint(
        {
            "generator": GENERATOR_NAME,
            "generator_version": generator_version,
            "outcome": outcome,
        }
    )
    source_timestamp = _source_timestamp_from_kb()

    _write_refresh_metadata(
        generator_version=generator_version,
        source_timestamp=source_timestamp,
        source_fingerprint=source_fingerprint,
        changed_files=list(_CHANGED_FILES),
    )
    print(
        f"expand_kb metadata: generator_version={generator_version} "
        f"source_timestamp={source_timestamp} changed_files={list(_CHANGED_FILES)}"
    )


if __name__ == "__main__":
    main()
