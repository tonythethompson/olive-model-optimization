"""Integration tests: troubleshoot_olive_error coverage.

Closes the test coverage gap by exercising multiple knowledge base entries,
verifying matched_entry is set correctly, and testing edge cases around
scoring, context, and response shape.
"""

import asyncio
import json

from olive_mcp_server.mcp_server import mcp
from olive_mcp_server.tools.troubleshooting import troubleshoot_olive_error


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Helper: invoke the tool directly (unit-level) and via MCP server (integration)
# ---------------------------------------------------------------------------


def _call_direct(**kwargs):
    """Call the troubleshooting function directly and return the dict."""
    return troubleshoot_olive_error(**kwargs)


def _call_via_server(**kwargs):
    """Call through the MCP server object and return the parsed JSON dict."""
    result, _ = _run(mcp.call_tool("troubleshoot_olive_error", kwargs))
    return json.loads(result[0].text)


# ---------------------------------------------------------------------------
# Response shape — every response must have these fields regardless of match
# ---------------------------------------------------------------------------


def test_response_shape_always_has_required_fields_direct():
    resp = _call_direct(error_message="anything")
    for key in ("matched_entry", "title", "root_cause", "workaround", "updated_config", "relevant_quirks"):
        assert key in resp, f"Missing required field: {key}"


def test_response_shape_always_has_required_fields_via_server():
    resp = _call_via_server(error_message="anything")
    for key in ("matched_entry", "title", "root_cause", "workaround", "updated_config", "relevant_quirks"):
        assert key in resp, f"Missing required field: {key}"


# ---------------------------------------------------------------------------
# Unmatched errors — matched_entry must be None, fallback title shown
# ---------------------------------------------------------------------------


def test_unmatched_error_returns_none_matched_entry_direct():
    resp = _call_direct(error_message="Some random unique unknown failure 12345xyz")
    assert resp["matched_entry"] is None
    assert resp["title"] == "No exact match found"
    assert isinstance(resp["updated_config"], dict)


def test_unmatched_error_returns_none_matched_entry_via_server():
    resp = _call_via_server(error_message="Some random unique unknown failure 12345xyz")
    assert resp["matched_entry"] is None
    assert resp["title"] == "No exact match found"


def test_empty_error_message_returns_no_match():
    resp = _call_direct(error_message="")
    assert resp["matched_entry"] is None


# ---------------------------------------------------------------------------
# Matched entries — verify each returns the correct entry ID and useful data
# ---------------------------------------------------------------------------


def test_match_onnx_export_external_data():
    """onnx-export-external-data: patterns include '2GB', 'external data', 'too large'."""
    resp = _call_direct(
        error_message="ValueError: The model file size is larger than 2GB. Please use use_external_data_format=True",
        pass_name="OnnxConversion",
    )
    assert resp["matched_entry"] == "onnx-export-external-data"
    assert "external" in resp["workaround"].lower()
    assert resp["updated_config"] != {}


def test_match_onnx_export_shape():
    """onnx-export-shape: patterns include 'dynamic shape', 'shape mismatch'."""
    resp = _call_direct(
        error_message="RuntimeError: Expected all tensors to be on the same device. Shape mismatch in forward pass.",
        pass_name="OnnxConversion",
    )
    assert resp["matched_entry"] == "onnx-export-shape"
    assert "dynamic" in resp["workaround"].lower() or "dynamic_axes" in json.dumps(resp["updated_config"])


def test_match_quant_accuracy_collapse():
    """quant-accuracy-collapse: patterns include 'accuracy drop', 'perplexity', 'collapsed'."""
    resp = _call_direct(
        error_message="accuracy dropped from 92% to 45% after INT8 quantization",
        pass_name="OnnxQuantization",
    )
    assert resp["matched_entry"] == "quant-accuracy-collapse"
    assert "per_channel" in json.dumps(resp["updated_config"]) or "calibration" in resp["workaround"].lower()


def test_match_ep_fallback_cpu():
    """ep-fallback-cpu: patterns include 'fallback', 'CPUExecutionProvider', 'not supported'."""
    resp = _call_direct(
        error_message=(
            "WARNING: op NotSupported is not supported by CUDAExecutionProvider, fallback to CPUExecutionProvider"
        ),
        pass_name="",
    )
    assert resp["matched_entry"] == "ep-fallback-cpu"


def test_match_oom_quantization():
    """oom-quantization: patterns include 'CUDA out of memory', 'OOM', 'out of memory'."""
    resp = _call_direct(
        error_message="RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB (GPU 0; 24.00 GiB total)",
        pass_name="OnnxQuantization",
    )
    assert resp["matched_entry"] == "oom-quantization"


def test_match_calibration_data_mismatch():
    """calibration-data-mismatch: patterns include 'calibration', 'dataloader', 'data_config'."""
    resp = _call_direct(
        error_message="Calibration dataloader returned empty batch — no data found in dataset",
        pass_name="OnnxStaticQuantization",
    )
    assert resp["matched_entry"] == "calibration-data-mismatch"


def test_match_lora_target_modules():
    """lora-target-modules: patterns include 'LoRA', 'target_modules', 'adapter'."""
    resp = _call_direct(
        error_message="LoRA target_modules ['q_proj'] not found in model keys",
        pass_name="LoRA",
    )
    assert resp["matched_entry"] == "lora-target-modules"


def test_match_tensorrt_build_slow():
    """tensorrt-build-slow: patterns include 'TensorRT', 'engine build', 'build time'."""
    resp = _call_direct(
        error_message="TensorRT engine build time exceeded 3600s, consider reducing search space",
        pass_name="",
    )
    assert resp["matched_entry"] == "tensorrt-build-slow"


def test_match_awq_slow_calibration():
    """awq-slow-calibration: patterns include 'AWQ', 'nvidia-modelopt', 'awq calibration'."""
    resp = _call_direct(
        error_message="AWQ calibration is taking too long with nvidia-modelopt, stuck at 50%",
        pass_name="NVModelOptQuantization",
    )
    assert resp["matched_entry"] == "awq-slow-calibration"


def test_match_qnn_layer_not_supported():
    """qnn-layer-not-supported: patterns include 'QNN', 'not supported', 'LayerNormalization'."""
    resp = _call_direct(
        error_message="QNN reports unsupported op LayerNormalization on Qualcomm NPU",
        pass_name="QNNQuantization",
    )
    assert resp["matched_entry"] == "qnn-layer-not-supported"


def test_match_int4_perplexity():
    """int4-perplexity: patterns include 'int4', 'perplexity', 'quality', 'degradation'."""
    resp = _call_direct(
        error_message="INT4 quantization caused large perplexity increase — quality degradation observed",
        pass_name="NVModelOptQuantization",
    )
    assert resp["matched_entry"] == "int4-perplexity"


def test_match_onnx_fp16_nan():
    """onnx-fp16-nan: patterns include 'FP16', 'NaN', 'inf', 'overflow'."""
    resp = _call_direct(
        error_message="OnnxFloatToFloat16 produced NaN values in output — FP16 overflow detected",
        pass_name="OnnxFloatToFloat16",
    )
    assert resp["matched_entry"] == "onnx-fp16-nan"


def test_match_multi_pass_cache_overwrite():
    """multi-pass-cache-overwrite: patterns include 'cache', 'overwrite', 'same name'."""
    resp = _call_direct(
        error_message="Pass output overwrite detected: cache file for same name output already exists",
        pass_name="",
    )
    assert resp["matched_entry"] == "multi-pass-cache-overwrite"


def test_match_transformer_fusion_missing_dims():
    """transformer-fusion-missing-dims: patterns include 'OrtTransformersOptimization', 'num_heads'."""
    resp = _call_direct(
        error_message="OrtTransformersOptimization requires num_heads and hidden_size for attention fusion",
        pass_name="OrtTransformersOptimization",
    )
    assert resp["matched_entry"] == "transformer-fusion-missing-dims"


def test_match_torchscript_export_fail():
    """torchscript-export-fail: patterns include 'torchscript', 'jit', 'tracer'."""
    resp = _call_direct(
        error_message="torch.jit.trace failed for HuggingFace model — TorchScript export error",
        pass_name="",
    )
    assert resp["matched_entry"] == "torchscript-export-fail"


def test_match_openvino_fallback():
    """openvino-fallback: patterns include 'OpenVINO', 'fallback', 'CPUExecutionProvider'."""
    resp = _call_direct(
        error_message="OpenVINO silently falls back to CPUExecutionProvider — op not supported",
        pass_name="",
    )
    assert resp["matched_entry"] == "openvino-fallback"
    assert "OpenVINO" in resp["workaround"] or "openvino" in json.dumps(resp["updated_config"]).lower()


def test_match_calibration_distribution_mismatch():
    """calibration-distribution-mismatch: patterns include 'calibration', 'distribution', 'domain shift'."""
    resp = _call_direct(
        error_message="Calibration data distribution does not match inference inputs — domain shift detected",
        pass_name="OnnxStaticQuantization",
    )
    assert resp["matched_entry"] == "calibration-distribution-mismatch"


def test_match_search_local_optima():
    """search-local-optima: patterns include 'search', 'local optima', 'latency vs accuracy'."""
    resp = _call_direct(
        error_message="Search found a local optima — latency vs accuracy tradeoff not converged",
        pass_name="",
    )
    assert resp["matched_entry"] == "search-local-optima"


def test_match_coreml_dynamic_shape():
    """coreml-dynamic-shape: patterns include 'CoreML', 'dynamic shape', 'input shape'."""
    resp = _call_direct(
        error_message="CoreML conversion fails with dynamic shape on input — flexible shape not supported",
        pass_name="",
    )
    assert resp["matched_entry"] == "coreml-dynamic-shape"


def test_match_lora_merge_fail():
    """lora-merge-fail: patterns include 'LoRA', 'merge', 'adapter', 'base model', 'merge_weights'."""
    resp = _call_direct(
        error_message="LoRA adapter cannot be merged into quantized base model — merge_weights not supported",
        pass_name="LoRA",
    )
    assert resp["matched_entry"] == "lora-merge-fail"
    assert "merge" in resp["workaround"].lower() or "merge" in json.dumps(resp["updated_config"]).lower()


# ---------------------------------------------------------------------------
# Scoring: pass_name context should boost the correct match
# ---------------------------------------------------------------------------


def test_pass_name_context_improves_scoring():
    """Providing a relevant pass_name should not break the correct match."""
    resp_with_pass = _call_direct(
        error_message="CUDA out of memory",
        pass_name="OnnxQuantization",
    )
    resp_without_pass = _call_direct(
        error_message="CUDA out of memory",
        pass_name="",
    )
    # Both should match the same entry
    assert resp_with_pass["matched_entry"] == "oom-quantization"
    assert resp_without_pass["matched_entry"] == "oom-quantization"


def test_config_context_contributes_to_scoring():
    """config_context patterns should contribute to the match score."""
    resp = _call_direct(
        error_message="Calibration failed",
        pass_name="OnnxStaticQuantization",
        config_context="data_config missing from recipe",
    )
    assert resp["matched_entry"] == "calibration-data-mismatch"


# ---------------------------------------------------------------------------
# matched_entry inversion guard — the field must NEVER be falsy when matched
# ---------------------------------------------------------------------------


def test_matched_entry_is_string_id_when_matched():
    """When a match is found, matched_entry must be a non-empty string (the entry ID)."""
    resp = _call_direct(
        error_message="Protobuf size exceeded 2GB limit during ONNX export",
        pass_name="OnnxConversion",
    )
    assert resp["matched_entry"] is not None
    assert isinstance(resp["matched_entry"], str)
    assert len(resp["matched_entry"]) > 0


def test_matched_entry_is_none_when_no_match():
    """When no match is found, matched_entry must be exactly None (not empty string, not 0)."""
    resp = _call_direct(
        error_message="Completely unrelated error message with no matching patterns at all 99999",
    )
    assert resp["matched_entry"] is None


# ---------------------------------------------------------------------------
# MCP server integration — exercise the tool through call_tool
# ---------------------------------------------------------------------------


def test_troubleshoot_multiple_entries_via_server():
    """Verify several distinct entries resolve correctly through the MCP server."""
    cases = [
        (
            "CUDA out of memory during quantization calibration",
            "oom-quantization",
        ),
        (
            "TensorRT engine build is very slow or fails",
            "tensorrt-build-slow",
        ),
        (
            "LoRA target_modules not found in model",
            "lora-target-modules",
        ),
        (
            "CoreML conversion fails with dynamic shapes on input",
            "coreml-dynamic-shape",
        ),
    ]
    for error_msg, expected_id in cases:
        resp = _call_via_server(error_message=error_msg)
        assert resp["matched_entry"] == expected_id, (
            f"Expected {expected_id} for '{error_msg}', got {resp['matched_entry']}"
        )


def test_troubleshoot_updated_config_not_empty_on_match():
    """When a match is found, updated_config should contain actionable guidance."""
    resp = _call_via_server(
        error_message="ONNX model exceeds 2GB protobuf limit, too large for standard export",
        pass_name="OnnxConversion",
    )
    assert resp["matched_entry"] is not None
    assert isinstance(resp["updated_config"], dict)
    assert len(resp["updated_config"]) > 0, "updated_config should not be empty for a matched entry"


def test_troubleshoot_relevant_quirks_always_populated():
    """relevant_quirks should be a non-empty list even when no match is found."""
    resp = _call_via_server(error_message="totally unknown error xyz 999")
    assert isinstance(resp["relevant_quirks"], list)
    assert len(resp["relevant_quirks"]) > 0


def test_troubleshoot_relevant_quirks_include_full_category_not_truncated():
    """All quirks from matched categories surface (not first-2 / max-6 truncation)."""
    from olive_mcp_server.tools import load_quirks
    from olive_mcp_server.tools.troubleshooting import reset_frequency_store, troubleshoot_olive_error

    reset_frequency_store()
    resp = troubleshoot_olive_error(
        error_message="cache output overwrite same name multi-pass",
        pass_name="OnnxQuantization",
    )
    quirks = resp["relevant_quirks"]
    assert isinstance(quirks, list)

    # multi-pass-cache-overwrite → pass_ordering + quantization + onnx_export (all entries)
    kb = load_quirks()
    expected = []
    for cat in ("pass_ordering", "quantization", "onnx_export"):
        for q in kb.get(cat, []):
            expected.append(q["title"])
    for title in expected:
        assert title in quirks, f"missing quirk title: {title}"

    # Previously truncated away — must appear
    assert "Graph Optimize Before Quantization" in quirks
    assert "External Data Format" in quirks
    assert "Symmetric vs Asymmetric Quantization" in quirks
    assert "Float16 After Quantization" in quirks


def test_troubleshoot_lora_quirks_include_qlora():
    """LoRA merge failures should surface every lora category quirk including QLoRA."""
    from olive_mcp_server.tools.troubleshooting import reset_frequency_store, troubleshoot_olive_error

    reset_frequency_store()
    resp = troubleshoot_olive_error(
        error_message="LoRA adapter cannot be merged into quantized base",
        pass_name="LoRA",
    )
    quirks = resp["relevant_quirks"]
    assert "QLoRA + Quantization" in quirks
    assert "Base Model Frozen" in quirks
    assert "Rank Selection" in quirks
