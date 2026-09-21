"""Domain-aware troubleshoot_olive_error coverage."""

from olive_mcp_server.mcp_server import _TOOL_IMPORTS, call_tool
from olive_mcp_server.tools.troubleshooting import (
    diagnose_error,
    reset_frequency_store,
    troubleshoot_olive_error,
)


def setup_function():
    reset_frequency_store()


def test_hf_config_matches_studio_domain():
    # Keyword mode: this suite exercises keyword-based domain matching, not
    # semantic retrieval, and must not risk a real embedding-model load in
    # the shared single-flight worker outliving the test.
    resp = troubleshoot_olive_error(
        "TypeError: PyTorchModelHandler.__init__() got an unexpected keyword argument 'hf_config'",
        mode="keyword",
    )
    assert resp["matched_entry"] == "studio-pytorch-hf-config"
    assert resp["domain"] == "studio"
    assert resp["applyable"] is False
    assert resp["updated_config"] == {}
    assert "HfModel" in resp["workaround"] or "HfModel" in resp["root_cause"]


def test_oom_still_matches_olive_first():
    resp = troubleshoot_olive_error("CUDA out of memory during OnnxQuantization")
    assert resp["matched_entry"] == "oom-quantization"
    assert resp["domain"] == "olive"
    assert resp["applyable"] is True
    assert resp["updated_config"]


def test_domain_studio_only_skips_olive_pool():
    resp = troubleshoot_olive_error(
        "CUDA out of memory during OnnxQuantization",
        domain="studio",
    )
    # OOM patterns are olive-only; studio domain should not fake an olive match
    assert resp["matched_entry"] is None
    assert resp["domain"] is None
    assert resp["applyable"] is False


def test_unmatched_has_no_fake_olive_domain():
    resp = troubleshoot_olive_error("completely-unknown-xyzzy-error-token-999")
    assert resp["matched_entry"] is None
    assert resp["domain"] is None
    assert resp["applyable"] is False
    assert resp["updated_config"] == {}
    assert isinstance(resp["relevant_quirks"], list)
    assert len(resp["relevant_quirks"]) > 0


def test_studio_applyable_cache_dir():
    resp = troubleshoot_olive_error(
        "Need an isolated Olive Studio cache directory for this experiment cache",
        domain="studio",
    )
    assert resp["matched_entry"] == "studio-unique-cache-dir"
    assert resp["domain"] == "studio"
    assert resp["applyable"] is True
    assert "engine" in resp["updated_config"]


def test_diagnose_error_alias():
    a = troubleshoot_olive_error("Protobuf size limit exceeded external data")
    b = diagnose_error("Protobuf size limit exceeded external data")
    assert a["matched_entry"] == b["matched_entry"] == "onnx-export-external-data"


def test_call_tool_diagnose_error_registered():
    assert "diagnose_error" in _TOOL_IMPORTS
    result = call_tool(
        "diagnose_error",
        {"error_message": "unexpected keyword argument 'hf_config'", "domain": "auto"},
    )
    assert result["domain"] == "studio"
    assert result["matched_entry"] == "studio-pytorch-hf-config"


def test_webgpu_entry_requires_webgpu_signal():
    """Generic isRunnable / localExecutionIssues must not alone match WebGPU."""
    generic = troubleshoot_olive_error(
        "Recipe validation failed: isRunnable false due to localExecutionIssues",
        domain="studio",
    )
    assert generic.get("matched_entry") != "studio-webgpu-local-blocked"

    specific = troubleshoot_olive_error(
        "WebGpuExecutionProvider selected; local Execute Live blocked for browser WebGPU",
        domain="studio",
    )
    assert specific["matched_entry"] == "studio-webgpu-local-blocked"


def test_openvino_npu_entry_requires_npu_signal():
    """Bare OpenVINOExecutionProvider / openvinoTargetDevice must not alone match NPU."""
    generic = troubleshoot_olive_error(
        "OpenVINOExecutionProvider selected with openvinoTargetDevice CPU",
        domain="studio",
    )
    assert generic.get("matched_entry") != "studio-openvino-npu-device"

    specific = troubleshoot_olive_error(
        "OpenVINO NPU path: set openvinoTargetDevice: NPU on Core Ultra NPU",
        domain="studio",
    )
    assert specific["matched_entry"] == "studio-openvino-npu-device"
