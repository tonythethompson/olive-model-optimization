"""Unit tests for Olive MCP tools."""

import pytest

from olive_mcp_server.tools.cli_helper import get_cli_command
from olive_mcp_server.tools.compatibility import get_model_compatibility
from olive_mcp_server.tools.config_generator import get_pass_config_template
from olive_mcp_server.tools.data_config import get_data_config_template
from olive_mcp_server.tools.docs_search import search_olive_documentation
from olive_mcp_server.tools.hardware_guide import get_hardware_optimization_guide
from olive_mcp_server.tools.integration_recipes import get_integration_recipe
from olive_mcp_server.tools.pass_catalog import get_olive_passes
from olive_mcp_server.tools.pass_chain import get_pass_chain
from olive_mcp_server.tools.pass_parameters import get_pass_parameters
from olive_mcp_server.tools.strategy_advisor import get_quantization_strategy
from olive_mcp_server.tools.tradeoff import evaluate_optimization_tradeoff
from olive_mcp_server.tools.troubleshooting import troubleshoot_olive_error


def test_get_olive_passes_no_filter():
    result = get_olive_passes()
    assert "passes" in result
    assert result["count"] > 0
    assert all("name" in p for p in result["passes"])


def test_get_olive_passes_filter_quantization():
    result = get_olive_passes(filter="quantization")
    assert result["filter"] == "quantization"
    assert all(p["type"] == "quantization" for p in result["passes"])


def test_get_pass_config_template_known_pass():
    result = get_pass_config_template(
        pass_name="OnnxQuantization",
        framework="onnx",
        optimization_target="latency",
    )
    assert "error" not in result
    assert result["pass_name"] == "OnnxQuantization"
    assert "config" in result
    assert result["config"]["passes"]["OnnxQuantization"]["params"]["quant_format"] in (
        "QOperator",
        "QDQ",
    )


def test_get_pass_config_template_unknown_pass():
    result = get_pass_config_template(pass_name="NotARealPass")
    assert "error" in result


def test_get_quantization_strategy_llm_nvidia():
    result = get_quantization_strategy(
        model_type="LLM",
        target_hardware="NVIDIA RTX 4090",
        latency_budget="<100ms",
        accuracy_threshold="<2% drop",
    )
    assert "error" not in result
    assert "NVModelOptQuantization" in result["recommended_algorithm"] or "AWQ" in result["recommended_algorithm"]
    assert result["target_hardware"] == "nvidia"
    assert result["model_type"] == "llm"
    assert "pass_chain" in result


def test_get_quantization_strategy_cnn_mobile():
    result = get_quantization_strategy(
        model_type="CNN",
        target_hardware="Qualcomm Snapdragon NPU",
        latency_budget="<50ms",
        accuracy_threshold="<1% drop",
    )
    assert result["model_type"] == "cnn"
    assert result["target_hardware"] == "qualcomm"
    assert "QNN" in result["recommended_algorithm"]


def test_get_hardware_optimization_guide_known():
    result = get_hardware_optimization_guide(
        target_hardware="NVIDIA RTX 4090",
        model_size="large",
        latency_goal="<100ms",
    )
    assert "error" not in result
    assert result["target_hardware"] == "NVIDIA RTX 4090"
    assert "TensorrtExecutionProvider" in result["execution_providers"]
    assert result["calibration_size"] >= 300


def test_get_hardware_optimization_guide_unknown():
    result = get_hardware_optimization_guide(target_hardware="MadeUpChip 9000")
    assert "error" in result


def test_get_pass_chain_valid():
    result = get_pass_chain(["OnnxConversion", "OnnxModelOptimizer", "OnnxQuantization"])
    assert result["valid"] is True
    assert len(result["errors"]) == 0
    assert result["chain"][0]["name"] == "OnnxConversion"


def test_get_pass_chain_invalid_order():
    result = get_pass_chain(["OnnxQuantization", "OnnxConversion"], source_format="PyTorch")
    assert result["valid"] is False
    assert len(result["errors"]) > 0


def test_troubleshoot_known_error():
    result = troubleshoot_olive_error(
        error_message="The ONNX model is larger than 2GB",
        pass_name="OnnxConversion",
    )
    assert "root_cause" in result
    assert "external" in result["workaround"].lower()


def test_get_model_compatibility_known():
    result = get_model_compatibility(
        model_name="Mistral 7B",
        framework="PyTorch",
    )
    assert "hardware_profiles" in result


def test_get_model_compatibility_filtered_by_hardware():
    result = get_model_compatibility(
        model_name="Mistral 7B",
        framework="PyTorch",
        hardware_target="NVIDIA RTX 4090",
    )
    assert result["selected_hardware"] == "NVIDIA RTX 4090"
    assert "hardware_compatibility" in result
    assert isinstance(result["compatibility_warnings"], list)


def test_get_cli_command():
    result = get_cli_command(
        optimization_goal="quantize",
        model="microsoft/phi-4",
        target="gpu",
    )
    assert "olive quantize" in result["command"]
    assert "--config" in result["command"]


def test_get_data_config_template():
    result = get_data_config_template(data_source="huggingface", task="calibration")
    assert len(result["data_configs"]) > 0
    assert result["data_configs"][0]["type"] == "HuggingFaceContainer"
    assert result["data_configs"][0]["name"] == "calibration_data"
    assert "sampling" in result["data_configs"][0]


def test_search_olive_documentation():
    # Keep the test local-only so CI does not depend on live network.
    result = search_olive_documentation(query="calibration data", top_k=3, live=False)
    assert result["count"] > 0
    assert len(result["results"]) <= 3


def test_search_olive_documentation_with_live_source(monkeypatch: pytest.MonkeyPatch):
    """Live search should merge cached fetched docs with local results.

    Ranking is deterministic here (local/live search functions are mocked
    directly) rather than relying on real embedding scores, since a live
    result is only surfaced when it is genuinely competitive with local KB
    hits — the old "always force one live result in" behavior was a bug.
    """
    from olive_mcp_server.tools import docs_search

    local_results = [{"source": "passes.foo", "snippet": "local hit", "relevance": 0.5}]
    live_results = [{"source": "live:index", "snippet": "live hit", "relevance": 0.9}]

    def fake_local(query, top_k, mode=None, budget_ms=None):
        """Return mocked local documentation search results and search metadata."""
        return local_results[:top_k], {
            "mode": "auto",
            "effective": "hybrid",
            "degraded": False,
        }

    monkeypatch.setattr(docs_search, "_search_local", fake_local)
    monkeypatch.setattr(
        docs_search,
        "_search_live",
        lambda query, top_k, mode=None, budget_ms=None: (
            live_results[:top_k],
            {"mode": "auto", "effective": "hybrid", "degraded": False},
        ),
    )

    result = search_olive_documentation(query="calibration data", top_k=3, live=True)
    assert result["count"] > 0
    assert any("live:" in r["source"] for r in result["results"])


def test_get_pass_parameters():
    result = get_pass_parameters(pass_name="OnnxQuantization", parameter_name="quant_format")
    assert result["parameter_name"] == "quant_format"
    assert "QOperator" in result["documentation"]["enum"]


def test_evaluate_optimization_tradeoff():
    result = evaluate_optimization_tradeoff(
        passes=["OnnxConversion", "OnnxQuantization"],
        evaluation_metrics=["accuracy", "latency", "size"],
    )
    assert result["predicted_outcomes"]["size"] < 100
    assert result["predicted_outcomes"]["latency"] < 1.0


def test_get_pass_chain_onnx_source_no_conversion_required():
    result = get_pass_chain(["OnnxQuantization"], source_format="onnx")
    assert result["valid"] is True
    assert len(result["errors"]) == 0


def test_get_pass_chain_missing_source_format_warns():
    # Absent source_format must still be reported when no compatible
    # conversion pass precedes the quantization pass.
    result = get_pass_chain(["OnnxQuantization"])
    assert result["valid"] is True
    assert len(result["errors"]) == 0
    assert any("no source_format was provided" in w for w in result["warnings"])


def test_get_integration_recipe_list():
    result = get_integration_recipe()
    assert "recipes" in result
    assert result["count"] > 0
    assert all("id" in r for r in result["recipes"])


def test_get_integration_recipe_filter_by_model_type():
    result = get_integration_recipe(model_type="LLM")
    assert result["count"] > 0
    assert all("LLM" in r.get("model_type", []) for r in result["recipes"])


def test_get_integration_recipe_filter_by_target_hardware():
    result = get_integration_recipe(target_hardware="CPU")
    assert result["count"] > 0
    assert any("CPU" in hardware for r in result["recipes"] for hardware in r["target_hardware"])


def test_get_integration_recipe_detail():
    result = get_integration_recipe(recipe_id="resnet50_cpu_ptq")
    assert "error" not in result
    assert result["recipe_id"] == "resnet50_cpu_ptq"
    assert "recipe" in result
    assert "passes" in result["recipe"]


def test_get_integration_recipe_not_found():
    result = get_integration_recipe(recipe_id="not-a-real-recipe")
    assert "error" in result


def test_get_model_compatibility_new_model():
    result = get_model_compatibility(model_name="google/vit-base-patch16-224", framework="PyTorch")
    assert result["model"] == "ViT-base"
    assert result["framework_supported"] is True
    assert "Intel Core i9 CPU" in result["hardware_profiles"]


def test_get_model_compatibility_azure_hardware():
    result = get_model_compatibility(
        model_name="Azure OpenAI Whisper",
        framework="ONNX",
        hardware_target="Azure ML N-series",
    )
    assert result["selected_hardware"] == "Azure ML N-series"
    assert "AzureMLQuantization" in result["hardware_compatibility"]


def test_get_integration_recipe_qnn():
    result = get_integration_recipe(recipe_id="qualcomm_qnn_mobile")
    assert "error" not in result
    pass_types = {p["type"] for p in result["recipe"]["passes"].values()}
    assert "QNNConversion" in pass_types


def test_get_integration_recipe_azure():
    result = get_integration_recipe(recipe_id="azure_ml_quant")
    assert "error" not in result
    pass_types = {p["type"] for p in result["recipe"]["passes"].values()}
    assert "AzureMLQuantization" in pass_types


def test_get_integration_recipe_openvino_vision():
    result = get_integration_recipe(recipe_id="openvino_vision")
    assert "error" not in result
    pass_types = {p["type"] for p in result["recipe"]["passes"].values()}
    assert "OpenVINOQuantization" in pass_types


def _recipe_execution_providers(result: dict) -> list[str]:
    accelerators = result["recipe"]["systems"]["local_system"]["config"]["accelerators"]
    return accelerators[0]["execution_providers"]


def test_get_integration_recipe_phi3_trt_rtx():
    result = get_integration_recipe(recipe_id="phi3_trt_rtx_awq")
    assert "error" not in result
    assert result["recipe_id"] == "phi3_trt_rtx_awq"
    assert "NvTensorRTRTXExecutionProvider" in _recipe_execution_providers(result)


def test_get_integration_recipe_resnet50_directml():
    result = get_integration_recipe(recipe_id="resnet50_directml")
    assert "error" not in result
    assert result["recipe_id"] == "resnet50_directml"
    assert "DmlExecutionProvider" in _recipe_execution_providers(result)


def test_get_integration_recipe_mobilenet_openvino_npu():
    result = get_integration_recipe(recipe_id="mobilenet_openvino_npu")
    assert "error" not in result
    assert result["recipe_id"] == "mobilenet_openvino_npu"
    assert "OpenVINOExecutionProvider" in _recipe_execution_providers(result)


def test_get_integration_recipe_bert_webgpu_fp16():
    result = get_integration_recipe(recipe_id="bert_webgpu_fp16")
    assert "error" not in result
    assert result["recipe_id"] == "bert_webgpu_fp16"
    assert "WebGpuExecutionProvider" in _recipe_execution_providers(result)


def test_get_integration_recipe_llama3_rocm():
    result = get_integration_recipe(recipe_id="llama3_rocm_gptq")
    assert "error" not in result
    assert result["recipe_id"] == "llama3_rocm_gptq"
    assert "ROCMExecutionProvider" in _recipe_execution_providers(result)
    pass_types = {p["type"] for p in result["recipe"]["passes"].values()}
    assert "GptqQuantizer" in pass_types
