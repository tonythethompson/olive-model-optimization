"""Property-based tests for get_model_info tool.

Feature: v0.3-agent-mcp-tools
Properties 7 & 8: VRAM Estimate Arithmetic, Recommended Quantization Threshold
Validates: Requirements 9.3, 9.5
"""

from __future__ import annotations

import json
from unittest.mock import patch

from hypothesis import given, settings
from hypothesis import strategies as st

from olive_mcp_server.tools.agent_model_info import get_model_info


# Feature: v0.3-agent-mcp-tools, Property 7: VRAM Estimate Arithmetic
class TestVRAMEstimateArithmetic:
    """For any positive float params_b, estimated_vram_gb == params_b * 2.0.

    The property asserts the invariant relationship between the reported
    params_b and estimated_vram_gb in the output: vram is always exactly
    double the reported param count.
    """

    @settings(max_examples=100)
    @given(params_b=st.floats(min_value=0.01, max_value=1000.0))
    def test_vram_equals_params_times_two(self, params_b: float) -> None:
        """estimated_vram_gb must always equal the tool's reported params_b * 2.0.

        We inject params via safetensors.total and verify the output relationship:
        estimated_vram_gb == result["params_b"] * 2.0 (exact — same float path).
        """
        # Mock HF API to return safetensors.total = params_b * 1e9
        hf_response = {
            "safetensors": {"total": params_b * 1e9},
            "config": {"architectures": ["TestArchitecture"]},
        }

        with patch(
            "olive_mcp_server.tools.agent_model_info._fetch_hf_metadata",
            return_value=hf_response,
        ):
            result = get_model_info("test-org/test-model")

        # Should not be an error
        assert "error" not in result, f"Unexpected error: {result}"
        # The property: vram is exactly double the reported params_b
        assert result["estimated_vram_gb"] == result["params_b"] * 2.0

    @settings(max_examples=100)
    @given(params_b=st.floats(min_value=0.01, max_value=1000.0))
    def test_vram_json_round_trip(self, params_b: float) -> None:
        """Output must survive JSON serialization round-trip."""
        hf_response = {
            "safetensors": {"total": params_b * 1e9},
            "config": {"architectures": ["RoundTripArch"]},
        }

        with patch(
            "olive_mcp_server.tools.agent_model_info._fetch_hf_metadata",
            return_value=hf_response,
        ):
            result = get_model_info("test-org/test-model")

        assert "error" not in result
        assert json.loads(json.dumps(result)) == result


# Feature: v0.3-agent-mcp-tools, Property 8: Recommended Quantization Threshold
class TestRecommendedQuantizationThreshold:
    """For params_b >= 6.0 -> int4; for params_b < 6.0 -> int8."""

    @settings(max_examples=100)
    @given(params_b=st.floats(min_value=6.0, max_value=1000.0))
    def test_large_models_get_int4(self, params_b: float) -> None:
        """Models with params_b >= 6.0 must recommend int4 quantization."""
        hf_response = {
            "safetensors": {"total": params_b * 1e9},
            "config": {"architectures": ["LargeModelArch"]},
        }

        with patch(
            "olive_mcp_server.tools.agent_model_info._fetch_hf_metadata",
            return_value=hf_response,
        ):
            result = get_model_info("test-org/large-model")

        assert "error" not in result
        # Derive expected from the returned params_b (the multiply-then-divide
        # injection may round across the 6.0 threshold).
        assert result["params_b"] >= 6.0
        assert result["recommended_quant"] == "int4"

    @settings(max_examples=100)
    @given(params_b=st.floats(min_value=0.001, max_value=5.999999999))
    def test_small_models_get_int8(self, params_b: float) -> None:
        """Models with params_b < 6.0 must recommend int8 quantization."""
        hf_response = {
            "safetensors": {"total": params_b * 1e9},
            "config": {"architectures": ["SmallModelArch"]},
        }

        with patch(
            "olive_mcp_server.tools.agent_model_info._fetch_hf_metadata",
            return_value=hf_response,
        ):
            result = get_model_info("test-org/small-model")

        assert "error" not in result
        # Derive expected from the returned params_b (the multiply-then-divide
        # injection may round across the 6.0 threshold).
        assert result["params_b"] < 6.0
        assert result["recommended_quant"] == "int8"

    @settings(max_examples=100)
    @given(params_b=st.floats(min_value=0.001, max_value=1000.0))
    def test_threshold_boundary_is_deterministic(self, params_b: float) -> None:
        """The int4/int8 boundary at 6.0 is consistent and deterministic."""
        hf_response = {
            "safetensors": {"total": params_b * 1e9},
            "config": {"architectures": ["BoundaryArch"]},
        }

        with patch(
            "olive_mcp_server.tools.agent_model_info._fetch_hf_metadata",
            return_value=hf_response,
        ):
            result = get_model_info("test-org/boundary-model")

        assert "error" not in result
        # Derive the expected quantization from the returned params_b, since the
        # injected multiply-then-divide value may round across the 6.0 threshold.
        expected = "int4" if result["params_b"] >= 6.0 else "int8"
        assert result["recommended_quant"] == expected

    @settings(max_examples=100)
    @given(params_b=st.floats(min_value=0.001, max_value=1000.0))
    def test_quant_json_round_trip(self, params_b: float) -> None:
        """Output must survive JSON serialization round-trip."""
        hf_response = {
            "safetensors": {"total": params_b * 1e9},
            "config": {"architectures": ["RoundTripArch"]},
        }

        with patch(
            "olive_mcp_server.tools.agent_model_info._fetch_hf_metadata",
            return_value=hf_response,
        ):
            result = get_model_info("test-org/test-model")

        assert "error" not in result
        assert json.loads(json.dumps(result)) == result
