"""Unit tests for get_model_info tool (agent_model_info.py).

Covers HF API success, timeout/404 fallback, invalid input, confidence levels,
and JSON serialization round-trip.

Requirements: 13.5, 13.6, 13.7
"""

from __future__ import annotations

import json

import pytest

from olive_mcp_server.tools import agent_model_info

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _patch_hf(monkeypatch: pytest.MonkeyPatch, return_value):
    """Monkeypatch _fetch_hf_metadata to return a fixed value."""
    monkeypatch.setattr(agent_model_info, "_fetch_hf_metadata", lambda _model_id: return_value)


def _assert_json_roundtrip(result: dict):
    """Assert that the result survives JSON serialization round-trip."""
    assert json.loads(json.dumps(result)) == result


# ---------------------------------------------------------------------------
# Test: HF API success (params from safetensors)
# ---------------------------------------------------------------------------


class TestHfApiSuccess:
    """HF API returns valid metadata with safetensors.total."""

    def test_params_architecture_source_confidence(self, monkeypatch: pytest.MonkeyPatch):
        _patch_hf(
            monkeypatch,
            {
                "safetensors": {"total": 7_000_000_000},
                "config": {"architectures": ["LlamaForCausalLM"]},
            },
        )

        result = agent_model_info.get_model_info("meta-llama/Llama-3-8B")

        assert "error" not in result
        assert result["params_b"] == pytest.approx(7.0, rel=1e-3)
        assert result["architecture"] == "LlamaForCausalLM"
        assert result["source"] == "huggingface_api"
        assert result["confidence"] == "high"
        assert result["estimated_vram_gb"] == pytest.approx(14.0, rel=1e-3)
        assert result["side_effect"] is False

    def test_json_roundtrip(self, monkeypatch: pytest.MonkeyPatch):
        _patch_hf(
            monkeypatch,
            {
                "safetensors": {"total": 7_000_000_000},
                "config": {"architectures": ["LlamaForCausalLM"]},
            },
        )
        result = agent_model_info.get_model_info("meta-llama/Llama-3-8B")
        _assert_json_roundtrip(result)


# ---------------------------------------------------------------------------
# Test: HF timeout fallback (family default)
# ---------------------------------------------------------------------------


class TestHfTimeoutFallback:
    """HF API returns None (simulating timeout). Falls back to heuristic."""

    def test_heuristic_family_default(self, monkeypatch: pytest.MonkeyPatch):
        # Use a model ID without explicit size token so it hits family default
        _patch_hf(monkeypatch, None)

        result = agent_model_info.get_model_info("meta-llama/Llama-3-instruct")

        assert "error" not in result
        assert result["source"] == "heuristic"
        assert result["params_b"] == 8.0  # llama-3 family default
        assert result["confidence"] == "low"
        assert result["architecture"] == "unknown"

    def test_heuristic_explicit_size_token_in_name(self, monkeypatch: pytest.MonkeyPatch):
        """Model with '8B' in name gets medium confidence via size token regex."""
        _patch_hf(monkeypatch, None)

        result = agent_model_info.get_model_info("meta-llama/Llama-3-8B")

        assert "error" not in result
        assert result["source"] == "heuristic"
        assert result["params_b"] == 8.0
        assert result["confidence"] == "medium"
        assert result["architecture"] == "unknown"

    def test_json_roundtrip(self, monkeypatch: pytest.MonkeyPatch):
        _patch_hf(monkeypatch, None)
        result = agent_model_info.get_model_info("meta-llama/Llama-3-instruct")
        _assert_json_roundtrip(result)


# ---------------------------------------------------------------------------
# Test: HF 404 fallback (explicit size token)
# ---------------------------------------------------------------------------


class TestHf404Fallback:
    """HF API returns None (simulating 404). Model ID has explicit 7B token."""

    def test_heuristic_explicit_size_token(self, monkeypatch: pytest.MonkeyPatch):
        _patch_hf(monkeypatch, None)

        result = agent_model_info.get_model_info("someone/custom-7b-model")

        assert "error" not in result
        assert result["source"] == "heuristic"
        assert result["params_b"] == 7.0
        assert result["confidence"] == "medium"

    def test_json_roundtrip(self, monkeypatch: pytest.MonkeyPatch):
        _patch_hf(monkeypatch, None)
        result = agent_model_info.get_model_info("someone/custom-7b-model")
        _assert_json_roundtrip(result)


# ---------------------------------------------------------------------------
# Test: Invalid model_id
# ---------------------------------------------------------------------------


class TestInvalidModelId:
    """Invalid model_id values trigger validation error."""

    def test_empty_string_error(self):
        result = agent_model_info.get_model_info("")

        assert result["error"] == "invalid_model_id"
        assert "message" in result
        assert len(result["message"]) > 0

    def test_json_roundtrip(self):
        result = agent_model_info.get_model_info("")
        _assert_json_roundtrip(result)

    @pytest.mark.parametrize(
        "bad_id",
        [
            "../etc/passwd",  # path traversal
            "org/model\nname",  # control character (newline)
            "org/model\tname",  # control character (tab)
            "org/model?x=1",  # query string
            "org/model#frag",  # fragment
            "no-slash-here",  # missing slash
            "a" * 257,  # exceeds 256 chars
        ],
        ids=[
            "path_traversal",
            "control_char_newline",
            "control_char_tab",
            "query_string",
            "fragment",
            "missing_slash",
            "too_long",
        ],
    )
    def test_invalid_model_id_rejected(self, bad_id: str):
        result = agent_model_info.get_model_info(bad_id)
        assert result["error"] == "invalid_model_id"
        assert "message" in result
        assert len(result["message"]) > 0


# ---------------------------------------------------------------------------
# Test: Explicit size token confidence vs family default
# ---------------------------------------------------------------------------


class TestConfidenceLevels:
    """Family default (no explicit B token) gives confidence='low'."""

    def test_family_default_confidence_low(self, monkeypatch: pytest.MonkeyPatch):
        _patch_hf(monkeypatch, None)

        result = agent_model_info.get_model_info("org/phi-3.5-mini-instruct")

        assert "error" not in result
        assert result["confidence"] == "low"
        # phi-3.5 family default is 3.8B
        assert result["params_b"] == 3.8

    def test_explicit_size_token_confidence_medium(self, monkeypatch: pytest.MonkeyPatch):
        """Model ID with explicit 13B token gets medium confidence."""
        _patch_hf(monkeypatch, None)

        result = agent_model_info.get_model_info("org/some-model-13b-chat")

        assert "error" not in result
        assert result["confidence"] == "medium"
        assert result["params_b"] == 13.0

    def test_json_roundtrip(self, monkeypatch: pytest.MonkeyPatch):
        _patch_hf(monkeypatch, None)
        result = agent_model_info.get_model_info("org/phi-3.5-mini-instruct")
        _assert_json_roundtrip(result)


# ---------------------------------------------------------------------------
# Test: Additional coverage — derived fields
# ---------------------------------------------------------------------------


class TestDerivedFields:
    """VRAM estimate and recommended quantization."""

    def test_vram_estimate_formula(self, monkeypatch: pytest.MonkeyPatch):
        """estimated_vram_gb = params_b * 2.0"""
        _patch_hf(
            monkeypatch,
            {
                "safetensors": {"total": 3_000_000_000},
                "config": {"architectures": ["PhiForCausalLM"]},
            },
        )
        result = agent_model_info.get_model_info("microsoft/phi-3")
        assert result["estimated_vram_gb"] == pytest.approx(result["params_b"] * 2.0)

    def test_recommended_quant_int4_large(self, monkeypatch: pytest.MonkeyPatch):
        """params_b >= 6.0 -> int4"""
        _patch_hf(
            monkeypatch,
            {
                "safetensors": {"total": 8_000_000_000},
                "config": {"architectures": ["LlamaForCausalLM"]},
            },
        )
        result = agent_model_info.get_model_info("meta-llama/Llama-3-8B")
        assert result["recommended_quant"] == "int4"

    def test_recommended_quant_int8_small(self, monkeypatch: pytest.MonkeyPatch):
        """params_b < 6.0 -> int8"""
        _patch_hf(
            monkeypatch,
            {
                "safetensors": {"total": 2_000_000_000},
                "config": {"architectures": ["BertModel"]},
            },
        )
        result = agent_model_info.get_model_info("google/bert-base-uncased")
        assert result["recommended_quant"] == "int8"

    def test_model_type_classification(self, monkeypatch: pytest.MonkeyPatch):
        """model_type derived from _normalize_model_type."""
        _patch_hf(monkeypatch, None)
        result = agent_model_info.get_model_info("meta-llama/Llama-3-8B")
        assert result["model_type"] == "llm"


# ---------------------------------------------------------------------------
# Test: HF API success via config.num_parameters path
# ---------------------------------------------------------------------------


class TestHfConfigNumParameters:
    """HF API returns config.num_parameters instead of safetensors.total."""

    def test_params_from_config(self, monkeypatch: pytest.MonkeyPatch):
        _patch_hf(
            monkeypatch,
            {
                "config": {
                    "num_parameters": 1_500_000_000,
                    "architectures": ["WhisperForConditionalGeneration"],
                },
            },
        )
        result = agent_model_info.get_model_info("openai/whisper-large")
        assert result["params_b"] == pytest.approx(1.5, rel=1e-3)
        assert result["architecture"] == "WhisperForConditionalGeneration"
        assert result["source"] == "huggingface_api"
        assert result["confidence"] == "high"

    def test_json_roundtrip(self, monkeypatch: pytest.MonkeyPatch):
        _patch_hf(
            monkeypatch,
            {
                "config": {
                    "num_parameters": 1_500_000_000,
                    "architectures": ["WhisperForConditionalGeneration"],
                },
            },
        )
        result = agent_model_info.get_model_info("openai/whisper-large")
        _assert_json_roundtrip(result)


# ---------------------------------------------------------------------------
# Test: Mixtral MoE heuristic and architecture-based model type
# ---------------------------------------------------------------------------


class TestMixtralMoeHeuristic:
    """Mixtral MoE identifiers must be classified correctly, not as 7B."""

    def test_mixtral_8x7b_params(self, monkeypatch: pytest.MonkeyPatch):
        """Mixtral-8x7B yields ~46.7B params (MoE total), not 7B."""
        _patch_hf(monkeypatch, None)
        result = agent_model_info.get_model_info("mistralai/Mixtral-8x7B-Instruct-v0.1")
        assert "error" not in result
        assert result["params_b"] == 46.7
        assert result["source"] == "heuristic"
        _assert_json_roundtrip(result)


class TestArchitectureBasedModelType:
    """Opaque model IDs should classify from their HF architecture field."""

    def test_whisper_architecture_classifies_speech(self, monkeypatch: pytest.MonkeyPatch):
        """An opaque model ID with WhisperForConditionalGeneration → model_type=speech."""
        _patch_hf(
            monkeypatch,
            {
                "config": {
                    "num_parameters": 244_000_000,
                    "architectures": ["WhisperForConditionalGeneration"],
                },
            },
        )
        result = agent_model_info.get_model_info("some-org/random-model-name")
        assert "error" not in result
        assert result["architecture"] == "WhisperForConditionalGeneration"
        assert result["model_type"] == "speech"
        _assert_json_roundtrip(result)
