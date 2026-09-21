"""Unit tests for get_runtime_ep_hints (no live Studio, no Olive)."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock
from urllib.error import URLError

import pytest

from olive_mcp_server.mcp_server import _TOOL_IMPORTS
from olive_mcp_server.tools import studio_loopback
from olive_mcp_server.tools.runtime_ep_hints import get_runtime_ep_hints
from olive_mcp_server.tools.studio_loopback import ENV_API_URL

_PROBE_PAYLOAD: dict[str, Any] = {
    "probedAt": "2026-08-06T12:00:00.000Z",
    "platform": {
        "os": "linux",
        "arch": "x64",
        "cpuModel": "AMD EPYC 7763",
        "cpuCores": 64,
    },
    "nvidia": {"gpus": [{"name": "NVIDIA A100", "vramGb": 40}]},
    "rocm": {"gpus": []},
    "openvino": {"devices": ["CPU", "GPU"]},
    "onnxRuntimeProviders": [
        "CPUExecutionProvider",
        "CUDAExecutionProvider",
    ],
    "detectedProviders": [
        "CPUExecutionProvider",
        "CUDAExecutionProvider",
    ],
    "recommendedProvider": "CUDAExecutionProvider",
    "notes": [
        "NVIDIA GPU detected via nvidia-smi.",
        "QNN is Windows-first in this Studio release.",
    ],
}

_RUNTIME_PAYLOAD: dict[str, Any] = {
    "venvExists": True,
    "oliveInstalled": True,
    "platform": "linux",
    "hint": "default family ready",
    "families": {
        "default": {
            "family": "default",
            "exists": True,
            "oliveInstalled": True,
            "capabilities": {
                "cpu": {"usable": True},
                "directml": {"usable": False, "reason": "unsupported"},
            },
        },
        "cuda": {
            "family": "cuda",
            "exists": True,
            "oliveInstalled": True,
            "capabilities": {
                "cpu": {"usable": True},
                "cuda": {"usable": True},
            },
        },
    },
}


@pytest.fixture(autouse=True)
def _clear_studio_url(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(ENV_API_URL, raising=False)


def _set_loopback_url(monkeypatch: pytest.MonkeyPatch, base: str = "http://127.0.0.1:3000") -> None:
    monkeypatch.setenv(ENV_API_URL, base)


def _mock_response(payload: dict[str, Any] | bytes | str, *, status: int = 200) -> MagicMock:
    if isinstance(payload, dict):
        raw = json.dumps(payload).encode("utf-8")
    elif isinstance(payload, str):
        raw = payload.encode("utf-8")
    else:
        raw = payload

    resp = MagicMock()
    resp.status = status
    resp.getcode.return_value = status
    resp.read.return_value = raw
    resp.__enter__.return_value = resp
    resp.__exit__.return_value = False
    return resp


def _patch_opener(monkeypatch: pytest.MonkeyPatch, side_effect=None, return_value=None):
    opener = MagicMock()
    if side_effect is not None:
        opener.open.side_effect = side_effect
    else:
        opener.open.return_value = return_value
    monkeypatch.setattr(studio_loopback, "_OPENER", opener)
    return opener


def test_missing_api_url_returns_studio_unavailable():
    result = get_runtime_ep_hints()
    assert result["error"] == "studio_unavailable"
    assert ENV_API_URL in result["message"]


def test_non_loopback_url_returns_studio_unavailable(monkeypatch: pytest.MonkeyPatch):
    _set_loopback_url(monkeypatch, "http://example.com:3000")
    opener = _patch_opener(monkeypatch, return_value=_mock_response(_PROBE_PAYLOAD))

    result = get_runtime_ep_hints()

    assert result["error"] == "studio_unavailable"
    assert "loopback" in result["message"].lower()
    opener.open.assert_not_called()


def test_happy_path_projects_probe_and_runtime(monkeypatch: pytest.MonkeyPatch):
    _set_loopback_url(monkeypatch)

    def _open(request, timeout=None):  # noqa: ANN001
        url = request.full_url
        if "/api/system/hardware-probe" in url:
            return _mock_response(_PROBE_PAYLOAD)
        if "/api/env/runtime" in url:
            return _mock_response(_RUNTIME_PAYLOAD)
        raise AssertionError(f"unexpected url: {url}")

    opener = _patch_opener(monkeypatch, side_effect=_open)

    result = get_runtime_ep_hints()

    assert "error" not in result
    assert result["source"] == "olive_studio"
    assert result["studio_base"] == "http://127.0.0.1:3000"
    assert result["probed_at"] == _PROBE_PAYLOAD["probedAt"]
    assert result["platform"]["os"] == "linux"
    assert result["platform"]["cpu_model"] == "AMD EPYC 7763"
    assert "CUDAExecutionProvider" in result["onnx_runtime_providers"]
    assert result["recommended_provider"] == "CUDAExecutionProvider"
    assert result["hardware_flags"]["has_nvidia_gpu"] is True
    assert result["hardware_flags"]["has_rocm_gpu"] is False
    assert result["hardware_flags"]["openvino_devices"] == ["CPU", "GPU"]
    assert len(result["capabilities_summary"]) == 2
    cuda_family = next(c for c in result["capabilities_summary"] if c["family"] == "cuda")
    assert cuda_family["olive_installed"] is True
    assert "CUDAExecutionProvider" in cuda_family["providers"]
    assert "python" not in result
    assert "disclaimer" in result
    assert opener.open.call_count == 2


def test_probe_timeout_returns_studio_unavailable(monkeypatch: pytest.MonkeyPatch):
    _set_loopback_url(monkeypatch)
    _patch_opener(monkeypatch, side_effect=URLError(TimeoutError("timed out")))

    result = get_runtime_ep_hints()

    assert result["error"] == "studio_unavailable"
    assert "timed out" in result["message"].lower()


def test_runtime_failure_still_returns_probe_hints(monkeypatch: pytest.MonkeyPatch):
    _set_loopback_url(monkeypatch)

    def _open(request, timeout=None):  # noqa: ANN001
        url = request.full_url
        if "/api/system/hardware-probe" in url:
            return _mock_response(_PROBE_PAYLOAD)
        raise URLError(ConnectionRefusedError("runtime down"))

    _patch_opener(monkeypatch, side_effect=_open)

    result = get_runtime_ep_hints()

    assert "error" not in result
    assert result["source"] == "olive_studio"
    assert result["onnx_runtime_providers"] == _PROBE_PAYLOAD["onnxRuntimeProviders"]
    assert result["capabilities_summary"] == []
    assert result["hardware_flags"]["has_nvidia_gpu"] is True


def test_refresh_appends_query(monkeypatch: pytest.MonkeyPatch):
    _set_loopback_url(monkeypatch)
    calls: list[str] = []

    def _open(request, timeout=None):  # noqa: ANN001
        calls.append(request.full_url)
        if "hardware-probe" in request.full_url:
            return _mock_response(_PROBE_PAYLOAD)
        return _mock_response(_RUNTIME_PAYLOAD)

    _patch_opener(monkeypatch, side_effect=_open)

    result = get_runtime_ep_hints(refresh=True)

    assert "error" not in result
    assert any("refresh=true" in url for url in calls)


def test_registered_in_tool_imports():
    assert "get_runtime_ep_hints" in _TOOL_IMPORTS
    module_path, attr = _TOOL_IMPORTS["get_runtime_ep_hints"]
    assert module_path == "olive_mcp_server.tools.runtime_ep_hints"
    assert attr == "get_runtime_ep_hints"
