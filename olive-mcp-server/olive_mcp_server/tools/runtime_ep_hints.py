"""Tool: get_runtime_ep_hints — read-only Studio probe/runtime projection.

Proxies Olive Studio ``/api/system/hardware-probe`` and ``/api/env/runtime``
when ``OLIVE_STUDIO_API_URL`` is set. No install logic, no Olive execution.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import urlencode

from .studio_loopback import resolve_studio_base, studio_request

_PROBE_PATH = "/api/system/hardware-probe"
_RUNTIME_PATH = "/api/env/runtime"
_MAX_NOTES = 12

_DISCLAIMER = (
    "Hints from local Olive Studio probe/venv only. Static MCP KB remains authoritative when Studio is unavailable."
)

# capability key → ORT-style provider name for a compact summary
_CAPABILITY_PROVIDER = {
    "cpu": "CPUExecutionProvider",
    "directml": "DmlExecutionProvider",
    "cuda": "CUDAExecutionProvider",
    "openvino": "OpenVINOExecutionProvider",
    "tensorrt": "TensorrtExecutionProvider",
    "tensorrtRtx": "NvTensorRTRTXExecutionProvider",
    "qnnPreparation": "QNNExecutionProvider",
    "qnnInference": "QNNExecutionProvider",
}


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _str_list(value: Any) -> list[str]:
    return [str(item) for item in _as_list(value) if item is not None]


def _platform_subset(probe: dict[str, Any]) -> dict[str, Any]:
    platform = _as_dict(probe.get("platform"))
    out: dict[str, Any] = {}
    if "os" in platform:
        out["os"] = platform["os"]
    if "arch" in platform:
        out["arch"] = platform["arch"]
    cpu = platform.get("cpuModel") or platform.get("cpu_model")
    if cpu is not None:
        out["cpu_model"] = cpu
    return out


def _gpu_count(section: Any) -> int:
    block = _as_dict(section)
    gpus = block.get("gpus")
    return len(gpus) if isinstance(gpus, list) else 0


def _has_directml_hint(
    providers: list[str],
    detected: list[str],
) -> bool:
    needle = "dml"
    for name in (*providers, *detected):
        lowered = name.lower()
        if needle in lowered or "directml" in lowered:
            return True
    return False


def _openvino_devices(probe: dict[str, Any]) -> list[Any]:
    ov = _as_dict(probe.get("openvino"))
    devices = ov.get("devices")
    return list(devices) if isinstance(devices, list) else []


def _providers_from_capabilities(caps: dict[str, Any]) -> list[str]:
    """Project usable capability flags into EP names (no python paths)."""
    seen: list[str] = []
    for key, provider in _CAPABILITY_PROVIDER.items():
        status = caps.get(key)
        if not isinstance(status, dict):
            continue
        if status.get("usable") is not True:
            continue
        if provider not in seen:
            seen.append(provider)
    return seen


def _capabilities_summary(runtime: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not runtime:
        return []
    families = runtime.get("families")
    if not isinstance(families, dict):
        return []

    summary: list[dict[str, Any]] = []
    for family_name, status in families.items():
        if not isinstance(status, dict):
            continue
        caps = _as_dict(status.get("capabilities"))
        summary.append(
            {
                "family": str(status.get("family") or family_name),
                "olive_installed": bool(status.get("oliveInstalled", status.get("olive_installed", False))),
                "providers": _providers_from_capabilities(caps),
            }
        )
    return summary


def _project_success(
    *,
    studio_base: str,
    probe: dict[str, Any],
    runtime: dict[str, Any] | None,
) -> dict[str, Any]:
    providers = _str_list(probe.get("onnxRuntimeProviders") or probe.get("onnx_runtime_providers"))
    detected = _str_list(probe.get("detectedProviders") or probe.get("detected_providers"))
    recommended = probe.get("recommendedProvider") or probe.get("recommended_provider")
    notes_raw = _as_list(probe.get("notes"))
    notes = [str(n) for n in notes_raw if n is not None][:_MAX_NOTES]

    return {
        "source": "olive_studio",
        "studio_base": studio_base,
        "probed_at": probe.get("probedAt") or probe.get("probed_at"),
        "platform": _platform_subset(probe),
        "onnx_runtime_providers": providers,
        "detected_providers": detected,
        "recommended_provider": recommended if recommended is not None else None,
        "hardware_flags": {
            "has_nvidia_gpu": _gpu_count(probe.get("nvidia")) > 0,
            "has_rocm_gpu": _gpu_count(probe.get("rocm")) > 0,
            "has_directml_hint": _has_directml_hint(providers, detected),
            "openvino_devices": _openvino_devices(probe),
        },
        "capabilities_summary": _capabilities_summary(runtime),
        "notes": notes,
        "disclaimer": _DISCLAIMER,
    }


def get_runtime_ep_hints(refresh: bool = False) -> dict[str, Any]:
    """Project Studio hardware-probe + env/runtime into EP hints for agents.

    Read-only. Requires ``OLIVE_STUDIO_API_URL`` (loopback only). Does not
    install packages or run Olive.

    Args:
        refresh: When True, request ``?refresh=true`` on the hardware probe
            to bypass Studio's probe cache.

    Returns:
        Stable hint object, or structured error
        (``studio_unavailable`` / ``invalid_bridge_response``).
    """
    base, resolve_err = resolve_studio_base()
    if resolve_err is not None:
        return resolve_err

    probe_path = _PROBE_PATH
    if refresh:
        probe_path = f"{_PROBE_PATH}?{urlencode({'refresh': 'true'})}"

    probe = studio_request("GET", probe_path)
    if "error" in probe:
        return probe

    # Runtime is best-effort: probe-derived hints still succeed if it fails.
    runtime_raw = studio_request("GET", _RUNTIME_PATH)
    runtime: dict[str, Any] | None = None if "error" in runtime_raw else runtime_raw

    return _project_success(studio_base=base, probe=probe, runtime=runtime)  # type: ignore[arg-type]
