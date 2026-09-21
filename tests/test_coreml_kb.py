"""CoreML MCP knowledge base entry validation.

Validates Properties 9 and 10 from the macOS Unsigned Release & CoreML EP spec:
- Property 9: passes.json lists CoreMLExecutionProvider in hardware_requirements
  for compatible passes.
- Property 10: passes.json does NOT list CoreMLExecutionProvider in
  hardware_requirements for GPU-only passes.
- Hardware profile schema: hardware_profiles.json has a valid CoreML entry with
  required fields (target, execution_providers, platform.os, platform.arch, etc.).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parent.parent
_KB_DIR = _ROOT / "olive_mcp_server" / "knowledge_base"
_PROFILES_PATH = _KB_DIR / "hardware_profiles.json"
_PASSES_PATH = _KB_DIR / "passes.json"

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def profiles_data() -> dict:
    return json.loads(_PROFILES_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def passes_data() -> dict:
    return json.loads(_PASSES_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def passes_by_name(passes_data: dict) -> dict[str, dict]:
    """Map pass name -> pass entry for quick lookup."""
    return {p["name"]: p for p in passes_data["passes"] if "name" in p}


@pytest.fixture(scope="module")
def coreml_profile(profiles_data: dict) -> dict:
    """Return the CoreML hardware profile entry."""
    for profile in profiles_data.get("profiles", []):
        if "CoreMLExecutionProvider" in profile.get("execution_providers", []):
            return profile
    pytest.fail("CoreML profile not found in hardware_profiles.json")


# ---------------------------------------------------------------------------
# Property 9: CoreML listed in hardware_requirements for compatible passes
# Validates: Requirements 8.1
# ---------------------------------------------------------------------------

_COREML_COMPATIBLE_PASSES = [
    "OnnxConversion",
    "OnnxStaticQuantization",
    "OnnxDynamicQuantization",
    "OnnxBlockWiseRtnQuantization",
    "OnnxKquantQuantization",
    "QATQuantizer",
    "OnnxHqqQuantization",
    "LoRA",
]


@pytest.mark.parametrize("pass_name", _COREML_COMPATIBLE_PASSES)
def test_coreml_listed_in_compatible_pass_hardware_requirements(
    passes_by_name: dict[str, dict], pass_name: str
) -> None:
    """**Validates: Requirements 8.1**

    For any pass in the CoreML-compatible set, its entry in passes.json SHALL
    list CoreMLExecutionProvider in the hardware_requirements array.
    """
    assert pass_name in passes_by_name, f"pass {pass_name!r} not found in passes.json"
    entry = passes_by_name[pass_name]
    hw = entry.get("hardware_requirements", [])
    assert "CoreMLExecutionProvider" in hw, (
        f"CoreMLExecutionProvider missing from {pass_name}.hardware_requirements: {hw}"
    )


# ---------------------------------------------------------------------------
# Property 10: CoreML NOT listed in hardware_requirements for GPU-only passes
# Validates: Requirements 8.2
# ---------------------------------------------------------------------------

_GPU_ONLY_PASSES = [
    "AutoAWQQuantizer",
    "GptqQuantizer",
    "Gptq",
    "SpinQuant",
    "QuaRot",
    "QLoRA",
]


@pytest.mark.parametrize("pass_name", _GPU_ONLY_PASSES)
def test_coreml_not_in_gpu_only_pass_hardware_requirements(passes_by_name: dict[str, dict], pass_name: str) -> None:
    """**Validates: Requirements 8.2**

    For any pass in the GPU-only set, its entry in passes.json SHALL NOT list
    CoreMLExecutionProvider in the hardware_requirements array.
    """
    assert pass_name in passes_by_name, f"pass {pass_name!r} not found in passes.json"
    entry = passes_by_name[pass_name]
    hw = entry.get("hardware_requirements", [])
    assert "CoreMLExecutionProvider" not in hw, (
        f"CoreMLExecutionProvider must NOT appear in {pass_name}.hardware_requirements"
    )


# ---------------------------------------------------------------------------
# Hardware profile schema validation
# Validates: Requirements 7.1, 7.2, 7.3
# ---------------------------------------------------------------------------


def test_coreml_profile_has_required_target(coreml_profile: dict) -> None:
    """Profile must have a non-empty target name."""
    assert isinstance(coreml_profile.get("target"), str)
    assert coreml_profile["target"].strip()


def test_coreml_profile_has_execution_providers(coreml_profile: dict) -> None:
    """Profile must list CoreMLExecutionProvider."""
    eps = coreml_profile.get("execution_providers", [])
    assert "CoreMLExecutionProvider" in eps


def test_coreml_profile_platform_os(coreml_profile: dict) -> None:
    """Platform.os must be macOS."""
    platform = coreml_profile.get("platform", {})
    assert platform.get("os") == "macOS"


def test_coreml_profile_platform_arch(coreml_profile: dict) -> None:
    """Platform.arch must be arm64."""
    platform = coreml_profile.get("platform", {})
    assert platform.get("arch") == "arm64"


def test_coreml_profile_platform_description(coreml_profile: dict) -> None:
    """Platform.description must be a non-empty string mentioning Apple Silicon."""
    platform = coreml_profile.get("platform", {})
    desc = platform.get("description", "")
    assert isinstance(desc, str) and desc.strip()
    assert "Apple Silicon" in desc


def test_coreml_profile_compatible_passes(coreml_profile: dict) -> None:
    """Compatible passes must include all CoreML-compatible quantization and fine-tuning methods."""
    compatible = set(coreml_profile.get("compatible_passes", []))
    expected = {
        "OnnxConversion",
        "OnnxStaticQuantization",
        "OnnxDynamicQuantization",
        "OnnxBlockWiseRtnQuantization",
        "OnnxKquantQuantization",
        "QATQuantizer",
        "OnnxHqqQuantization",
        "LoRA",
    }
    missing = expected - compatible
    assert not missing, f"CoreML profile missing compatible_passes: {missing}"


def test_coreml_profile_incompatible_passes(coreml_profile: dict) -> None:
    """Incompatible passes must list all GPU-only quantization methods."""
    incompatible = set(coreml_profile.get("incompatible_passes", []))
    expected = {
        "AutoAWQQuantizer",
        "GptqQuantizer",
        "Gptq",
        "SpinQuant",
        "QuaRot",
        "QLoRA",
    }
    missing = expected - incompatible
    assert not missing, f"CoreML profile missing incompatible_passes: {missing}"


@pytest.mark.parametrize("field", ["compatible_passes", "incompatible_passes"])
def test_coreml_profile_uses_registered_pass_ids(
    coreml_profile: dict, passes_by_name: dict[str, dict], field: str
) -> None:
    """Every CoreML profile pass reference must resolve to a passes.json entry."""
    referenced = set(coreml_profile.get(field, []))
    unknown = referenced - passes_by_name.keys()
    assert not unknown, f"CoreML profile {field} contains unknown pass IDs: {unknown}"


def test_coreml_profile_has_notes(coreml_profile: dict) -> None:
    """Profile notes must mention coremltools and Neural Engine."""
    notes = coreml_profile.get("notes", "")
    assert isinstance(notes, str) and notes.strip()
    assert "coremltools" in notes.lower()
    assert "neural engine" in notes.lower()
