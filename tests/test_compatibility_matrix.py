"""Schema, provenance, and pass-integrity checks for compatibility_matrix.json.

Validates the real knowledge-base matrix against passes.json and
schemas/compatibility-v1.json rules. Negative cases use in-memory fixtures.
No Olive execution, network, or model downloads.
"""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parent.parent
_KB_DIR = _ROOT / "olive_mcp_server" / "knowledge_base"
_MATRIX_PATH = _KB_DIR / "compatibility_matrix.json"
_PASSES_PATH = _KB_DIR / "passes.json"
_SCHEMA_PATH = _ROOT / "schemas" / "compatibility-v1.json"

_SUPPORT_STATES = frozenset({"supported", "warning", "unsupported"})
_EVIDENCE_TYPES = frozenset(
    {
        "olive_docs",
        "ort_docs",
        "github",
        "availability",
        "benchmark",
        "release_notes",
    }
)
# Evidence types whose ``version`` is an Olive release (must sit in support window).
_OLIVE_EVIDENCE_TYPES = frozenset({"olive_docs", "release_notes"})
_MATRIX_VERSION_RE = re.compile(r"^\d+\.\d+\.\d+$")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_FRAMEWORKS = frozenset({"PyTorch", "ONNX", "HuggingFace", "TensorFlow"})


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    assert isinstance(data, dict), f"{path.name} root must be an object"
    return data


@pytest.fixture(scope="module")
def matrix() -> dict[str, Any]:
    return _load_json(_MATRIX_PATH)


@pytest.fixture(scope="module")
def pass_names() -> set[str]:
    data = _load_json(_PASSES_PATH)
    passes = data.get("passes", [])
    assert isinstance(passes, list) and passes, "passes.json must list passes"
    names = {p["name"] for p in passes if isinstance(p, dict) and "name" in p}
    assert names, "passes.json yielded no pass names"
    return names


@pytest.fixture(scope="module")
def schema() -> dict[str, Any]:
    return _load_json(_SCHEMA_PATH)


# ---------------------------------------------------------------------------
# Version helpers
# ---------------------------------------------------------------------------


def _parse_version(value: str) -> tuple[int, ...]:
    """Parse a dotted version string into a comparable int tuple.

    Accepts ``0.12.1``, ``1.19``, ``0.8.0rc1`` (numeric prefix only).
    Raises ValueError on empty/malformed input.
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"empty version: {value!r}")
    # Strip common pre-release / build suffixes for comparison.
    core = re.split(r"[^0-9.]", value.strip(), maxsplit=1)[0].rstrip(".")
    if not core or not re.fullmatch(r"\d+(\.\d+)*", core):
        raise ValueError(f"malformed version: {value!r}")
    return tuple(int(part) for part in core.split("."))


def _version_in_range(version: str, vmin: str, vmax: str) -> bool:
    """Return True if version is within [vmin, vmax] (inclusive, tuple compare)."""
    v = _parse_version(version)
    lo = _parse_version(vmin)
    hi = _parse_version(vmax)
    # Pad to equal length for lexicographic tuple compare.
    width = max(len(v), len(lo), len(hi))
    v = v + (0,) * (width - len(v))
    lo = lo + (0,) * (width - len(lo))
    hi = hi + (0,) * (width - len(hi))
    return lo <= v <= hi


# ---------------------------------------------------------------------------
# Claim iteration + validation
# ---------------------------------------------------------------------------


def _iter_claims(
    matrix_data: dict[str, Any],
) -> Iterator[tuple[str, str, str, dict[str, Any]]]:
    """Yield (model_name, hardware, claim_key, claim_body) for every pass claim."""
    for entry in matrix_data.get("models") or []:
        if not isinstance(entry, dict):
            continue
        model = entry.get("model", "<missing>")
        hardware = entry.get("hardware") or {}
        if not isinstance(hardware, dict):
            continue
        for hw_name, passes in hardware.items():
            if not isinstance(passes, dict):
                continue
            for claim_key, claim in passes.items():
                if isinstance(claim, dict):
                    yield str(model), str(hw_name), str(claim_key), claim


def _collect_top_level_errors(matrix_data: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for key in ("version", "last_updated", "models"):
        if key not in matrix_data:
            errors.append(f"missing top-level field: {key}")

    version = matrix_data.get("version")
    if version is not None and (not isinstance(version, str) or not _MATRIX_VERSION_RE.fullmatch(version)):
        errors.append(f"version must be semver X.Y.Z, got {version!r}")

    last_updated = matrix_data.get("last_updated")
    if last_updated is not None and (not isinstance(last_updated, str) or not _DATE_RE.fullmatch(last_updated)):
        errors.append(f"last_updated must be YYYY-MM-DD, got {last_updated!r}")

    models = matrix_data.get("models")
    if models is not None and (not isinstance(models, list) or len(models) < 1):
        errors.append("models must be a non-empty array")
    return errors


def _collect_olive_window(
    matrix_data: dict[str, Any],
) -> tuple[list[str], str | None, str | None]:
    errors: list[str] = []
    ovs = matrix_data.get("olive_version_support")
    window_min: str | None = None
    window_max: str | None = None
    if ovs is None:
        return errors, window_min, window_max
    if not isinstance(ovs, dict):
        errors.append("olive_version_support must be an object")
        return errors, window_min, window_max

    window_min = ovs.get("min")
    window_max = ovs.get("max")
    for label, raw in (("min", window_min), ("max", window_max)):
        if raw is None:
            continue
        try:
            _parse_version(str(raw))
        except ValueError as exc:
            errors.append(f"olive_version_support.{label}: {exc}")
    if window_min is not None and window_max is not None:
        try:
            lo = _parse_version(str(window_min))
            hi = _parse_version(str(window_max))
            width = max(len(lo), len(hi))
            lo_p = lo + (0,) * (width - len(lo))
            hi_p = hi + (0,) * (width - len(hi))
            if lo_p > hi_p:
                errors.append(f"olive_version_support malformed range: min={window_min!r} > max={window_max!r}")
        except ValueError:
            pass  # already recorded above
    return errors, window_min, window_max


def _collect_evidence_window_errors(
    claim_loc: str,
    *,
    etype: Any,
    ever: Any,
    enforce_olive_window: bool,
    window_min: str | None,
    window_max: str | None,
) -> list[str]:
    if not (
        enforce_olive_window
        and window_min is not None
        and window_max is not None
        and etype in _OLIVE_EVIDENCE_TYPES
        and isinstance(ever, str)
        and ever.strip()
    ):
        return []
    try:
        if _version_in_range(ever, str(window_min), str(window_max)):
            return []
        return [f"{claim_loc}: evidence.version {ever!r} outside olive_version_support [{window_min}, {window_max}]"]
    except ValueError as exc:
        return [f"{claim_loc}: evidence.version unparseable for window check: {exc}"]


def _collect_claim_evidence_errors(
    claim: dict[str, Any],
    claim_loc: str,
    *,
    enforce_olive_window: bool,
    window_min: str | None,
    window_max: str | None,
) -> list[str]:
    errors: list[str] = []
    if "evidence" not in claim:
        return errors
    evidence = claim.get("evidence")
    if not isinstance(evidence, dict):
        errors.append(f"{claim_loc}: evidence must be an object")
        return errors

    for ereq in ("reference", "type", "version"):
        if ereq not in evidence:
            errors.append(f"{claim_loc}: evidence missing required field {ereq!r}")
    ref = evidence.get("reference")
    if ref is not None and (not isinstance(ref, str) or not ref.strip()):
        errors.append(f"{claim_loc}: evidence.reference must be non-empty")
    etype = evidence.get("type")
    if etype is not None and etype not in _EVIDENCE_TYPES:
        errors.append(f"{claim_loc}: invalid evidence.type {etype!r}; expected one of {sorted(_EVIDENCE_TYPES)}")
    ever = evidence.get("version")
    if ever is not None and (not isinstance(ever, str) or not ever.strip()):
        errors.append(f"{claim_loc}: evidence.version must be non-empty")

    errors.extend(
        _collect_evidence_window_errors(
            claim_loc,
            etype=etype,
            ever=ever,
            enforce_olive_window=enforce_olive_window,
            window_min=window_min,
            window_max=window_max,
        )
    )
    return errors


def _collect_claim_errors(
    claim: dict[str, Any],
    claim_loc: str,
    *,
    model_name: str,
    hw_name: str,
    known_passes: set[str],
    seen_triples: set[tuple[str, str, str]],
    enforce_olive_window: bool,
    window_min: str | None,
    window_max: str | None,
) -> list[str]:
    errors: list[str] = []
    for req in ("support", "olive_pass", "evidence"):
        if req not in claim:
            errors.append(f"{claim_loc}: missing required field {req!r}")

    support = claim.get("support")
    if support is not None and support not in _SUPPORT_STATES:
        errors.append(f"{claim_loc}: invalid support state {support!r}; expected one of {sorted(_SUPPORT_STATES)}")

    olive_pass = claim.get("olive_pass")
    if olive_pass is not None:
        if not isinstance(olive_pass, str) or not olive_pass.strip():
            errors.append(f"{claim_loc}: olive_pass must be non-empty string")
        else:
            if olive_pass not in known_passes:
                errors.append(f"{claim_loc}: unknown olive_pass {olive_pass!r} (not in passes.json)")
            triple = (str(model_name), str(hw_name), olive_pass)
            if triple in seen_triples:
                errors.append(f"duplicate model/hardware/pass claim: {model_name!r} / {hw_name!r} / {olive_pass!r}")
            else:
                seen_triples.add(triple)

    errors.extend(
        _collect_claim_evidence_errors(
            claim,
            claim_loc,
            enforce_olive_window=enforce_olive_window,
            window_min=window_min,
            window_max=window_max,
        )
    )
    return errors


def _collect_frameworks_errors(loc: str, model_name: str, frameworks: Any) -> list[str]:
    if frameworks is None:
        return []
    if not isinstance(frameworks, list) or len(frameworks) < 1:
        return [f"{loc} ({model_name}): frameworks must be non-empty array"]
    return [f"{loc} ({model_name}): unknown framework {fw!r}" for fw in frameworks if fw not in _FRAMEWORKS]


def _collect_hardware_map_errors(
    loc: str,
    model_name: str,
    hardware: Any,
    *,
    known_passes: set[str],
    seen_triples: set[tuple[str, str, str]],
    enforce_olive_window: bool,
    window_min: str | None,
    window_max: str | None,
) -> list[str]:
    if hardware is None:
        return []
    if not isinstance(hardware, dict):
        return [f"{loc} ({model_name}): hardware must be an object"]

    errors: list[str] = []
    for hw_name, passes in hardware.items():
        hw_loc = f"{loc} ({model_name}) / hardware[{hw_name!r}]"
        if not isinstance(passes, dict):
            errors.append(f"{hw_loc}: pass map must be an object")
            continue
        for claim_key, claim in passes.items():
            claim_loc = f"{hw_loc} / {claim_key}"
            if not isinstance(claim, dict):
                errors.append(f"{claim_loc}: claim must be an object")
                continue
            errors.extend(
                _collect_claim_errors(
                    claim,
                    claim_loc,
                    model_name=str(model_name),
                    hw_name=str(hw_name),
                    known_passes=known_passes,
                    seen_triples=seen_triples,
                    enforce_olive_window=enforce_olive_window,
                    window_min=window_min,
                    window_max=window_max,
                )
            )
    return errors


def _collect_model_entry_errors(
    entry: Any,
    idx: int,
    *,
    known_passes: set[str],
    seen_models: set[str],
    seen_triples: set[tuple[str, str, str]],
    enforce_olive_window: bool,
    window_min: str | None,
    window_max: str | None,
) -> list[str]:
    errors: list[str] = []
    loc = f"models[{idx}]"
    if not isinstance(entry, dict):
        errors.append(f"{loc}: entry must be an object")
        return errors

    for req in ("model", "frameworks", "hardware"):
        if req not in entry:
            errors.append(f"{loc}: missing required field {req!r}")

    model_name = entry.get("model")
    if not isinstance(model_name, str) or not model_name.strip():
        errors.append(f"{loc}: model must be a non-empty string")
        model_name = f"<invalid-{idx}>"
    elif model_name in seen_models:
        errors.append(f"duplicate model entry: {model_name!r}")
    else:
        seen_models.add(model_name)

    errors.extend(_collect_frameworks_errors(loc, str(model_name), entry.get("frameworks")))
    errors.extend(
        _collect_hardware_map_errors(
            loc,
            str(model_name),
            entry.get("hardware"),
            known_passes=known_passes,
            seen_triples=seen_triples,
            enforce_olive_window=enforce_olive_window,
            window_min=window_min,
            window_max=window_max,
        )
    )
    return errors


def collect_matrix_errors(
    matrix_data: dict[str, Any],
    known_passes: set[str],
    *,
    enforce_olive_window: bool = True,
) -> list[str]:
    """Return human-readable validation errors (empty list => valid).

    Checks schema-level required fields, support enum, evidence provenance,
    pass registry membership, uniqueness, olive_version_support shape, and
    (optionally) that olive-scoped evidence versions sit inside the declared
    Olive support window.
    """
    errors = _collect_top_level_errors(matrix_data)
    window_errors, window_min, window_max = _collect_olive_window(matrix_data)
    errors.extend(window_errors)

    models = matrix_data.get("models")
    if not isinstance(models, list):
        return errors

    seen_models: set[str] = set()
    seen_triples: set[tuple[str, str, str]] = set()
    for idx, entry in enumerate(models):
        errors.extend(
            _collect_model_entry_errors(
                entry,
                idx,
                known_passes=known_passes,
                seen_models=seen_models,
                seen_triples=seen_triples,
                enforce_olive_window=enforce_olive_window,
                window_min=window_min,
                window_max=window_max,
            )
        )
    return errors


def _minimal_valid_matrix(
    *,
    olive_pass: str = "OnnxConversion",
    support: str = "supported",
    evidence_version: str = "0.12.1",
    evidence_type: str = "olive_docs",
    window_min: str = "0.12.0",
    window_max: str = "0.12.1",
) -> dict[str, Any]:
    """Build a tiny valid matrix for negative-case mutation."""
    return {
        "version": "0.0.0",
        "last_updated": "2026-08-05",
        "olive_version_support": {"min": window_min, "max": window_max},
        "models": [
            {
                "model": "Fixture Model",
                "frameworks": ["ONNX"],
                "hardware": {
                    "NVIDIA RTX 4090": {
                        olive_pass: {
                            "support": support,
                            "olive_pass": olive_pass,
                            "note": "fixture",
                            "evidence": {
                                "reference": "https://example.com/docs",
                                "type": evidence_type,
                                "version": evidence_version,
                            },
                        }
                    }
                },
            }
        ],
    }


# ===========================================================================
# Positive: real matrix
# ===========================================================================


def test_matrix_file_exists() -> None:
    # Arrange / Act / Assert
    assert _MATRIX_PATH.is_file(), f"missing {_MATRIX_PATH}"
    assert _PASSES_PATH.is_file(), f"missing {_PASSES_PATH}"
    assert _SCHEMA_PATH.is_file(), f"missing {_SCHEMA_PATH}"


def test_real_matrix_has_required_top_level_fields(matrix: dict[str, Any]) -> None:
    # Arrange — fixture loads real matrix
    # Act / Assert
    assert matrix.get("version") == "0.4.0"
    assert _MATRIX_VERSION_RE.fullmatch(matrix["version"])
    assert _DATE_RE.fullmatch(matrix["last_updated"])
    assert isinstance(matrix["models"], list) and len(matrix["models"]) >= 1
    ovs = matrix.get("olive_version_support")
    assert isinstance(ovs, dict)
    assert "min" in ovs and "max" in ovs


def test_real_matrix_passes_full_validation(matrix: dict[str, Any], pass_names: set[str]) -> None:
    """Every claim has olive_pass + evidence; passes exist; no duplicates."""
    # Arrange / Act
    errors = collect_matrix_errors(matrix, pass_names, enforce_olive_window=True)
    # Assert
    assert errors == [], "real matrix validation failed:\n" + "\n".join(errors)


def test_real_matrix_claim_count_matches_expansion(matrix: dict[str, Any]) -> None:
    """v0.3.x expansion annotated 169+ evidence-backed claims (subtask 10 + EP gaps)."""
    # Arrange / Act
    claims = list(_iter_claims(matrix))
    # Assert — allow growth; never shrink below the evidence-backed baseline
    assert len(claims) >= 169, f"expected >=169 claims, got {len(claims)}"
    assert len(matrix["models"]) >= 20


def test_real_matrix_every_claim_has_schema_fields(matrix: dict[str, Any]) -> None:
    # Arrange
    claims = list(_iter_claims(matrix))
    assert claims, "matrix has no pass claims"

    # Act / Assert
    for model, hw, key, claim in claims:
        loc = f"{model}/{hw}/{key}"
        assert "support" in claim, f"{loc}: missing support"
        assert claim["support"] in _SUPPORT_STATES, f"{loc}: bad support"
        assert isinstance(claim.get("olive_pass"), str) and claim["olive_pass"].strip(), f"{loc}: missing olive_pass"
        evidence = claim.get("evidence")
        assert isinstance(evidence, dict), f"{loc}: missing evidence object"
        assert isinstance(evidence.get("reference"), str) and evidence["reference"].strip(), (
            f"{loc}: empty evidence.reference"
        )
        assert evidence.get("type") in _EVIDENCE_TYPES, f"{loc}: bad evidence.type"
        assert isinstance(evidence.get("version"), str) and evidence["version"].strip(), (
            f"{loc}: empty evidence.version"
        )


def test_real_matrix_olive_passes_exist_in_passes_json(matrix: dict[str, Any], pass_names: set[str]) -> None:
    # Arrange / Act
    unknown = sorted(
        {claim["olive_pass"] for _, _, _, claim in _iter_claims(matrix) if claim.get("olive_pass") not in pass_names}
    )
    # Assert
    assert unknown == [], f"unknown olive_pass values: {unknown}"


def test_real_matrix_no_duplicate_model_hardware_pass(matrix: dict[str, Any]) -> None:
    # Arrange
    seen: set[tuple[str, str, str]] = set()
    dupes: list[str] = []

    # Act
    for model, hw, _key, claim in _iter_claims(matrix):
        olive_pass = claim.get("olive_pass", "")
        triple = (model, hw, olive_pass)
        if triple in seen:
            dupes.append(f"{model} / {hw} / {olive_pass}")
        seen.add(triple)

    # Assert
    assert dupes == [], f"duplicate claims: {dupes}"


def test_real_matrix_support_states_valid(matrix: dict[str, Any]) -> None:
    # Arrange / Act
    invalid = [
        f"{model}/{hw}/{key}={claim.get('support')!r}"
        for model, hw, key, claim in _iter_claims(matrix)
        if claim.get("support") not in _SUPPORT_STATES
    ]
    # Assert
    assert invalid == [], f"invalid support states: {invalid}"


def test_real_matrix_olive_evidence_within_support_window(matrix: dict[str, Any]) -> None:
    # Arrange
    ovs = matrix.get("olive_version_support") or {}
    vmin, vmax = ovs.get("min"), ovs.get("max")
    assert vmin and vmax, "matrix must declare olive_version_support"

    # Act
    outside: list[str] = []
    for model, hw, key, claim in _iter_claims(matrix):
        evidence = claim.get("evidence") or {}
        if evidence.get("type") not in _OLIVE_EVIDENCE_TYPES:
            continue
        ver = evidence.get("version", "")
        try:
            if not _version_in_range(str(ver), str(vmin), str(vmax)):
                outside.append(f"{model}/{hw}/{key} version={ver!r}")
        except ValueError as exc:
            outside.append(f"{model}/{hw}/{key} unparseable: {exc}")

    # Assert
    assert outside == [], f"olive evidence outside support window: {outside}"


def test_schema_documents_required_pass_compat_fields(schema: dict[str, Any]) -> None:
    # Arrange
    pass_compat = schema["$defs"]["pass_compat"]
    evidence = schema["$defs"]["pass_evidence"]
    # Act / Assert
    assert set(pass_compat["required"]) >= {"support", "olive_pass", "evidence"}
    assert set(evidence["required"]) >= {"reference", "type", "version"}
    assert set(pass_compat["properties"]["support"]["enum"]) == _SUPPORT_STATES


# ===========================================================================
# Negative: fixtures / mutated matrices
# ===========================================================================


def test_rejects_unknown_pass_name(pass_names: set[str]) -> None:
    # Arrange
    data = _minimal_valid_matrix(olive_pass="NotARealOlivePass")
    # Act
    errors = collect_matrix_errors(data, pass_names)
    # Assert
    assert any("unknown olive_pass" in e for e in errors), errors


def test_rejects_absent_provenance_missing_evidence(pass_names: set[str]) -> None:
    # Arrange
    data = _minimal_valid_matrix()
    claim = data["models"][0]["hardware"]["NVIDIA RTX 4090"]["OnnxConversion"]
    del claim["evidence"]
    # Act
    errors = collect_matrix_errors(data, pass_names)
    # Assert
    assert any("missing required field 'evidence'" in e for e in errors), errors


def test_rejects_absent_provenance_empty_reference(pass_names: set[str]) -> None:
    # Arrange
    data = _minimal_valid_matrix()
    claim = data["models"][0]["hardware"]["NVIDIA RTX 4090"]["OnnxConversion"]
    claim["evidence"]["reference"] = "   "
    # Act
    errors = collect_matrix_errors(data, pass_names)
    # Assert
    assert any("evidence.reference must be non-empty" in e for e in errors), errors


def test_rejects_absent_provenance_missing_evidence_fields(pass_names: set[str]) -> None:
    # Arrange
    data = _minimal_valid_matrix()
    claim = data["models"][0]["hardware"]["NVIDIA RTX 4090"]["OnnxConversion"]
    claim["evidence"] = {"reference": "https://example.com"}  # no type/version
    # Act
    errors = collect_matrix_errors(data, pass_names)
    # Assert
    assert any("evidence missing required field 'type'" in e for e in errors), errors
    assert any("evidence missing required field 'version'" in e for e in errors), errors


def test_rejects_duplicate_model_hardware_pass_claims(pass_names: set[str]) -> None:
    # Arrange — two claim keys pointing at the same olive_pass under one HW
    data = _minimal_valid_matrix(olive_pass="OnnxConversion")
    hw = data["models"][0]["hardware"]["NVIDIA RTX 4090"]
    hw["OnnxConversion_dup"] = copy.deepcopy(hw["OnnxConversion"])
    # Act
    errors = collect_matrix_errors(data, pass_names)
    # Assert
    assert any("duplicate model/hardware/pass claim" in e for e in errors), errors


def test_rejects_duplicate_model_entries(pass_names: set[str]) -> None:
    # Arrange
    data = _minimal_valid_matrix()
    data["models"].append(copy.deepcopy(data["models"][0]))
    # Act
    errors = collect_matrix_errors(data, pass_names)
    # Assert
    assert any("duplicate model entry" in e for e in errors), errors


def test_rejects_invalid_support_state(pass_names: set[str]) -> None:
    # Arrange
    data = _minimal_valid_matrix(support="maybe")
    # Act
    errors = collect_matrix_errors(data, pass_names)
    # Assert
    assert any("invalid support state" in e for e in errors), errors


def test_accepts_all_valid_support_states(pass_names: set[str]) -> None:
    # Arrange / Act / Assert
    for state in sorted(_SUPPORT_STATES):
        data = _minimal_valid_matrix(support=state)
        errors = collect_matrix_errors(data, pass_names)
        assert errors == [], f"support={state!r} should be valid, got {errors}"


def test_rejects_malformed_version_range_min_greater_than_max(
    pass_names: set[str],
) -> None:
    # Arrange
    data = _minimal_valid_matrix(window_min="0.13.0", window_max="0.12.0")
    # evidence version also won't matter once range is invalid
    # Act
    errors = collect_matrix_errors(data, pass_names, enforce_olive_window=True)
    # Assert
    assert any("malformed range" in e for e in errors), errors


def test_rejects_malformed_version_range_unparseable(pass_names: set[str]) -> None:
    # Arrange
    data = _minimal_valid_matrix(window_min="not-a-version", window_max="0.12.1")
    # Act
    errors = collect_matrix_errors(data, pass_names)
    # Assert
    assert any("olive_version_support.min" in e for e in errors), errors


def test_rejects_claim_outside_olive_support_window(pass_names: set[str]) -> None:
    # Arrange — olive_docs evidence pinned to a release outside the window
    data = _minimal_valid_matrix(
        evidence_type="olive_docs",
        evidence_version="0.9.0",
        window_min="0.12.0",
        window_max="0.12.1",
    )
    # Act
    errors = collect_matrix_errors(data, pass_names, enforce_olive_window=True)
    # Assert
    assert any("outside olive_version_support" in e for e in errors), errors


def test_ort_docs_evidence_not_bound_to_olive_window(pass_names: set[str]) -> None:
    # Arrange — ORT version strings must not be checked against Olive window
    data = _minimal_valid_matrix(
        evidence_type="ort_docs",
        evidence_version="1.19",
        window_min="0.12.0",
        window_max="0.12.1",
    )
    # Act
    errors = collect_matrix_errors(data, pass_names, enforce_olive_window=True)
    # Assert
    assert errors == [], errors


def test_rejects_missing_olive_pass_field(pass_names: set[str]) -> None:
    # Arrange
    data = _minimal_valid_matrix()
    claim = data["models"][0]["hardware"]["NVIDIA RTX 4090"]["OnnxConversion"]
    del claim["olive_pass"]
    # Act
    errors = collect_matrix_errors(data, pass_names)
    # Assert
    assert any("missing required field 'olive_pass'" in e for e in errors), errors


def test_rejects_invalid_evidence_type(pass_names: set[str]) -> None:
    # Arrange
    data = _minimal_valid_matrix(evidence_type="blog_post")
    # Act
    errors = collect_matrix_errors(data, pass_names)
    # Assert
    assert any("invalid evidence.type" in e for e in errors), errors


def test_rejects_missing_top_level_models(pass_names: set[str]) -> None:
    # Arrange
    data = {
        "version": "0.0.0",
        "last_updated": "2026-08-05",
    }
    # Act
    errors = collect_matrix_errors(data, pass_names)
    # Assert
    assert any("missing top-level field: models" in e for e in errors), errors


def test_version_helpers_reject_garbage() -> None:
    # Arrange / Act / Assert — pure unit edges for helpers
    with pytest.raises(ValueError):
        _parse_version("")
    with pytest.raises(ValueError):
        _parse_version("latest")
    assert _version_in_range("0.12.1", "0.12.0", "0.12.1") is True
    assert _version_in_range("0.11.0", "0.12.0", "0.12.1") is False
