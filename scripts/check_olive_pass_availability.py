#!/usr/bin/env python3
"""CI helper: compare compatibility_matrix claims to the installed Olive pass registry.

Installs nothing and runs no optimization. Requires olive-ai already installed.
Enumerates pass names from Olive's package config / registry only — no model
download, no pass execution, no CUDA.

Exit codes:
  0 — every matrix supported/warning olive_pass is present in the registry
  1 — missing claimed passes, or registry/matrix could not be loaded
"""

from __future__ import annotations

import importlib
import json
import logging
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

_LOG = logging.getLogger(__name__)

# Claimed support levels that must exist in the pinned Olive registry.
_REQUIRED_SUPPORT = frozenset({"supported", "warning"})

# Passes documented in the matrix under legacy or workflow names that differ from
# olive_config.json keys in Olive 0.12.x (enumeration is case-insensitive).
_PASS_REGISTRY_ALIASES: dict[str, tuple[str, ...]] = {
    "qnnquantization": ("qnnpreprocess", "qnnconversion", "onnxquantization", "onnxstaticquantization"),
    "onnxmodeloptimizer": ("onnxpeepholeoptimizer",),
    # Olive 0.13 exposes QAIRT through QairtPipelinePass and the remaining
    # entries as GraphSurgeries, not as separate registered pass names.
    "qairtpipeline": ("qairtpipelinepass",),
    "quantizeembeddingint8": ("graphsurgeries",),
    "shareembeddinglmhead": ("graphsurgeries",),
    "simplifiedlayernormtormsnorm": ("graphsurgeries",),
}

# Cloud workflow passes are valid matrix claims but are not listed in local olive_config.json.
_CLOUD_ONLY_PASSES = frozenset({"azuremlquantization"})

_SCRIPT_DIR = Path(__file__).resolve().parent
_MCP_ROOT = _SCRIPT_DIR.parent
_DEFAULT_MATRIX = _MCP_ROOT / "olive_mcp_server" / "knowledge_base" / "compatibility_matrix.json"


def _load_matrix(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"compatibility matrix not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("compatibility_matrix.json root must be an object")
    return data


def claimed_passes(matrix: dict[str, Any]) -> dict[str, set[str]]:
    """Map olive_pass -> set of support levels seen for supported/warning claims."""
    models = matrix.get("models")
    if not isinstance(models, list):
        raise ValueError("compatibility_matrix.json missing models[]")

    claimed: dict[str, set[str]] = {}
    for model in models:
        if not isinstance(model, dict):
            continue
        hardware = model.get("hardware")
        if not isinstance(hardware, dict):
            continue
        for _hw_name, passes in hardware.items():
            if not isinstance(passes, dict):
                continue
            for _key, claim in passes.items():
                if not isinstance(claim, dict):
                    continue
                support = claim.get("support")
                if support not in _REQUIRED_SUPPORT:
                    continue
                olive_pass = claim.get("olive_pass")
                if not isinstance(olive_pass, str) or not olive_pass.strip():
                    # Fall back to claim key only when olive_pass is absent
                    # (legacy rows); prefer explicit olive_pass when present.
                    if isinstance(_key, str) and _key.strip():
                        olive_pass = _key
                    else:
                        continue
                name = olive_pass.strip()
                claimed.setdefault(name, set()).add(str(support))
    return claimed


def _olive_config_json_path() -> Path | None:
    """Locate olive_config.json without importing olive (avoids heavy transitive deps on CI)."""
    try:
        from importlib.metadata import files

        cfg = files("olive").joinpath("olive_config.json")
        path = Path(str(cfg))
        if path.is_file():
            return path
    except Exception as exc:  # noqa: BLE001 — fall through to site-packages walk
        _LOG.debug("importlib.metadata olive_config lookup failed: %s", exc)

    try:
        import site

        search_roots = [*site.getsitepackages(), site.getusersitepackages()]
        for root in search_roots:
            candidate = Path(root) / "olive" / "olive_config.json"
            if candidate.is_file():
                return candidate
    except Exception as exc:  # noqa: BLE001
        _LOG.debug("site-packages olive_config walk failed: %s", exc)
        return None
    return None


def _names_from_installed_config_json() -> set[str] | None:
    """Read pass keys from the installed olive_config.json (Olive 0.12.x default)."""
    path = _olive_config_json_path()
    if path is None:
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return _pass_keys_from_mapping(raw.get("passes") if isinstance(raw, dict) else None)


def _names_from_pass_registry_class() -> set[str] | None:
    """Newer olive-ai: olive.passes.PassRegistry (scripts/sync-pass-catalog.mjs)."""
    try:
        from olive.passes import PassRegistry  # type: ignore
    except ImportError:
        return None

    try:
        registry = PassRegistry()
    except Exception:  # noqa: BLE001 — try next strategy
        return None

    if hasattr(registry, "get_all_passes"):
        try:
            names = registry.get_all_passes()
        except Exception:  # noqa: BLE001
            return None
        if isinstance(names, dict):
            return {str(k) for k in names}
        if isinstance(names, (set, list, tuple)):
            out = {str(x) for x in names}
            return out or None
    return None


def _names_from_pass_registry_module() -> set[str] | None:
    """Olive 0.12.x: REGISTRY alias on olive.passes (Pass.registry)."""
    try:
        from olive.passes import REGISTRY  # type: ignore
    except ImportError:
        return None
    if isinstance(REGISTRY, dict) and REGISTRY:
        return {str(k) for k in REGISTRY}
    return None


def _names_from_package_config() -> set[str] | None:
    """Fallback: OlivePackageConfig default pass table (olive_config.json)."""
    try:
        from olive.package_config import OlivePackageConfig  # type: ignore
    except ImportError:
        return None

    cfg = None
    if hasattr(OlivePackageConfig, "load_default_config"):
        try:
            cfg = OlivePackageConfig.load_default_config()
        except Exception:  # noqa: BLE001
            cfg = None
    if cfg is None and hasattr(OlivePackageConfig, "get_default_config_path"):
        try:
            path = Path(OlivePackageConfig.get_default_config_path())
            if path.is_file():
                raw = json.loads(path.read_text(encoding="utf-8"))
                return _pass_keys_from_mapping(raw.get("passes"))
        except Exception:  # noqa: BLE001
            return None
    if cfg is None:
        return None

    passes = getattr(cfg, "passes", None)
    if isinstance(passes, dict):
        return {str(k) for k in passes}
    return None


def _pass_keys_from_mapping(passes: Any) -> set[str] | None:
    if not isinstance(passes, dict) or not passes:
        return None
    return {str(k) for k in passes}


def _names_from_olive_config_json() -> set[str] | None:
    """Fallback: read olive_config.json shipped beside the olive package."""
    try:
        import olive  # type: ignore
    except ImportError:
        return None

    olive_root = Path(olive.__file__).resolve().parent
    candidates = [
        olive_root / "olive_config.json",
        olive_root / "package_config.json",
    ]
    for path in candidates:
        if not path.is_file():
            continue
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        names = _pass_keys_from_mapping(raw.get("passes") if isinstance(raw, dict) else None)
        if names:
            return names
    return None


def _names_from_registry_attrs() -> set[str] | None:
    """Last resort: introspect REGISTRY-like module attributes."""
    module_candidates = (
        "olive.passes.olive_pass",
        "olive.passes.pass_config",
        "olive.passes",
    )
    attr_candidates = ("REGISTRY", "registry", "PASS_REGISTRY", "_REGISTRY")

    for mod_name in module_candidates:
        try:
            mod = importlib.import_module(mod_name)
        except ImportError:
            continue
        for attr in attr_candidates:
            reg = getattr(mod, attr, None)
            if isinstance(reg, dict) and reg:
                return {str(k) for k in reg}
            if isinstance(reg, (set, list, tuple)) and reg:
                return {str(x) for x in reg}
    return None


def enumerate_olive_pass_names() -> set[str]:
    """Return pass type names available in the installed Olive package."""
    for loader in (
        _names_from_installed_config_json,
        _names_from_pass_registry_class,
        _names_from_pass_registry_module,
        _names_from_package_config,
        _names_from_olive_config_json,
        _names_from_registry_attrs,
    ):
        names = loader()
        if names:
            return names
    raise RuntimeError(
        "Could not enumerate Olive passes. Is olive-ai installed, and does this "
        "version expose PassRegistry / OlivePackageConfig / olive_config.json?"
    )


def _claim_in_registry(claimed_name: str, available_lower: set[str]) -> bool:
    """Return True when a matrix olive_pass is present or has a known alias."""
    lowered = claimed_name.lower()
    if lowered in _CLOUD_ONLY_PASSES:
        return True
    # Check registry first — if Olive exposes the pass, trust the real probe.
    if lowered in available_lower:
        return True
    return any(alias in available_lower for alias in _PASS_REGISTRY_ALIASES.get(lowered, ()))


def olive_version_string() -> str:
    try:
        import olive  # type: ignore

        ver = getattr(olive, "__version__", None)
        if isinstance(ver, str) and ver:
            return ver
    except ImportError:
        pass
    try:
        from importlib.metadata import version

        return version("olive-ai")
    except Exception:  # noqa: BLE001 — best-effort label only
        return "unknown"


def main(argv: Iterable[str] | None = None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    matrix_path = Path(args[0]) if args else _DEFAULT_MATRIX

    try:
        matrix = _load_matrix(matrix_path)
        claimed = claimed_passes(matrix)
        available = enumerate_olive_pass_names()
    except Exception as exc:  # noqa: BLE001 — surface load errors as CI failure
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    ver = olive_version_string()
    print(f"Olive version: {ver}")
    print(f"Registry passes: {len(available)}")
    print(f"Matrix supported/warning olive_pass claims: {len(claimed)}")
    print(f"Matrix path: {matrix_path}")

    available_lower = {name.lower() for name in available}
    missing = sorted(name for name in claimed if not _claim_in_registry(name, available_lower))
    if missing:
        print(
            "\nFAIL: claimed supported/warning pass(es) not in Olive registry:",
            file=sys.stderr,
        )
        for name in missing:
            levels = ", ".join(sorted(claimed[name]))
            print(f"  - {name} (matrix support: {levels})", file=sys.stderr)
        print(
            "\nThese names must exist in the pinned Olive pass registry (enumeration only — no optimization was run).",
            file=sys.stderr,
        )
        return 1

    print(
        "OK: all matrix supported/warning olive_pass claims were recognized by "
        "the Olive registry, alias mapping, or cloud-only allowlist."
    )
    # Helpful sample for CI logs (sorted, capped).
    sample = ", ".join(sorted(claimed)[:12])
    if len(claimed) > 12:
        sample += ", ..."
    print(f"Checked: {sample}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
