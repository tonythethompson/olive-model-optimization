"""Shared helpers for Olive MCP tools.

Heavy tool modules (docs search, etc.) are imported lazily via ``__getattr__``
so the HTTP diagnosis path can load ``troubleshooting`` without requiring
BeautifulSoup / requests in the project .venv.
"""

from __future__ import annotations

import importlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

KB_DIR = Path(__file__).parent.parent / "knowledge_base"

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "get_cli_command": (".cli_helper", "get_cli_command"),
    "get_model_compatibility": (".compatibility", "get_model_compatibility"),
    "get_pass_config_template": (".config_generator", "get_pass_config_template"),
    "get_data_config_template": (".data_config", "get_data_config_template"),
    "search_olive_documentation": (".docs_search", "search_olive_documentation"),
    "get_hardware_optimization_guide": (".hardware_guide", "get_hardware_optimization_guide"),
    "get_integration_recipe": (".integration_recipes", "get_integration_recipe"),
    "get_olive_passes": (".pass_catalog", "get_olive_passes"),
    "get_pass_chain": (".pass_chain", "get_pass_chain"),
    "get_pass_parameters": (".pass_parameters", "get_pass_parameters"),
    "get_quantization_strategy": (".strategy_advisor", "get_quantization_strategy"),
    "evaluate_optimization_tradeoff": (".tradeoff", "evaluate_optimization_tradeoff"),
    "troubleshoot_olive_error": (".troubleshooting", "troubleshoot_olive_error"),
    "diagnose_error": (".troubleshooting", "diagnose_error"),
    "get_error_frequency_summary": (".troubleshooting", "get_error_frequency_summary"),
    "get_context_for_pipeline": (".passive_context", "get_context_for_pipeline"),
    "validate_ui_state_recipe": (".studio_recipe", "validate_ui_state_recipe"),
    "get_recipe_for_ui_state": (".studio_recipe", "get_recipe_for_ui_state"),
    "get_runtime_ep_hints": (".runtime_ep_hints", "get_runtime_ep_hints"),
    "record_troubleshoot_feedback": (".feedback", "record_troubleshoot_feedback"),
    "plan_optimization": (".agent_planner", "plan_optimization"),
}


def load_json(name: str) -> dict[str, Any]:
    """
    Load and parse a UTF-8 JSON file from the knowledge base.

    Parameters:
        name (str): Name of the JSON file within the knowledge base directory.

    Returns:
        dict[str, Any]: Parsed JSON object.
    """
    path = KB_DIR / name
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_passes() -> list[dict[str, Any]]:
    return load_json("passes.json").get("passes", [])


def load_hardware_profiles() -> list[dict[str, Any]]:
    return load_json("hardware_profiles.json").get("profiles", [])


def load_quirks() -> dict[str, Any]:
    return load_json("quirks.json").get("categories", {})


@lru_cache(maxsize=1)
def _cached_troubleshooting() -> tuple[dict[str, Any], ...]:
    entries = load_json("troubleshooting.json").get("entries", [])
    return tuple(_normalize_entry(e, default_domain="olive") for e in entries)


def load_troubleshooting() -> list[dict[str, Any]]:
    """Load and normalize Olive troubleshooting entries (cached; returns a list copy).

    Returns:
        list[dict[str, Any]]: Troubleshooting entries with normalized domains and applyability values.
    """
    return list(_cached_troubleshooting())


@lru_cache(maxsize=1)
def _cached_studio_troubleshooting() -> tuple[dict[str, Any], ...]:
    try:
        entries = load_json("studio_troubleshooting.json").get("entries", [])
    except FileNotFoundError:
        return ()
    return tuple(_normalize_entry(e, default_domain="studio") for e in entries)


def load_studio_troubleshooting() -> list[dict[str, Any]]:
    """
    Load and normalize studio troubleshooting entries (cached; returns a list copy).

    Returns:
        list[dict[str, Any]]: The normalized troubleshooting entries, or an empty
        list when the data file is unavailable.
    """
    return list(_cached_studio_troubleshooting())


def _normalize_entry(entry: dict[str, Any], default_domain: str) -> dict[str, Any]:
    """
    Normalize a troubleshooting entry for diagnostic routing.

    Parameters:
        entry (dict[str, Any]): Entry to copy and normalize.
        default_domain (str): Domain used when the entry's domain is missing or invalid.

    Returns:
        dict[str, Any]: A copied entry with a valid domain and boolean ``applyable`` value.
    """
    out = dict(entry)
    domain = out.get("domain") or default_domain
    out["domain"] = domain if domain in ("olive", "studio") else default_domain
    if "applyable" not in out:
        cfg = out.get("updated_config")
        out["applyable"] = isinstance(cfg, dict) and len(cfg) > 0
    else:
        out["applyable"] = bool(out["applyable"])
    return out


def load_compatibility_matrix() -> list[dict[str, Any]]:
    """Load model compatibility entries from the compatibility matrix.

    Returns:
        list[dict[str, Any]]: The model compatibility entries.
    """
    return load_json("compatibility_matrix.json").get("models", [])


def load_integration_recipes() -> list[dict[str, Any]]:
    """Load the integration recipes from the knowledge base.

    Returns:
        list[dict[str, Any]]: The available integration recipes.
    """
    return load_json("integration_recipes.json").get("recipes", [])


def __getattr__(name: str) -> Any:
    """
    Resolve a lazily exported tool or raise an attribute error for unknown names.

    Parameters:
        name (str): Name of the exported attribute to resolve.

    Returns:
        Any: The imported attribute associated with the requested name.

    Raises:
        AttributeError: If the name is not a lazily exported attribute.
    """
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr = target
    module = importlib.import_module(module_name, __name__)
    value = getattr(module, attr)
    globals()[name] = value
    return value


__all__ = [
    "get_cli_command",
    "get_model_compatibility",
    "get_pass_config_template",
    "get_data_config_template",
    "search_olive_documentation",
    "get_hardware_optimization_guide",
    "get_integration_recipe",
    "get_olive_passes",
    "get_pass_chain",
    "get_pass_parameters",
    "get_quantization_strategy",
    "evaluate_optimization_tradeoff",
    "troubleshoot_olive_error",
    "diagnose_error",
    "get_error_frequency_summary",
    "get_context_for_pipeline",
    "validate_ui_state_recipe",
    "get_recipe_for_ui_state",
    "get_runtime_ep_hints",
    "record_troubleshoot_feedback",
    "plan_optimization",
    "KB_DIR",
    "load_json",
    "load_passes",
    "load_hardware_profiles",
    "load_quirks",
    "load_troubleshooting",
    "load_studio_troubleshooting",
    "load_compatibility_matrix",
    "load_integration_recipes",
]
