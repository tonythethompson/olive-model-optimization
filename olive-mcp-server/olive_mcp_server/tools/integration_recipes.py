"""Tool: get_integration_recipe.

Provides ready-to-run Olive recipe templates for common model + hardware
combinations, plus a filterable catalog view.
"""

from copy import deepcopy
from typing import Any

from . import load_integration_recipes

OLIVE_SCHEMA_URL = "https://microsoft.github.io/Olive/0.13.0/schema.json"

# The bundled catalog deliberately retains its historical 0.13-era source
# representation. Recipes returned by this tool use the Olive 0.13 ModelConfig
# shape: ``type`` plus a required nested ``config`` object. The only historical
# model-handler spelling we normalize is Hugging Face's.
_MODEL_TYPE_ALIASES = {
    "HuggingfaceModel": "HfModel",
}


def _normalize_input_model(recipe: dict[str, Any]) -> None:
    """Normalize a historical input model to Olive 0.13 ``ModelConfig``."""
    input_model = recipe.get("input_model")
    if not isinstance(input_model, dict):
        return

    model_config = input_model.get("config")
    if not isinstance(model_config, dict):
        model_config = {}
        input_model["config"] = model_config

    # Recover any fields produced by the prior flat-template implementation.
    # ``type`` and ``config`` belong to ModelConfig; model-specific values such
    # as model_path and task belong inside the required config object.
    for key in list(input_model):
        if key not in {"type", "config"}:
            model_config.setdefault(key, input_model.pop(key))

    model_type = input_model.get("type")
    if isinstance(model_type, str):
        input_model["type"] = _MODEL_TYPE_ALIASES.get(model_type, model_type)


def _current_schema_recipe(recipe: dict[str, Any]) -> dict[str, Any]:
    """Normalize legacy catalog recipes to the current Olive schema shape.

    The catalog remains a historical reference source, but returned recipes
    are migrated to the validated Olive 0.13 model/pass/data shapes. Models
    use the required ``input_model.type`` plus ``input_model.config`` form,
    passes are arrays of ``RunPassConfig`` objects, and data components use
    ``type``/``params``.
    """
    out = deepcopy(recipe)
    _normalize_input_model(out)
    data_configs = out.get("data_configs")
    if isinstance(data_configs, list):
        for data_config in data_configs:
            if not isinstance(data_config, dict):
                continue
            legacy_params = data_config.pop("params_config", None)
            if legacy_params is not None:
                component_type = data_config.get("type", "DataContainer")
                data_config["type"] = "DataContainer"
                data_config["load_dataset_config"] = {
                    "type": component_type,
                    "params": legacy_params,
                }
            for component_name in (
                "load_dataset_config",
                "pre_process_data_config",
                "post_process_data_config",
                "dataloader_config",
            ):
                component = data_config.get(component_name)
                if isinstance(component, dict) and "params" not in component:
                    data_config[component_name] = {"params": component}

    passes = out.get("passes")
    if isinstance(passes, dict):
        normalized: dict[str, list[dict[str, Any]]] = {}
        for pass_id, pass_config in passes.items():
            entries = pass_config if isinstance(pass_config, list) else [pass_config]
            current_entries: list[dict[str, Any]] = []
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                current = dict(entry)
                if "params" in current and "config" not in current:
                    current["config"] = current.pop("params")
                current.setdefault("type", pass_id)
                current_entries.append(current)
            normalized[pass_id] = current_entries
        out["passes"] = normalized

    out["schema_source"] = OLIVE_SCHEMA_URL
    return out


def _matches_filter(value: str, candidates: list[str]) -> bool:
    """Return True when value is a case-insensitive substring of any candidate."""
    query = value.lower()
    return any(query in candidate.lower() for candidate in candidates)


def _recipe_summary(recipe: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": recipe["id"],
        "name": recipe["name"],
        "description": recipe["description"],
        "model_type": recipe.get("model_type", []),
        "target_hardware": recipe.get("target_hardware", []),
        "source_format": recipe.get("source_format", ""),
        "passes": recipe.get("passes", []),
        "notes": recipe.get("notes", ""),
    }


def get_integration_recipe(
    recipe_id: str = "",
    model_type: str = "",
    target_hardware: str = "",
    source_format: str = "",
) -> dict[str, Any]:
    """Return a full Olive recipe template or a filtered list of recipe summaries.

    Args:
        recipe_id: Exact recipe ID to retrieve; if omitted, a catalog is returned.
        model_type: Optional filter (case-insensitive substring of model_type tags).
        target_hardware: Optional filter (case-insensitive substring of target hardware list).
        source_format: Optional filter (e.g. PyTorch, HuggingFace, ONNX).

    Returns:
        Full recipe when recipe_id is given, otherwise a list of summaries.
    """
    recipes = load_integration_recipes()

    if recipe_id:
        query = recipe_id.lower()
        for recipe in recipes:
            if recipe["id"].lower() == query:
                normalized_recipe = _current_schema_recipe(recipe["recipe"])
                return {
                    "recipe_id": recipe["id"],
                    "name": recipe["name"],
                    "description": recipe["description"],
                    "recipe": normalized_recipe,
                    "notes": recipe.get("notes", ""),
                    "schema_source": OLIVE_SCHEMA_URL,
                }
        return {"error": f"Recipe '{recipe_id}' not found."}

    filtered = recipes
    if model_type:
        filtered = [
            r for r in filtered if _matches_filter(model_type, r.get("model_type", []))
        ]
    if target_hardware:
        filtered = [
            r
            for r in filtered
            if _matches_filter(target_hardware, r.get("target_hardware", []))
        ]
    if source_format:
        query = source_format.lower()
        filtered = [r for r in filtered if r.get("source_format", "").lower() == query]

    return {
        "recipes": [_recipe_summary(r) for r in filtered],
        "count": len(filtered),
    }
