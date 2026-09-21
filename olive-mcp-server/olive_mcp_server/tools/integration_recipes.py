"""Tool: get_integration_recipe.

Provides ready-to-run Olive recipe templates for common model + hardware
combinations, plus a filterable catalog view.
"""

from typing import Any

from . import load_integration_recipes


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
                return {
                    "recipe_id": recipe["id"],
                    "name": recipe["name"],
                    "description": recipe["description"],
                    "recipe": recipe["recipe"],
                    "notes": recipe.get("notes", ""),
                }
        return {"error": f"Recipe '{recipe_id}' not found."}

    filtered = recipes
    if model_type:
        filtered = [r for r in filtered if _matches_filter(model_type, r.get("model_type", []))]
    if target_hardware:
        filtered = [r for r in filtered if _matches_filter(target_hardware, r.get("target_hardware", []))]
    if source_format:
        query = source_format.lower()
        filtered = [r for r in filtered if r.get("source_format", "").lower() == query]

    return {
        "recipes": [_recipe_summary(r) for r in filtered],
        "count": len(filtered),
    }
