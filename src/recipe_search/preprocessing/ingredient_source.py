from __future__ import annotations
from recipe_search.io.models import RecipeRecord
from recipe_search.preprocessing.ingredient_normalizer import IngredientNormalizer


class IngredientSourceSelector:
    """Prefer NER; fallback to raw ingredients normalization."""
    def __init__(self, normalizer: IngredientNormalizer) -> None:
        self.normalizer = normalizer

    def canonical_ingredients(self, rec: RecipeRecord) -> list[str]:
        if rec.ner:
            return [str(x).strip().lower() for x in rec.ner if str(x).strip()]
        return self.normalizer.normalize_many(rec.ingredients)
