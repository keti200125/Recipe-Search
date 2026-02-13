import re
from typing import Iterable, List, Optional
import inflect


class IngredientNormalizer:
    def __init__(self) -> None:
        self.inflect = inflect.engine()

    def normalize(self, ingredient: Optional[str]) -> str:
        ingredient = (ingredient or "").lower().strip()
        if not ingredient:
            return ""

        ingredient = BRACKETS.sub(" ", ingredient)
        ingredient = PERCENT.sub(" ", ingredient)
        ingredient = MIXED_FRAC.sub(" ", ingredient)
        ingredient = FRAC.sub(" ", ingredient)
        ingredient = RANGE.sub(" ", ingredient)
        ingredient = NUM.sub(" ", ingredient)
        ingredient = PUNCT.sub(" ", ingredient)

        tokens = ingredient.split()
        tokens = [
            t for t in tokens
            if t not in UNITS and t not in DESCRIPTORS and t not in STOPWORDS
        ]
        tokens = [
            self.inflect.singular_noun(t) or t  # type: ignore[arg-typ]
            for t in tokens
        ]

        return " ".join(tokens).strip()

    def normalize_many(self, ingredients: Iterable[Optional[str]]) -> List[str]:
        out_ingr: List[str] = []
        for ingredient in ingredients:
            norm = self.normalize(ingredient)
            if norm:
                out_ingr.append(norm)
        return out_ingr


# CONSTANTS

UNITS = {
    "lb", "lbs", "pound", "pounds",
    "c", "cup", "cups",
    "tsp", "teaspoon", "teaspoons",
    "tbsp", "tablespoon", "tablespoons",
    "oz", "ounce", "ounces",
    "g", "gram", "grams",
    "kg", "ml", "l", "pinch",
}

DESCRIPTORS = {
            "chopped", "minced", "diced", "sliced", "fresh",
            "optional", "to", "taste", "large", "small",
            "ground", "crushed",
}

STOPWORDS = {
    "of", "or", "and", "with", "without", "for",
}

BRACKETS = re.compile(r"\([^)]*\)")
PERCENT = re.compile(r"\b\d+(\.\d+)?%\b")
MIXED_FRAC = re.compile(r"\b\d+\s+\d+/\d+\b")
FRAC = re.compile(r"\b\d+/\d+\b")
RANGE = re.compile(r"\b\d+\s*-\s*\d+\b")
NUM = re.compile(r"\b\d+(\.\d+)?\b")
PUNCT = re.compile(r"[^a-zA-Z\s-]")

