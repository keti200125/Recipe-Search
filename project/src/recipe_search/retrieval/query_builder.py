from __future__ import annotations
from dataclasses import dataclass
from typing import List, Set

from recipe_search.preprocessing.ingredient_normalizer import IngredientNormalizer


@dataclass(frozen=True)
class BuiltQuery:
    query_tokens: List[str]  
    query_set: Set[str] 


class QueryBuilder:
    def __init__(self, normalizer: IngredientNormalizer) -> None:
        self.normalizer = normalizer

    def build(self, available_items: list[str]) -> BuiltQuery:
        tokens = self.normalizer.normalize_many(available_items)

        seen = set()
        uniq: List[str] = []
        for t in tokens:
            if t not in seen:
                uniq.append(t)
                seen.add(t)

        return BuiltQuery(
            query_tokens=uniq,
            query_set=set(uniq),
        )
