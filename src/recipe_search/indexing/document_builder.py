from __future__ import annotations
from recipe_search.io.models import RecipeRecord, Document
from recipe_search.preprocessing.ingredient_source import IngredientSourceSelector


class DocumentBuilder:
    def __init__(self, source_selector: IngredientSourceSelector) -> None:
        self.source_selector = source_selector

    def build(self, rec: RecipeRecord) -> Document:

        canon = self.source_selector.canonical_ingredients(rec)
        
        text = " ".join(canon).lower().strip()


        return Document(
            doc_id=rec.recipe_id,
            title=rec.title,
            canon_ingredients=canon,
            text=text,
            source=rec.source,
            link=rec.link,
        )

    def build_many(self, records: list[RecipeRecord]) -> list[Document]:
        return [self.build(r) for r in records]
