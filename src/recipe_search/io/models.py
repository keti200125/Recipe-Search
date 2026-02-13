from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List


@dataclass(frozen=True)
class RecipeRecord:
    recipe_id: int
    title: str
    ingredients: List[str]
    ner: List[str]
    source: Optional[str] = None
    link: Optional[str] = None


@dataclass(frozen=True)
class Document:
    doc_id: int
    title: str
    canon_ingredients: List[str]
    text: str
    source: Optional[str] = None
    link: Optional[str] = None
