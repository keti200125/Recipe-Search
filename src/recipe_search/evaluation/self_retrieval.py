from __future__ import annotations
from typing import Dict, Any, Tuple, Iterable

from recipe_search.preprocessing.ingredient_normalizer import IngredientNormalizer
from recipe_search.retrieval.query_builder import QueryBuilder


def evaluate_self_retrieval(
    index: Any,
    documents_by_id: Dict[int, Dict[str, Any]],
    k: int = 10,
    max_queries: int = 500,
) -> Tuple[float, float]:

    normalizer = IngredientNormalizer()
    qb = QueryBuilder(normalizer)

    hits = 0
    total = 0

    for doc_id, doc in list(documents_by_id.items())[:max_queries]:
        query_tokens = doc.get("ingredients", [])
        built = qb.build(query_tokens)

        results = index.search(built.query_tokens, top_k=k)
        retrieved_ids = [r.doc_id for r in results]

        if doc_id in retrieved_ids:
            hits += 1
        total += 1

    recall_at_k = hits / max(total, 1)
    precision_at_k = recall_at_k / k 
    return precision_at_k, recall_at_k
