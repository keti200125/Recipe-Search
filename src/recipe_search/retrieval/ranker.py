from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Set, Dict, Any, List


@dataclass(frozen=True)
class RerankedResult:
    doc_id: int
    base_score: float
    final_score: float
    matched: int
    missing: int
    ingredient_coverage: float 
    query_coverage: float
    f1_overlap: float


class IngredientOverlapRanker:

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha

    def rerank(
        self,
        results: Iterable[Any],
        documents_by_id: Dict[int, Dict[str, Any]],
        query_set: Set[str],
        top_k: int | None = None,
    ) -> List[RerankedResult]:
        qn = max(len(query_set), 1)
        out: List[RerankedResult] = []

        for r in results:
            doc = documents_by_id.get(r.doc_id)
            if not doc:
                continue

            doc_ings = set(doc.get("ingredients", []))
            dn = max(len(doc_ings), 1)

            matched = len(doc_ings & query_set)
            missing = len(doc_ings - query_set)

            ingredient_coverage = matched / dn
            query_coverage = matched / qn

            if ingredient_coverage + query_coverage == 0:
                f1_overlap = 0.0
            else:
                f1_overlap = (
                    2 * ingredient_coverage * query_coverage
                    / (ingredient_coverage + query_coverage)
                )

            final_score = float(r.score) * (f1_overlap ** self.alpha)

            out.append(
                RerankedResult(
                    doc_id=r.doc_id,
                    base_score=float(r.score),
                    final_score=final_score,
                    matched=matched,
                    missing=missing,
                    ingredient_coverage=ingredient_coverage,
                    query_coverage=query_coverage,
                    f1_overlap=f1_overlap,
                )
            )

        out.sort(key=lambda x: x.final_score, reverse=True)
        return out[:top_k] if top_k is not None else out
