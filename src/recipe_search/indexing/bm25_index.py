from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional

from rank_bm25 import BM25Okapi


@dataclass(frozen=True)
class BM25Result:
    doc_id: int
    score: float


class BM25Index:
    def __init__(self) -> None:
        self._bm25: Optional[BM25Okapi] = None
        self._doc_ids: List[int] = []
        self._corpus_tokens: List[List[str]] = []

    def fit(self, documents: list) -> None:
        self._doc_ids = [d.doc_id for d in documents]
        self._corpus_tokens = [d.canon_ingredients for d in documents]

        self._bm25 = BM25Okapi(
            self._corpus_tokens,
            k1=1.2, 
            # b=0.4,
            b=0.95,
            epsilon=0.25
        )

    def search(self, query_tokens: List[str], top_k: int = 10) -> list[BM25Result]:
        if self._bm25 is None:
            raise RuntimeError("BM25Index is not fitted.")

        scores = self._bm25.get_scores(query_tokens)
        ranked_idx = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)
        results = [
            BM25Result(doc_id=self._doc_ids[i], score=float(scores[i]))
            for i in ranked_idx if scores[i] > 0.0][:top_k]
        return results

