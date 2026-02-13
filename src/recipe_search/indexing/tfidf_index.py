from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


@dataclass(frozen=True)
class TFIDFResult:
    doc_id: int
    score: float


class TFIDFIndex:
    def __init__(self) -> None:
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.doc_matrix = None
        self._doc_ids: List[int] = []

    def identity_analyzer(self, x):
        return x

    def fit(self, documents: list) -> None:
        self._doc_ids = [d.doc_id for d in documents]
        corpus = [d.canon_ingredients for d in documents]

        self.vectorizer = TfidfVectorizer(
            analyzer=self.identity_analyzer,
            lowercase=False,
            min_df=1,
        )
        self.doc_matrix = self.vectorizer.fit_transform(corpus)

    def search(self, query_tokens: List[str], top_k: int = 10) -> list[TFIDFResult]:
        if self.vectorizer is None or self.doc_matrix is None:
            raise RuntimeError("TFIDFIndex is not fitted.")

        q_vec = self.vectorizer.transform([query_tokens])
        scores = linear_kernel(q_vec, self.doc_matrix).ravel()

        ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [TFIDFResult(doc_id=self._doc_ids[i], score=float(scores[i])) for i in ranked_idx  if scores[i] > 0.0]

