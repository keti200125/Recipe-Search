import pickle
from recipe_search.preprocessing.ingredient_normalizer import IngredientNormalizer
from recipe_search.retrieval.query_builder import QueryBuilder


def evaluate(index, documents_by_id, K=10, max_queries=500):
    normalizer = IngredientNormalizer()
    qb = QueryBuilder(normalizer)

    hits = 0
    total = 0

    for doc_id, doc in list(documents_by_id.items())[:max_queries]:
        query_tokens = doc["ingredients"]
        built = qb.build(query_tokens)

        results = index.search(built.query_tokens, top_k=K)
        retrieved_ids = [r.doc_id for r in results]

        if doc_id in retrieved_ids:
            hits += 1
        total += 1

    recall_at_k = hits / total
    precision_at_k = recall_at_k / K

    return precision_at_k, recall_at_k


if __name__ == "__main__":
    with open("data/artifacts/documents.pkl", "rb") as f:
        documents_by_id = pickle.load(f)

    with open("data/artifacts/bm25.pkl", "rb") as f:
        bm25 = pickle.load(f)

    with open("data/artifacts/tfidf.pkl", "rb") as f:
        tfidf = pickle.load(f)

    for name, index in [("BM25", bm25), ("TF-IDF", tfidf)]:
        p, r = evaluate(index, documents_by_id, K=10)
        print(f"{name} | Precision@10: {p:.4f} | Recall@10: {r:.4f}")
