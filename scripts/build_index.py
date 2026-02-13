import os
import pickle

from recipe_search.io.dataset_loader import DatasetLoader
from recipe_search.preprocessing.ingredient_normalizer import IngredientNormalizer
from recipe_search.preprocessing.ingredient_source import IngredientSourceSelector
from recipe_search.indexing.document_builder import DocumentBuilder
from recipe_search.indexing.bm25_index import BM25Index
from recipe_search.indexing.tfidf_index import TFIDFIndex

DATA_PATH = "data/raw/recipenlg.csv"
ART_DIR = "data/artifacts"
SAMPLE_N = 40_000

def main():
    os.makedirs(ART_DIR, exist_ok=True)

    loader = DatasetLoader(random_sample=SAMPLE_N, seed=42)
    df = loader.load_csv(DATA_PATH)
    records = loader.to_records(df)

    normalizer = IngredientNormalizer()
    selector = IngredientSourceSelector(normalizer)
    builder = DocumentBuilder(selector)
    docs = builder.build_many(records)

    bm25 = BM25Index()
    bm25.fit(docs)

    tfidf = TFIDFIndex()
    tfidf.fit(docs)

    documents_by_id = {
        d.doc_id: {
            "title": d.title,
            "ingredients": d.canon_ingredients,
            "link": d.link,
            "source": d.source,
        }
        for d in docs
    }

    with open(os.path.join(ART_DIR, "documents.pkl"), "wb") as f:
        pickle.dump(documents_by_id, f)

    with open(os.path.join(ART_DIR, "bm25.pkl"), "wb") as f:
        pickle.dump(bm25, f)

    with open(os.path.join(ART_DIR, "tfidf.pkl"), "wb") as f:
        pickle.dump(tfidf, f)

    print(f"Saved artifacts in: {ART_DIR}")

if __name__ == "__main__":
    main()
