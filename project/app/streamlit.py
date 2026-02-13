import pickle
import streamlit as st

from recipe_search.preprocessing.ingredient_normalizer import IngredientNormalizer
from recipe_search.retrieval.query_builder import QueryBuilder
from recipe_search.evaluation.self_retrieval import evaluate_self_retrieval


st.set_page_config(page_title="Recipe Search", layout="wide")

@st.cache_resource
def load_artifacts():
    with open("data/artifacts/documents.pkl", "rb") as f:
        docs = pickle.load(f)
    with open("data/artifacts/bm25.pkl", "rb") as f:
        bm25 = pickle.load(f)
    with open("data/artifacts/tfidf.pkl", "rb") as f:
        tfidf = pickle.load(f)
    return docs, bm25, tfidf

documents_by_id, bm25, tfidf = load_artifacts()

normalizer = IngredientNormalizer()
query_builder = QueryBuilder(normalizer)

st.title("🔎 Търсене на рецепти по налични продукти")

raw = st.text_input("Продукти (разделени със запетая)", "chicken, tomato, garlic")
model = st.selectbox("Модел", ["BM25", "TF-IDF"])
top_k = st.slider("Top K", 5, 20, 10)

items = [x.strip() for x in raw.split(",") if x.strip()]
built = query_builder.build(items)

if st.button("Търси"):
    if model == "BM25":
        results = bm25.search(built.query_tokens, top_k=top_k)
    else:
        results = tfidf.search(built.query_tokens, top_k=top_k)

    for r in results:
        doc = documents_by_id[r.doc_id]
        with st.container(border=True):
            st.subheader(doc["title"])
            st.caption(f"score: {r.score:.4f} | doc_id: {r.doc_id}")
            if doc.get("link"):
                st.write(f"🔗 {doc['link']}")
            st.write("**Съставки:** " + ", ".join(doc["ingredients"][:25]))

st.divider()
st.subheader("📊 Оценяване (self-retrieval)")

eval_model = st.selectbox("Evaluation model", ["BM25", "TF-IDF"], key="eval_model")
eval_k = st.selectbox("K", [5, 10, 20], index=1, key="eval_k")
eval_n = st.slider("Брой заявки (sample)", 100, 2000, 500, 100, key="eval_n")

if st.button("Run evaluation"):
    idx = bm25 if eval_model == "BM25" else tfidf
    with st.spinner("Running self-retrieval evaluation..."):
        p_at_k, r_at_k = evaluate_self_retrieval(
            index=idx,
            documents_by_id=documents_by_id,
            k=int(eval_k),
            max_queries=int(eval_n),
        )

    c1, c2 = st.columns(2)
    c1.metric(f"Precision@{eval_k}", f"{p_at_k:.4f}")
    c2.metric(f"Recall@{eval_k}", f"{r_at_k:.4f}")

    st.caption(
        "Забележка: При self-retrieval има точно 1 релевантна рецепта за заявка, "
        "затова Precision@K = Recall@K / K."
    )

