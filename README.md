# Recipe Search by Available Ingredients 🍳 

A system for searching and ranking recipes based on available ingredients, built using classical **Information Retrieval (IR)** methods. The project uses the **RecipeNLG** dataset and compares **BM25** and **TF-IDF** models for recipe retrieval.

---

## Overview

Recipes are treated as text documents and ingredients as terms.  
Users provide a list of available ingredients, and the system returns recipes that match them fully or partially, ranked by relevance.

---

## Dataset

The project uses the **RecipeNLG** dataset (Kaggle), which contains over 1.3 million recipes collected from various culinary websites. Each recipe includes:
- title;
- raw ingredient list;
- NER-extracted ingredients;
- link to the original source.

When available, NER ingredients are preferred; otherwise, raw ingredients are normalized.

---

## Preprocessing

Ingredient normalization includes:
- removing quantities and measurement units;
- removing descriptive words and stopwords;
- punctuation cleaning;
- converting words to singular form.

This ensures a unified representation suitable for indexing and retrieval.

---

## Retrieval Models

- **TF-IDF** – a baseline vector space model;
- **BM25** – a probabilistic ranking model with document length normalization.

---

## Evaluation

Due to the lack of manual relevance labels, evaluation is performed using the **self-retrieval** approach:
- each recipe’s ingredients are used as a query;
- the original recipe is considered relevant.

Evaluation metrics:
- **Precision@K**
- **Recall@K**

---

## Running the Project

### Install dependencies
```bash
pip install -r requirements.txt
```

### Build indexes
```bash
python scripts/build_index.py
```

### Run the demo application
```bash
streamlit run app/streamlit.py
``





