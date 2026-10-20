# NewsPulse AI — News & Incident Intelligence Engine
**NLP Assignment 4: POS Tagging, Word Embeddings, and Text Classification**

---

## 🌟 Overview
NewsPulse AI is a modular NLP pipeline and interactive Streamlit web application engineered for news snippet processing, syntactic linguistic filtering, word representation learning, and document classification.

---

## 🏗️ Architecture & Modules

### 1. `corpus_data.py` (Domain Dataset)
- **63 multi-sentence news/incident snippets** across 4 balanced categories:
  - `AI & Cybersecurity` (15 snippets)
  - `Finance & Markets` (15 snippets)
  - `Enterprise Cloud & Hardware` (15 snippets)
  - `Policy & Regulation` (18 snippets)
- Helper utilities for dataset retrieval, category filtering, and formatted title listing for dropdown select menus.

### 2. `embedding_classifier.py` (Core NLP Engine)
- **POS Tagging & Filtering (`perform_pos_tagging`)**:
  - Uses NLTK's `averaged_perceptron_tagger` (with clean regex/rule fallback).
  - Calculates POS family frequency and density distributions:
    - Nouns (`NN*`), Verbs (`VB*`), Adjectives (`JJ*`), Adverbs (`RB*`), Determiners (`DT*`), Prepositions (`IN*`), Pronouns (`PRP*`/`WP*`), Conjunctions (`CC*`), Other.
    - Computes **Noun-to-Verb Ratio** and **Adjective Density Ratio**.
  - Creates a **POS-Filtered Content Token Stream** retaining only content-bearing syntactic tokens (`NN*`, `VB*`, `JJ*`) while eliminating functional syntactic noise.
- **Word Embeddings (`train_and_extract_embeddings`)**:
  - Trains a `gensim.models.Word2Vec` model (`vector_size=50`, `window=4`, `min_count=1`, CBOW).
  - Includes an automatic **Truncated SVD / PPMI Co-occurrence Matrix** dense fallback if `gensim` is unavailable.
  - Extracts 50-D mean document embeddings for:
    - (a) **All Tokens Vector**
    - (b) **POS-Filtered Content Tokens Vector**
  - Synthesizes a **102-Dimensional Hybrid Feature Vector**:
    $$\mathbf{v}_{\text{doc}} = \big[\mathbf{v}_{\text{all}} \;\Vert\; \mathbf{v}_{\text{content}} \;\Vert\; \text{Noun/Verb Ratio} \;\Vert\; \text{Adj Ratio}\big]$$
  - Projects high-dimensional embeddings to 2D via **Principal Component Analysis (PCA)** for semantic vector-space plotting.
  - Computes top-$k$ cosine nearest semantic neighbours for interactive token querying.
- **Classification Pipeline (`classify_document`)**:
  - Trains dual classifiers on the 102-D document feature vectors:
    - **Multinomial Logistic Regression** (`max_iter=1000`)
    - **Random Forest Classifier** (`n_estimators=200`)
  - Evaluates models with **5-Fold Stratified Cross-Validation** (Accuracy & Weighted F1).
  - Generates class probability distributions and consensus detection.

### 3. `app.py` (Streamlit UI & Interactive Visualizations)
- **Modern Dark Cyberpunk/Fintech UI**:
  - Glassmorphic dark theme with gradient titles, badges, and responsive containers.
  - **Input Modes**:
    1. 📋 Choose from 63 curated sample articles across the 4 domains.
    2. ✍️ Paste custom multi-sentence articles/incident reports.
    3. 📎 Upload custom `.txt` or `.pdf` incident reports.
- **Tab 1: 📄 Article Input & POS Tagging**:
  - Summary KPI cards (Total Tokens, Unique Tokens, Content Tokens, Noun/Verb Ratio, Adjective Density).
  - Color-coded interactive full token stream with POS badges.
  - Content-bearing POS-filtered token stream.
  - Horizontal bar chart of POS family distributions and full tag data table.
- **Tab 2: 🧠 Word Embeddings & 2D Vector Space**:
  - 50-D document embedding heatmaps (All Tokens vs. POS-Filtered Content Tokens).
  - Interactive 2D PCA vector space scatter plot with semantic clusters and hover metadata.
  - **Top-5 Nearest Neighbour Word Explorer**: Select any content token to view cosine-similarity matches with live similarity bars.
  - 102-D composite feature breakdown visualization.
- **Tab 3: 🎯 Embedding-Based Text Classification**:
  - Dual model side-by-side predictions (Logistic Regression & Random Forest).
  - Cross-validation performance metrics (Accuracy, F1-score).
  - Horizontal probability distribution plots per category.
  - Model consensus indicator (Agreement/Disagreement).
  - Feature dimension composition breakdown.

---

## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch Streamlit Application
```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.
