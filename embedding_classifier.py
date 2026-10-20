"""
embedding_classifier.py
-----------------------
Core NLP engine for NewsPulse AI:
  1. POS tagging  (NLTK averaged_perceptron_tagger)
  2. Word embeddings (gensim Word2Vec with sklearn SVD fallback)
  3. Text classification (LogisticRegression + RandomForest)
  4. 2D PCA projection for vector-space visualisation
"""

from __future__ import annotations

import re
import string
import warnings
import numpy as np
from collections import Counter, defaultdict
from typing import List, Tuple, Dict, Optional, Any

warnings.filterwarnings("ignore")

# ── Optional heavy imports ────────────────────────────────────────────────────
try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk import pos_tag

    # Silently download required NLTK data
    for _pkg in ["punkt", "averaged_perceptron_tagger", "punkt_tab",
                  "averaged_perceptron_tagger_eng"]:
        try:
            nltk.download(_pkg, quiet=True)
        except Exception:
            pass
    NLTK_OK = True
except ImportError:
    NLTK_OK = False

try:
    from gensim.models import Word2Vec
    GENSIM_OK = True
except ImportError:
    GENSIM_OK = False

from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer


# ══════════════════════════════════════════════════════════════════════════════
# 1. POS TAGGING
# ══════════════════════════════════════════════════════════════════════════════

# POS family prefixes
_POS_FAMILIES = {
    "Nouns": ("NN",),
    "Verbs": ("VB",),
    "Adjectives": ("JJ",),
    "Adverbs": ("RB",),
    "Determiners": ("DT",),
    "Prepositions": ("IN",),
    "Pronouns": ("PRP", "WP"),
    "Conjunctions": ("CC",),
    "Other": (),          # catch-all
}

# Content-bearing tags to KEEP in the filtered stream
_CONTENT_PREFIXES = ("NN", "VB", "JJ")   # NN*, VB*, JJ*


def _simple_tokenise(text: str) -> List[str]:
    """Fallback tokeniser when NLTK is unavailable."""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    return [t for t in text.split() if t]


def _simple_pos_tag(tokens: List[str]) -> List[Tuple[str, str]]:
    """Minimal fallback tagger: assign NN to all tokens."""
    return [(t, "NN") for t in tokens]


def perform_pos_tagging(text: str) -> Dict[str, Any]:
    """
    Parameters
    ----------
    text : str  Raw input text.

    Returns
    -------
    dict with keys:
        tokens          – list[str]
        tagged          – list[(token, tag)]
        pos_counts      – {family: count}
        pos_ratios      – {family: ratio}
        noun_verb_ratio – float
        adj_ratio       – float
        content_tokens  – list[str]  (POS-filtered stream)
    """
    # ── Tokenise ─────────────────────────────────────────────────
    if NLTK_OK:
        try:
            tokens = word_tokenize(text.lower())
            tokens = [t for t in tokens if t not in string.punctuation and t.strip()]
        except Exception:
            tokens = _simple_tokenise(text)
    else:
        tokens = _simple_tokenise(text)

    if not tokens:
        return {
            "tokens": [],
            "tagged": [],
            "pos_counts": {k: 0 for k in _POS_FAMILIES},
            "pos_ratios": {k: 0.0 for k in _POS_FAMILIES},
            "noun_verb_ratio": 0.0,
            "adj_ratio": 0.0,
            "content_tokens": [],
        }

    # ── Tag ──────────────────────────────────────────────────────
    if NLTK_OK:
        try:
            tagged = pos_tag(tokens, tagset=None)
        except Exception:
            tagged = _simple_pos_tag(tokens)
    else:
        tagged = _simple_pos_tag(tokens)

    # ── Aggregate POS families ────────────────────────────────────
    family_counts: Dict[str, int] = {k: 0 for k in _POS_FAMILIES}
    content_tokens: List[str] = []

    for token, tag in tagged:
        placed = False
        for family, prefixes in _POS_FAMILIES.items():
            if family == "Other":
                continue
            for pfx in prefixes:
                if tag.startswith(pfx):
                    family_counts[family] += 1
                    placed = True
                    break
            if placed:
                break
        if not placed:
            family_counts["Other"] += 1

        # Content-bearing filter
        if any(tag.startswith(pfx) for pfx in _CONTENT_PREFIXES):
            content_tokens.append(token)

    total = max(len(tagged), 1)
    pos_ratios = {k: v / total for k, v in family_counts.items()}

    nouns = family_counts["Nouns"]
    verbs = max(family_counts["Verbs"], 1)
    adj   = family_counts["Adjectives"]

    return {
        "tokens": tokens,
        "tagged": tagged,
        "pos_counts": family_counts,
        "pos_ratios": pos_ratios,
        "noun_verb_ratio": round(nouns / verbs, 4),
        "adj_ratio": round(adj / total, 4),
        "content_tokens": content_tokens,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 2. WORD EMBEDDINGS
# ══════════════════════════════════════════════════════════════════════════════

_VECTOR_SIZE = 50
_WINDOW = 4
_MIN_COUNT = 1


def _build_sentences(corpus_texts: List[str]) -> List[List[str]]:
    """Tokenise all corpus texts into sentence lists for Word2Vec training."""
    sentences = []
    for text in corpus_texts:
        if NLTK_OK:
            try:
                toks = word_tokenize(text.lower())
                toks = [t for t in toks if t not in string.punctuation and t.strip()]
            except Exception:
                toks = _simple_tokenise(text)
        else:
            toks = _simple_tokenise(text)
        if toks:
            sentences.append(toks)
    return sentences


def _svd_fallback_embeddings(corpus_texts: List[str]) -> Dict[str, np.ndarray]:
    """
    Build word embeddings via SVD on a word co-occurrence matrix.
    Used when gensim is unavailable.
    """
    all_tokens: List[List[str]] = _build_sentences(corpus_texts)
    vocab = list({tok for sent in all_tokens for tok in sent})
    w2i = {w: i for i, w in enumerate(vocab)}
    V = len(vocab)

    # Build word-context co-occurrence using a window
    cooc = np.zeros((V, V), dtype=np.float32)
    for sent in all_tokens:
        for pos, word in enumerate(sent):
            lo = max(0, pos - _WINDOW)
            hi = min(len(sent), pos + _WINDOW + 1)
            for ctx_pos in range(lo, hi):
                if ctx_pos != pos:
                    cooc[w2i[word], w2i[sent[ctx_pos]]] += 1.0

    # PPMI
    row_sum = cooc.sum(axis=1, keepdims=True)
    col_sum = cooc.sum(axis=0, keepdims=True)
    total   = cooc.sum() + 1e-9
    ppmi    = np.maximum(np.log((cooc * total) / (row_sum * col_sum + 1e-9) + 1e-9), 0)

    n_components = min(_VECTOR_SIZE, V - 1) if V > 1 else 1
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    vecs = svd.fit_transform(ppmi)

    # Pad to exactly _VECTOR_SIZE
    if vecs.shape[1] < _VECTOR_SIZE:
        pad = np.zeros((V, _VECTOR_SIZE - vecs.shape[1]), dtype=np.float32)
        vecs = np.hstack([vecs, pad])

    return {word: vecs[w2i[word]] for word in vocab}


def _mean_vector(tokens: List[str], wv: Dict[str, np.ndarray]) -> np.ndarray:
    """Compute mean embedding vector for a token list."""
    vecs = [wv[t] for t in tokens if t in wv]
    if not vecs:
        return np.zeros(_VECTOR_SIZE, dtype=np.float32)
    return np.mean(vecs, axis=0).astype(np.float32)


def train_and_extract_embeddings(
    corpus_texts: List[str],
    query_tokens: List[str],
    query_content_tokens: List[str],
    noun_verb_ratio: float,
    adj_ratio: float,
) -> Dict[str, Any]:
    """
    Train embeddings on corpus_texts and extract document vectors for the query.

    Returns
    -------
    dict:
        wv              – {word: np.ndarray(50,)}  full vocabulary
        all_vec         – 50-D mean vector over all query tokens
        content_vec     – 50-D mean vector over POS-filtered content tokens
        doc_vector      – 102-D combined feature vector [all_vec | content_vec | ratios]
        model_type      – "Word2Vec" or "SVD"
        pca_2d          – {word: (x, y)} for top-N vocabulary words
        vocab           – list[str]
    """
    sentences = _build_sentences(corpus_texts)

    if GENSIM_OK:
        model = Word2Vec(
            sentences=sentences,
            vector_size=_VECTOR_SIZE,
            window=_WINDOW,
            min_count=_MIN_COUNT,
            workers=2,
            epochs=10,
            sg=0,           # CBOW
        )
        wv: Dict[str, np.ndarray] = {w: model.wv[w] for w in model.wv.index_to_key}
        model_type = "Word2Vec (gensim)"
    else:
        wv = _svd_fallback_embeddings(corpus_texts)
        model_type = "SVD Co-occurrence (sklearn)"

    # ── Document vectors ─────────────────────────────────────────
    all_vec     = _mean_vector(query_tokens, wv)
    content_vec = _mean_vector(query_content_tokens, wv)

    # Feature engineering: concat both vecs + two scalar ratios
    ratio_arr   = np.array([noun_verb_ratio, adj_ratio], dtype=np.float32)
    doc_vector  = np.concatenate([all_vec, content_vec, ratio_arr])   # shape (102,)

    # ── 2D PCA projection ─────────────────────────────────────────
    vocab_list = list(wv.keys())[:300]          # limit to 300 words for speed
    if len(vocab_list) >= 2:
        mat = np.stack([wv[w] for w in vocab_list])
        n_c = min(2, mat.shape[0], mat.shape[1])
        pca = PCA(n_components=n_c, random_state=42)
        coords_2d = pca.fit_transform(mat)
        if coords_2d.shape[1] < 2:
            coords_2d = np.hstack([coords_2d, np.zeros((len(vocab_list), 1))])
        pca_2d = {w: (float(coords_2d[i, 0]), float(coords_2d[i, 1]))
                  for i, w in enumerate(vocab_list)}
    else:
        pca_2d = {}

    return {
        "wv": wv,
        "all_vec": all_vec,
        "content_vec": content_vec,
        "doc_vector": doc_vector,
        "model_type": model_type,
        "pca_2d": pca_2d,
        "vocab": vocab_list,
    }


def get_nearest_words(
    token: str,
    wv: Dict[str, np.ndarray],
    top_k: int = 5,
) -> List[Tuple[str, float]]:
    """Return top-k semantically nearest words by cosine similarity."""
    if token not in wv:
        return []
    query_vec = wv[token]
    norm_q = np.linalg.norm(query_vec) + 1e-9
    sims = []
    for word, vec in wv.items():
        if word == token:
            continue
        norm_w = np.linalg.norm(vec) + 1e-9
        cos = float(np.dot(query_vec, vec) / (norm_q * norm_w))
        sims.append((word, cos))
    sims.sort(key=lambda x: x[1], reverse=True)
    return sims[:top_k]


# ══════════════════════════════════════════════════════════════════════════════
# 3. CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════

def classify_document(
    doc_vector: np.ndarray,
    corpus_items: List[Dict],
    embedding_results: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Train LogisticRegression and RandomForest on corpus embeddings,
    then classify the query doc_vector.

    Parameters
    ----------
    doc_vector   : 102-D feature vector for the query document.
    corpus_items : list of corpus dicts (each has 'text', 'label', 'category').
    embedding_results : output of train_and_extract_embeddings (contains wv).

    Returns
    -------
    dict:
        lr_pred, lr_proba, lr_acc, lr_f1
        rf_pred, rf_proba, rf_acc, rf_f1
        categories      – list[str]
        label_names     – {int: str}
    """
    wv = embedding_results["wv"]
    corpus_texts  = [item["text"]  for item in corpus_items]
    corpus_labels = [item["label"] for item in corpus_items]
    corpus_cats   = [item["category"] for item in corpus_items]

    # Build feature matrix for corpus
    X_corpus = []
    for item in corpus_items:
        pos_result = perform_pos_tagging(item["text"])
        all_v   = _mean_vector(pos_result["tokens"],         wv)
        cont_v  = _mean_vector(pos_result["content_tokens"], wv)
        ratios  = np.array([pos_result["noun_verb_ratio"], pos_result["adj_ratio"]], dtype=np.float32)
        X_corpus.append(np.concatenate([all_v, cont_v, ratios]))

    X = np.array(X_corpus, dtype=np.float32)
    y = np.array(corpus_labels, dtype=int)

    label_names = {
        0: "AI & Cybersecurity",
        1: "Finance & Markets",
        2: "Enterprise Cloud & Hardware",
        3: "Policy & Regulation",
    }

    # ── Logistic Regression ───────────────────────────────────────
    lr = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    lr.fit(X, y)
    lr_cv_acc = cross_val_score(lr, X, y, cv=5, scoring="accuracy").mean()
    lr_cv_f1  = cross_val_score(lr, X, y, cv=5, scoring="f1_weighted").mean()
    q_vec = doc_vector.reshape(1, -1).astype(np.float32)
    lr_pred  = int(lr.predict(q_vec)[0])
    lr_proba = lr.predict_proba(q_vec)[0].tolist()

    # ── Random Forest ─────────────────────────────────────────────
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    rf_cv_acc = cross_val_score(rf, X, y, cv=5, scoring="accuracy").mean()
    rf_cv_f1  = cross_val_score(rf, X, y, cv=5, scoring="f1_weighted").mean()
    rf_pred  = int(rf.predict(q_vec)[0])
    rf_proba = rf.predict_proba(q_vec)[0].tolist()

    return {
        "lr_pred":   lr_pred,
        "lr_proba":  lr_proba,
        "lr_acc":    round(lr_cv_acc, 4),
        "lr_f1":     round(lr_cv_f1,  4),
        "rf_pred":   rf_pred,
        "rf_proba":  rf_proba,
        "rf_acc":    round(rf_cv_acc, 4),
        "rf_f1":     round(rf_cv_f1,  4),
        "label_names": label_names,
        "classes":   list(lr.classes_),
    }
