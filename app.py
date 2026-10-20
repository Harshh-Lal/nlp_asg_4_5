"""
app.py  ──  NewsPulse AI — News & Incident Intelligence Engine
--------------------------------------------------------------
Streamlit application with five tabs:
  Tab 1 : 📄 Article Input & POS Tagging
  Tab 2 : 🧠 Word Embeddings & 2D Vector Space
  Tab 3 : 🎯 Embedding-Based Text Classification
  Tab 4 : 🏷️ Named Entity Recognition (NER)
  Tab 5 : 🕸️ Entity Relationship & Intelligence Summary

Run:
    python -m streamlit run app.py
"""

from __future__ import annotations

import io
import re
import textwrap
import warnings
from typing import Dict, List, Optional, Tuple
from collections import Counter

import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

# ── Local modules ─────────────────────────────────────────────────────────────
from corpus_data import (
    CORPUS,
    CATEGORIES,
    get_sample_titles,
    get_item_by_index,
    get_all_texts,
)
from embedding_classifier import (
    perform_pos_tagging,
    train_and_extract_embeddings,
    classify_document,
    get_nearest_words,
)
from ner_engine import (
    extract_entities,
    render_displacy_html,
    ENTITY_DESCRIPTIONS,
    ENTITY_COLORS,
)

# ── Optional PDF reader ────────────────────────────────────────────────────────
try:
    import pdfplumber
    PDF_OK = True
except ImportError:
    PDF_OK = False


# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG & THEME
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="NewsPulse AI — Intelligence Engine",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base ────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background: #0a0e1a !important;
    font-family: 'Inter', sans-serif;
    color: #e2e8f0;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1321 0%, #111827 100%) !important;
    border-right: 1px solid #1e2d4a;
}

/* ── Header ─────────────────────────────────────────────── */
.main-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
    border: 1px solid #312e81;
    border-radius: 16px;
    padding: 24px 32px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.main-header::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle at 30% 40%, rgba(99,102,241,0.15) 0%, transparent 60%),
                radial-gradient(circle at 70% 60%, rgba(139,92,246,0.10) 0%, transparent 60%);
    pointer-events: none;
}
.main-header h1 {
    font-size: 2rem;
    font-weight: 800;
    background: linear-gradient(90deg, #818cf8, #a78bfa, #e879f9);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    letter-spacing: -0.5px;
}
.main-header p {
    color: #94a3b8;
    margin: 6px 0 0;
    font-size: 0.9rem;
}

/* ── Cards ───────────────────────────────────────────────── */
.metric-card {
    background: linear-gradient(135deg, #111827, #1a2035);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 16px 20px;
    text-align: center;
    transition: border-color 0.2s, transform 0.2s;
}
.metric-card:hover {
    border-color: #6366f1;
    transform: translateY(-2px);
}
.metric-card .val {
    font-size: 1.8rem;
    font-weight: 700;
    color: #818cf8;
}
.metric-card .lbl {
    font-size: 0.75rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 4px;
}

/* ── POS badge ───────────────────────────────────────────── */
.pos-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 6px;
    font-size: 0.75rem;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
    margin: 2px 2px;
}
.pos-NN  { background:#1e3a5f; color:#93c5fd; border:1px solid #2563eb; }
.pos-VB  { background:#14532d; color:#86efac; border:1px solid #16a34a; }
.pos-JJ  { background:#3b0764; color:#d8b4fe; border:1px solid #9333ea; }
.pos-RB  { background:#422006; color:#fed7aa; border:1px solid #ea580c; }
.pos-DT  { background:#1c1c1c; color:#9ca3af; border:1px solid #374151; }
.pos-IN  { background:#111827; color:#6b7280; border:1px solid #374151; }
.pos-OT  { background:#0f172a; color:#475569; border:1px solid #1e293b; }

/* ── Token stream ────────────────────────────────────────── */
.token-stream {
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 16px;
    line-height: 2.2;
    font-size: 0.88rem;
    max-height: 280px;
    overflow-y: auto;
}

/* ── Section title ───────────────────────────────────────── */
.section-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #a5b4fc;
    border-left: 3px solid #6366f1;
    padding-left: 10px;
    margin: 20px 0 12px;
    letter-spacing: 0.02em;
}

/* ── Classifier pill ─────────────────────────────────────── */
.pred-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 8px 18px;
    border-radius: 24px;
    font-weight: 700;
    font-size: 1rem;
    letter-spacing: 0.02em;
}
.pill-ai   { background: linear-gradient(90deg,#312e81,#4c1d95); color:#c4b5fd; border:1px solid #7c3aed; }
.pill-fin  { background: linear-gradient(90deg,#064e3b,#065f46); color:#6ee7b7; border:1px solid #059669; }
.pill-cloud{ background: linear-gradient(90deg,#1e3a5f,#0c4a6e); color:#7dd3fc; border:1px solid #0284c7; }
.pill-pol  { background: linear-gradient(90deg,#7c2d12,#9a3412); color:#fdba74; border:1px solid #ea580c; }

/* ── Nearest words ────────────────────────────────────────── */
.nn-row {
    display:flex; align-items:center; gap:10px;
    padding:6px 10px; border-radius:8px;
    margin:4px 0; background:#111827;
    border:1px solid #1e2d4a;
}
.nn-word { font-family:'JetBrains Mono', monospace; font-weight:600; color:#a5b4fc; width:160px; }
.nn-bar  { flex:1; background:#1e293b; border-radius:4px; height:8px; overflow:hidden; }
.nn-fill { height:100%; border-radius:4px;
           background:linear-gradient(90deg,#6366f1,#a78bfa); }
.nn-score{ color:#64748b; font-size:0.78rem; width:50px; text-align:right; }

/* ── Intelligence Box ─────────────────────────────────────── */
.intel-box {
    background: linear-gradient(135deg, #0d1321, #161f38);
    border: 1px solid #233554;
    border-radius: 12px;
    padding: 18px 24px;
    margin-bottom: 16px;
}
.intel-header {
    font-size: 0.95rem;
    font-weight: 700;
    color: #93c5fd;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.intel-item {
    font-size: 0.88rem;
    color: #cbd5e1;
    line-height: 1.6;
    margin: 4px 0;
}

/* ── Tabs ─────────────────────────────────────────────────── */
[data-testid="stTab"] { color: #64748b !important; }
[data-testid="stTab"][aria-selected="true"] { color: #818cf8 !important; font-weight:700; }

/* ── Scrollbar ────────────────────────────────────────────── */
::-webkit-scrollbar { width:6px; }
::-webkit-scrollbar-track { background:#0d1117; }
::-webkit-scrollbar-thumb { background:#1e3a5f; border-radius:3px; }

/* ── Sidebar labels ───────────────────────────────────────── */
[data-testid="stSidebar"] label, [data-testid="stSidebar"] .stSelectbox label {
    color: #94a3b8 !important;
    font-weight: 500;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

_TAG_COLOURS = {
    "NN": "pos-NN", "NNS": "pos-NN", "NNP": "pos-NN", "NNPS": "pos-NN",
    "VB": "pos-VB", "VBD": "pos-VB", "VBG": "pos-VB", "VBN": "pos-VB",
    "VBP": "pos-VB", "VBZ": "pos-VB",
    "JJ": "pos-JJ", "JJR": "pos-JJ", "JJS": "pos-JJ",
    "RB": "pos-RB", "RBR": "pos-RB", "RBS": "pos-RB",
    "DT": "pos-DT", "IN": "pos-IN",
}

_CATEGORY_ICONS = {
    "AI & Cybersecurity": "🛡️",
    "Finance & Markets": "📈",
    "Enterprise Cloud & Hardware": "☁️",
    "Policy & Regulation": "⚖️",
}

_PILL_CLASSES = {
    0: "pill-ai",
    1: "pill-fin",
    2: "pill-cloud",
    3: "pill-pol",
}

_LABEL_NAMES = {
    0: "AI & Cybersecurity",
    1: "Finance & Markets",
    2: "Enterprise Cloud & Hardware",
    3: "Policy & Regulation",
}


def _tag_to_class(tag: str) -> str:
    for pfx in ["NN", "VB", "JJ", "RB", "DT", "IN"]:
        if tag.startswith(pfx):
            return _TAG_COLOURS.get(pfx, "pos-OT")
    return "pos-OT"


def _render_token_stream(tagged: List[Tuple[str, str]]) -> str:
    """Render a coloured HTML token stream."""
    parts = []
    for token, tag in tagged:
        cls = _tag_to_class(tag)
        parts.append(
            f'<span class="pos-badge {cls}" title="{tag}">{token}</span>'
        )
    return '<div class="token-stream">' + " ".join(parts) + "</div>"


def _pdf_to_text(uploaded_file) -> str:
    if not PDF_OK:
        return "⚠️ pdfplumber not installed. Install via: pip install pdfplumber"
    with pdfplumber.open(io.BytesIO(uploaded_file.read())) as pdf:
        return "\n".join(
            page.extract_text() or "" for page in pdf.pages
        )


@st.cache_resource(show_spinner=False)
def _get_corpus_texts() -> List[str]:
    return [item["text"] for item in CORPUS]


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:12px 0 20px;'>
        <div style='font-size:2.4rem;'>🔍</div>
        <div style='font-size:1.1rem;font-weight:800;
             background:linear-gradient(90deg,#818cf8,#a78bfa);
             -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>
            NewsPulse AI
        </div>
        <div style='color:#475569;font-size:0.75rem;margin-top:4px;'>
            NLP Assignment 4 &amp; 5
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<div style='color:#94a3b8;font-size:0.8rem;font-weight:600;"
                "text-transform:uppercase;letter-spacing:0.1em;margin-bottom:8px;'>"
                "Input Mode</div>", unsafe_allow_html=True)

    input_mode = st.radio(
        "Select input method",
        ["📋 Sample Article", "✍️ Custom Text", "📎 Upload File"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("""
    <div style='color:#475569;font-size:0.78rem;line-height:1.7;'>
        <b style='color:#64748b;'>NLP Engine Stack</b><br>
        🔤 NLTK · Averaged Perceptron POS<br>
        🧬 Gensim Word2Vec / SVD (50-D)<br>
        🤖 LogReg + Random Forest (102-D)<br>
        🏷️ spaCy NER + Custom EntityRuler<br>
        🕸️ Sentence Entity Linking Engine<br>
        <br>
        <b style='color:#64748b;'>Corpus</b><br>
        63 multi-sentence snippets · 4 categories
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="main-header">
    <h1>🔍 NewsPulse AI — News &amp; Incident Intelligence Engine</h1>
    <p>POS Tagging · Word Embeddings · Classification · Named Entity Recognition (NER) · Entity Linking</p>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# INPUT SECTION
# ══════════════════════════════════════════════════════════════════════════════

article_text: str = ""
article_title: str = "Custom Input"
true_label: Optional[int] = None

if input_mode == "📋 Sample Article":
    titles = get_sample_titles()
    sel = st.selectbox("Choose a sample article:", titles)
    idx = titles.index(sel)
    item = get_item_by_index(idx)
    article_text  = item["text"]
    article_title = item["title"]
    true_label    = item["label"]

    icon = _CATEGORY_ICONS.get(item["category"], "📰")
    st.info(
        f"{icon} **{item['category']}** — {item['title']}"
    )
    with st.expander("📄 Article Text", expanded=True):
        st.write(article_text)

elif input_mode == "✍️ Custom Text":
    article_text = st.text_area(
        "Paste your news article / incident report:",
        height=200,
        placeholder="Enter any multi-sentence news text or incident disclosure here …",
    )
    article_title = "Custom Input"

else:  # Upload
    uploaded = st.file_uploader("Upload a .txt or .pdf file:", type=["txt", "pdf"])
    if uploaded:
        if uploaded.name.endswith(".pdf"):
            article_text = _pdf_to_text(uploaded)
        else:
            article_text = uploaded.read().decode("utf-8", errors="ignore")
        article_title = uploaded.name
        with st.expander("📄 Uploaded Content", expanded=False):
            st.write(article_text[:2000] + ("…" if len(article_text) > 2000 else ""))


# ── Analyse button ─────────────────────────────────────────────────────────────
analyse_clicked = st.button(
    "⚡ Analyse Article",
    type="primary",
    use_container_width=True,
    disabled=(not article_text.strip()),
)


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE  –  cache computation across tab switches
# ══════════════════════════════════════════════════════════════════════════════

if "results" not in st.session_state:
    st.session_state.results = None

if analyse_clicked and article_text.strip():
    with st.spinner("🔄 Running full NLP & NER pipeline (POS, Embeddings, Classification, EntityRuler) …"):
        # 1. POS tagging
        pos_result = perform_pos_tagging(article_text)

        # 2. Embeddings
        corpus_texts = _get_corpus_texts()
        emb_result = train_and_extract_embeddings(
            corpus_texts=corpus_texts,
            query_tokens=pos_result["tokens"],
            query_content_tokens=pos_result["content_tokens"],
            noun_verb_ratio=pos_result["noun_verb_ratio"],
            adj_ratio=pos_result["adj_ratio"],
        )

        # 3. Classification
        clf_result = classify_document(
            doc_vector=emb_result["doc_vector"],
            corpus_items=CORPUS,
            embedding_results=emb_result,
        )

        # 4. Named Entity Recognition
        ner_result = extract_entities(article_text)

        st.session_state.results = {
            "text": article_text,
            "title": article_title,
            "true_label": true_label,
            "pos": pos_result,
            "emb": emb_result,
            "clf": clf_result,
            "ner": ner_result,
        }
    st.success("✅ Full intelligence pipeline complete!")


# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════

if st.session_state.results:
    R = st.session_state.results
    pos = R["pos"]
    emb = R["emb"]
    clf = R["clf"]
    ner = R["ner"]

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📄 Article Input & POS Tagging",
        "🧠 Word Embeddings & 2D Vector Space",
        "🎯 Embedding-Based Text Classification",
        "🏷️ Named Entity Recognition (NER)",
        "🕸️ Entity Relationship & Intelligence Summary",
    ])

    # ══════════════════════════════════════════════════════════════
    # TAB 1 : POS TAGGING
    # ══════════════════════════════════════════════════════════════
    with tab1:
        st.markdown(f"<div class='section-title'>📰 {R['title']}</div>",
                    unsafe_allow_html=True)

        c1, c2, c3, c4, c5 = st.columns(5)
        metrics = [
            (c1, len(pos["tokens"]),              "Total Tokens"),
            (c2, len(set(pos["tokens"])),          "Unique Tokens"),
            (c3, len(pos["content_tokens"]),       "Content Tokens"),
            (c4, pos["noun_verb_ratio"],            "Noun/Verb Ratio"),
            (c5, f'{pos["adj_ratio"]:.1%}',         "Adj Density"),
        ]
        for col, val, label in metrics:
            with col:
                st.markdown(
                    f'<div class="metric-card">'
                    f'<div class="val">{val}</div>'
                    f'<div class="lbl">{label}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        st.markdown("<div class='section-title'>🏷️ Full POS Token Stream</div>",
                    unsafe_allow_html=True)
        st.markdown(_render_token_stream(pos["tagged"]), unsafe_allow_html=True)
        st.caption(
            "🔵 Nouns (NN*)  🟢 Verbs (VB*)  🟣 Adjectives (JJ*)  "
            "🟠 Adverbs (RB*)  ⚫ Functional syntactic tokens"
        )

        st.markdown("<div class='section-title'>🧹 POS-Filtered Content Token Stream "
                    "<small style='color:#475569;font-weight:400;'>"
                    "(NN* · VB* · JJ* only)</small></div>",
                    unsafe_allow_html=True)
        content_tagged = [(t, tag) for t, tag in pos["tagged"]
                          if t in pos["content_tokens"]]
        st.markdown(_render_token_stream(content_tagged), unsafe_allow_html=True)

        st.markdown("<div class='section-title'>📊 POS Family Distribution</div>",
                    unsafe_allow_html=True)

        families = list(pos["pos_counts"].keys())
        counts   = list(pos["pos_counts"].values())
        ratios_  = [pos["pos_ratios"][f] for f in families]

        col_chart, col_table = st.columns([3, 2])
        with col_chart:
            colours = ["#3b82f6","#22c55e","#a855f7","#f97316",
                       "#64748b","#334155","#1e293b","#0f172a"]
            fig_bar = go.Figure(go.Bar(
                x=counts, y=families,
                orientation="h",
                marker_color=colours[:len(families)],
                text=[f"{r:.1%}" for r in ratios_],
                textposition="outside",
            ))
            fig_bar.update_layout(
                paper_bgcolor="#0a0e1a",
                plot_bgcolor="#111827",
                font_color="#94a3b8",
                margin=dict(l=10, r=40, t=10, b=10),
                height=320,
                xaxis=dict(showgrid=True, gridcolor="#1e293b"),
                yaxis=dict(showgrid=False),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with col_table:
            df_pos = pd.DataFrame({
                "POS Family": families,
                "Count": counts,
                "Ratio": [f"{r:.2%}" for r in ratios_],
            })
            st.dataframe(
                df_pos,
                hide_index=True,
                use_container_width=True,
                height=310,
            )

        with st.expander("🔎 Full Tagged Token Table"):
            df_tagged = pd.DataFrame(pos["tagged"], columns=["Token", "POS Tag"])
            st.dataframe(df_tagged, use_container_width=True, height=300)

    # ══════════════════════════════════════════════════════════════
    # TAB 2 : EMBEDDINGS & 2D VECTOR SPACE
    # ══════════════════════════════════════════════════════════════
    with tab2:
        st.markdown(f"<div class='section-title'>🧬 Embedding Engine: "
                    f"<code style='color:#818cf8;'>{emb['model_type']}</code> "
                    f"· vector_size=50 · window=4</div>",
                    unsafe_allow_html=True)

        c1, c2 = st.columns(2)

        def _vec_heatmap(vec: np.ndarray, title: str):
            mat = vec.reshape(5, 10)
            fig = go.Figure(go.Heatmap(
                z=mat,
                colorscale="Plasma",
                showscale=True,
                colorbar=dict(thickness=12, outlinecolor="#1e293b"),
            ))
            fig.update_layout(
                title=dict(text=title, font=dict(color="#94a3b8", size=13)),
                paper_bgcolor="#0a0e1a", plot_bgcolor="#111827",
                margin=dict(l=10, r=10, t=36, b=10), height=220,
                xaxis=dict(showticklabels=False, showgrid=False),
                yaxis=dict(showticklabels=False, showgrid=False),
            )
            return fig

        with c1:
            st.plotly_chart(
                _vec_heatmap(emb["all_vec"], "50-D Mean Vector — All Tokens"),
                use_container_width=True,
            )
        with c2:
            st.plotly_chart(
                _vec_heatmap(emb["content_vec"], "50-D Mean Vector — Content Tokens"),
                use_container_width=True,
            )

        st.markdown("<div class='section-title'>🗺️ 2D PCA Vector Space"
                    "<small style='color:#475569;font-weight:400;'>"
                    " (top 200 vocabulary words)</small></div>",
                    unsafe_allow_html=True)

        pca_data = emb["pca_2d"]
        query_content_set = set(pos["content_tokens"])
        query_all_set     = set(pos["tokens"])

        if pca_data:
            words  = list(pca_data.keys())[:200]
            xs     = [pca_data[w][0] for w in words]
            ys     = [pca_data[w][1] for w in words]
            labels = []
            for w in words:
                if w in query_content_set:
                    labels.append("Content Token")
                elif w in query_all_set:
                    labels.append("Query Token")
                else:
                    labels.append("Corpus Word")

            df_pca = pd.DataFrame({"word": words, "x": xs, "y": ys, "type": labels})

            colour_map = {
                "Content Token": "#a78bfa",
                "Query Token":   "#34d399",
                "Corpus Word":   "#1e3a5f",
            }
            size_map = {
                "Content Token": 10,
                "Query Token":   8,
                "Corpus Word":   5,
            }
            df_pca["size"] = df_pca["type"].map(size_map)

            fig_scatter = px.scatter(
                df_pca, x="x", y="y",
                color="type",
                text="word",
                color_discrete_map=colour_map,
                hover_data={"word": True, "x": ":.3f", "y": ":.3f"},
            )
            fig_scatter.update_traces(
                textposition="top center",
                textfont=dict(size=9, color="#94a3b8"),
                marker=dict(size=df_pca["size"].tolist()),
            )
            fig_scatter.update_layout(
                paper_bgcolor="#0a0e1a",
                plot_bgcolor="#111827",
                font_color="#94a3b8",
                legend=dict(
                    bgcolor="#111827", bordercolor="#1e3a5f",
                    font=dict(color="#94a3b8"),
                ),
                xaxis=dict(showgrid=True, gridcolor="#1e293b", zeroline=False, title="PCA-1"),
                yaxis=dict(showgrid=True, gridcolor="#1e293b", zeroline=False, title="PCA-2"),
                margin=dict(l=10, r=10, t=10, b=10),
                height=500,
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        st.markdown("<div class='section-title'>🔎 Nearest Word Explorer"
                    "<small style='color:#475569;font-weight:400;'>"
                    " — click any content token to find top-5 semantic neighbours</small></div>",
                    unsafe_allow_html=True)

        unique_content = list(dict.fromkeys(pos["content_tokens"]))[:40]
        if unique_content:
            sel_token = st.selectbox(
                "Select a content token:",
                unique_content,
                label_visibility="collapsed",
            )
            neighbours = get_nearest_words(sel_token, emb["wv"], top_k=5)
            if neighbours:
                max_sim = max(s for _, s in neighbours) if neighbours else 1.0
                st.markdown("**Top-5 semantic neighbours:**")
                html_rows = ""
                for word, sim in neighbours:
                    pct = int(max(sim, 0) / max(max_sim, 1e-9) * 100)
                    html_rows += (
                        f'<div class="nn-row">'
                        f'<span class="nn-word">{word}</span>'
                        f'<div class="nn-bar"><div class="nn-fill" '
                        f'style="width:{pct}%;"></div></div>'
                        f'<span class="nn-score">{sim:.3f}</span>'
                        f'</div>'
                    )
                st.markdown(html_rows, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════
    # TAB 3 : CLASSIFICATION
    # ══════════════════════════════════════════════════════════════
    with tab3:
        label_names = clf["label_names"]

        if R["true_label"] is not None:
            true_cat = label_names[R["true_label"]]
            icon     = _CATEGORY_ICONS.get(true_cat, "📰")
            st.info(f"{icon} Ground-truth label: **{true_cat}**")

        colA, colB = st.columns(2)

        def _render_prediction(col, model_name: str, pred: int,
                                proba: List[float], acc: float, f1: float):
            cat      = label_names[pred]
            icon     = _CATEGORY_ICONS.get(cat, "📰")
            pill_cls = _PILL_CLASSES.get(pred, "pill-ai")
            with col:
                st.markdown(f"<div class='section-title'>{model_name}</div>",
                            unsafe_allow_html=True)
                st.markdown(
                    f'<div class="pred-pill {pill_cls}">'
                    f'{icon} {cat}</div>',
                    unsafe_allow_html=True,
                )
                st.markdown("")
                m1, m2 = st.columns(2)
                with m1:
                    st.metric("CV Accuracy", f"{acc:.1%}")
                with m2:
                    st.metric("CV F1 (weighted)", f"{f1:.1%}")

                cats_list = [label_names[i] for i in sorted(label_names)]
                fig_prob = go.Figure(go.Bar(
                    y=cats_list,
                    x=[proba[i] for i in sorted(label_names)],
                    orientation="h",
                    marker_color=["#a78bfa","#34d399","#38bdf8","#fb923c"],
                    text=[f"{proba[i]:.1%}" for i in sorted(label_names)],
                    textposition="outside",
                ))
                fig_prob.update_layout(
                    paper_bgcolor="#0a0e1a", plot_bgcolor="#111827",
                    font_color="#94a3b8",
                    margin=dict(l=10, r=60, t=10, b=10),
                    height=220,
                    xaxis=dict(range=[0, 1.15], showgrid=True, gridcolor="#1e293b",
                               tickformat=".0%"),
                    yaxis=dict(showgrid=False),
                )
                st.plotly_chart(fig_prob, use_container_width=True)

        _render_prediction(
            colA, "🔷 Logistic Regression",
            clf["lr_pred"], clf["lr_proba"], clf["lr_acc"], clf["lr_f1"],
        )
        _render_prediction(
            colB, "🌳 Random Forest",
            clf["rf_pred"], clf["rf_proba"], clf["rf_acc"], clf["rf_f1"],
        )

        agree = clf["lr_pred"] == clf["rf_pred"]
        st.markdown("---")
        if agree:
            cat = label_names[clf["lr_pred"]]
            icon = _CATEGORY_ICONS.get(cat, "📰")
            st.success(f"✅ Both models agree: **{icon} {cat}**")
        else:
            lr_cat = label_names[clf["lr_pred"]]
            rf_cat = label_names[clf["rf_pred"]]
            st.warning(f"⚠️ Models disagree — LR: **{lr_cat}** | RF: **{rf_cat}**")

        st.markdown("<div class='section-title'>📊 Model Comparison (5-Fold CV)</div>",
                    unsafe_allow_html=True)
        df_metrics = pd.DataFrame({
            "Model": ["Logistic Regression", "Random Forest"],
            "Accuracy (CV)": [f"{clf['lr_acc']:.1%}", f"{clf['rf_acc']:.1%}"],
            "F1 Weighted (CV)": [f"{clf['lr_f1']:.1%}", f"{clf['rf_f1']:.1%}"],
            "Prediction": [label_names[clf["lr_pred"]], label_names[clf["rf_pred"]]],
        })
        st.dataframe(df_metrics, hide_index=True, use_container_width=True)

    # ══════════════════════════════════════════════════════════════
    # TAB 4 : NAMED ENTITY RECOGNITION (NER)
    # ══════════════════════════════════════════════════════════════
    with tab4:
        st.markdown(f"<div class='section-title'>🏷️ Named Entity Recognition (NER) — "
                    f"<code style='color:#818cf8;'>{ner['engine_type']}</code></div>",
                    unsafe_allow_html=True)

        entities_list = ner["entities"]
        all_tags = sorted(list(set(e["label"] for e in entities_list))) if entities_list else []

        # ── KPI Cards ──────────────────────────────────────────────
        k1, k2, k3, k4 = st.columns(4)
        top_tag = max(ner["entity_counts"], key=ner["entity_counts"].get) if ner["entity_counts"] else "None"
        with k1:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="val">{len(entities_list)}</div>'
                f'<div class="lbl">Entities Detected</div></div>',
                unsafe_allow_html=True,
            )
        with k2:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="val">{len(all_tags)}</div>'
                f'<div class="lbl">Entity Classes</div></div>',
                unsafe_allow_html=True,
            )
        with k3:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="val">{top_tag}</div>'
                f'<div class="lbl">Top Tag</div></div>',
                unsafe_allow_html=True,
            )
        with k4:
            rule_count = sum(1 for e in entities_list if "Rule-Based" in e["confidence_type"])
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="val">{rule_count}</div>'
                f'<div class="lbl">EntityRuler Hits</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown("")

        # ── Tag Multiselect Filter ─────────────────────────────────
        if all_tags:
            selected_tags = st.multiselect(
                "Filter displayed entity tags in article markup:",
                options=all_tags,
                default=all_tags,
            )
        else:
            selected_tags = []

        # ── Inline DisplaCy Highlighted View ────────────────────────
        st.markdown("<div class='section-title'>📄 Interactive DisplaCy-Style Inline Entity Highlighting</div>",
                    unsafe_allow_html=True)
        displacy_html = render_displacy_html(R["text"], entities_list, selected_tags)
        st.markdown(displacy_html, unsafe_allow_html=True)

        st.markdown("")

        # ── Entity Distribution Chart + Grouped Table ───────────────
        c_chart, c_groups = st.columns([1, 1])

        with c_chart:
            st.markdown("<div class='section-title'>📊 Entity Class Breakdown</div>",
                        unsafe_allow_html=True)
            if ner["entity_counts"]:
                df_counts = pd.DataFrame(
                    list(ner["entity_counts"].items()),
                    columns=["Tag", "Count"]
                ).sort_values("Count", ascending=False)

                colors = [
                    ENTITY_COLORS.get(t, {}).get("border", "#6366f1")
                    for t in df_counts["Tag"]
                ]
                fig_donut = go.Figure(go.Pie(
                    labels=df_counts["Tag"],
                    values=df_counts["Count"],
                    hole=0.55,
                    marker=dict(colors=colors),
                    textinfo="label+value",
                    textfont=dict(color="#ffffff", size=12),
                ))
                fig_donut.update_layout(
                    paper_bgcolor="#0a0e1a",
                    font_color="#94a3b8",
                    margin=dict(l=10, r=10, t=10, b=10),
                    height=300,
                    legend=dict(bgcolor="#111827", bordercolor="#1e3a5f", font=dict(color="#94a3b8")),
                )
                st.plotly_chart(fig_donut, use_container_width=True)

        with c_groups:
            st.markdown("<div class='section-title'>📁 Grouped Semantic Entities</div>",
                        unsafe_allow_html=True)

            who_ents = [e["text"] for e in entities_list if e["label"] in ["PERSON", "ORG"]]
            where_ents = [e["text"] for e in entities_list if e["label"] in ["GPE", "LOC"]]
            when_ents = [e["text"] for e in entities_list if e["label"] in ["DATE", "TIME"]]
            finance_ents = [e["text"] for e in entities_list if e["label"] in ["MONEY", "PERCENT"]]
            tech_ents = [e["text"] for e in entities_list if e["label"] in ["TECH_PRODUCT", "SECURITY_THREAT", "LAW"]]

            tabs_group = st.tabs(["👥 Who", "📍 Where", "🕒 When", "💰 Impact", "⚡ Tech & Threat"])
            with tabs_group[0]:
                st.write(", ".join(dict.fromkeys(who_ents)) if who_ents else "No PERSON / ORG entities detected.")
            with tabs_group[1]:
                st.write(", ".join(dict.fromkeys(where_ents)) if where_ents else "No GPE / Location entities detected.")
            with tabs_group[2]:
                st.write(", ".join(dict.fromkeys(when_ents)) if when_ents else "No DATE entities detected.")
            with tabs_group[3]:
                st.write(", ".join(dict.fromkeys(finance_ents)) if finance_ents else "No Financial figures detected.")
            with tabs_group[4]:
                st.write(", ".join(dict.fromkeys(tech_ents)) if tech_ents else "No Tech/Threat entities detected.")

        # ── Full Entities Table ─────────────────────────────────────
        with st.expander("🔍 Complete Extracted Entity Table & Confidence Offsets", expanded=False):
            if entities_list:
                df_all_ents = pd.DataFrame([
                    {
                        "Entity": e["text"],
                        "Tag": e["label"],
                        "Span": f"[{e['start_char']} : {e['end_char']}]",
                        "Source / Engine": e["confidence_type"],
                        "Description": e["description"],
                    }
                    for e in entities_list
                ])
                st.dataframe(df_all_ents, use_container_width=True, hide_index=True)

    # ══════════════════════════════════════════════════════════════
    # TAB 5 : ENTITY RELATIONSHIPS & INTELLIGENCE SUMMARY
    # ══════════════════════════════════════════════════════════════
    with tab5:
        st.markdown("<div class='section-title'>🕸️ Sentence-Level Entity Co-occurrence &amp; Linking</div>",
                    unsafe_allow_html=True)

        pairs = ner["co_occurrence_pairs"]

        if pairs:
            # Pair type filter
            pair_types = sorted(list(set(p["pair_type"] for p in pairs)))
            selected_pair_types = st.multiselect(
                "Filter by Entity Relationship Type:",
                options=pair_types,
                default=pair_types[:min(4, len(pair_types))],
            )

            filtered_pairs = [p for p in pairs if p["pair_type"] in selected_pair_types]

            # Co-occurrence pairs table
            df_pairs = pd.DataFrame([
                {
                    "Entity A": p["entity_1"],
                    "Type A": p["type_1"],
                    "Relationship": "⟷",
                    "Entity B": p["entity_2"],
                    "Type B": p["type_2"],
                    "Relationship Class": p["pair_type"],
                    "Context Sentence": p["sentence"],
                }
                for p in filtered_pairs
            ])

            st.dataframe(
                df_pairs,
                use_container_width=True,
                hide_index=True,
                height=260,
            )

            # ── Relationship Class Frequency Bar Chart ──────────────
            st.markdown("<div class='section-title'>📈 Top Entity Relationship Types</div>",
                        unsafe_allow_html=True)
            pair_counts = Counter(p["pair_type"] for p in pairs).most_common(8)
            fig_rel = go.Figure(go.Bar(
                x=[cnt for _, cnt in pair_counts],
                y=[pt for pt, _ in pair_counts],
                orientation="h",
                marker_color="#818cf8",
                text=[str(cnt) for _, cnt in pair_counts],
                textposition="outside",
            ))
            fig_rel.update_layout(
                paper_bgcolor="#0a0e1a", plot_bgcolor="#111827",
                font_color="#94a3b8",
                margin=dict(l=10, r=40, t=10, b=10),
                height=240,
                xaxis=dict(showgrid=True, gridcolor="#1e293b"),
                yaxis=dict(showgrid=False),
            )
            st.plotly_chart(fig_rel, use_container_width=True)

        else:
            st.info("No multi-entity sentence co-occurrences were found in this snippet.")

        # ── Automated Executive Intelligence Briefing ───────────────
        st.markdown("<div class='section-title'>🛡️ Executive Intelligence Briefing</div>",
                    unsafe_allow_html=True)

        # Build intelligence dossier from detected entities
        who_uniq = list(dict.fromkeys(e["text"] for e in entities_list if e["label"] in ["PERSON", "ORG"]))
        gpe_uniq = list(dict.fromkeys(e["text"] for e in entities_list if e["label"] in ["GPE", "LOC"]))
        threat_uniq = list(dict.fromkeys(e["text"] for e in entities_list if e["label"] in ["SECURITY_THREAT"]))
        tech_uniq = list(dict.fromkeys(e["text"] for e in entities_list if e["label"] in ["TECH_PRODUCT"]))
        money_uniq = list(dict.fromkeys(e["text"] for e in entities_list if e["label"] in ["MONEY", "PERCENT"]))
        law_uniq = list(dict.fromkeys(e["text"] for e in entities_list if e["label"] in ["LAW"]))

        pred_cat = clf["label_names"][clf["lr_pred"]]

        c_int1, c_int2 = st.columns(2)
        with c_int1:
            st.markdown(f"""
            <div class="intel-box">
                <div class="intel-header">🏢 Identified Actors &amp; Organizations</div>
                <div class="intel-item"><b>Primary Parties:</b> {', '.join(who_uniq) if who_uniq else 'Unspecified actors / anonymous syndicate'}</div>
                <div class="intel-item"><b>Jurisdiction / Origin:</b> {', '.join(gpe_uniq) if gpe_uniq else 'Global / Distributed'}</div>
                <div class="intel-item"><b>Intelligence Domain:</b> <span style="color:#a78bfa;">{pred_cat}</span></div>
            </div>
            """, unsafe_allow_html=True)

        with c_int2:
            st.markdown(f"""
            <div class="intel-box">
                <div class="intel-header">⚡ Technical Vectors &amp; Threat Profile</div>
                <div class="intel-item"><b>Detected Threats:</b> {', '.join(threat_uniq) if threat_uniq else 'No overt malicious exploit patterns'}</div>
                <div class="intel-item"><b>Systems &amp; Frameworks:</b> {', '.join(tech_uniq) if tech_uniq else 'General enterprise infrastructure'}</div>
                <div class="intel-item"><b>Financial Exposure / Metrics:</b> {', '.join(money_uniq) if money_uniq else 'Qualitative operational impact'}</div>
                {f'<div class="intel-item"><b>Regulatory Frameworks:</b> {", ".join(law_uniq)}</div>' if law_uniq else ''}
            </div>
            """, unsafe_allow_html=True)

else:
    # ── Landing placeholder ────────────────────────────────────────
    st.markdown("""
    <div style="
        display:flex; flex-direction:column; align-items:center;
        justify-content:center; height:380px;
        background:linear-gradient(135deg,#0f172a,#1e1b4b);
        border:1px solid #312e81; border-radius:16px;
        color:#475569; text-align:center; padding:32px;
    ">
        <div style="font-size:4rem;margin-bottom:16px;">🔍</div>
        <div style="font-size:1.4rem;font-weight:700;color:#64748b;margin-bottom:8px;">
            Ready to Analyse
        </div>
        <div style="font-size:0.9rem;color:#334155;max-width:440px;line-height:1.7;">
            Select an input method from the sidebar, then click
            <strong style="color:#6366f1;">⚡ Analyse Article</strong>
            to run the full NLP &amp; NER pipeline — POS tagging, Word2Vec embeddings,
            dual-model classification, EntityRuler tagging, and relationship intelligence.
        </div>
        <div style="margin-top:24px;display:flex;gap:20px;flex-wrap:wrap;justify-content:center;">
            <span style="color:#3b4d6e;font-size:0.8rem;">🏷️ POS Tagging</span>
            <span style="color:#3b4d6e;font-size:0.8rem;">🧬 Word2Vec</span>
            <span style="color:#3b4d6e;font-size:0.8rem;">🤖 Classification</span>
            <span style="color:#3b4d6e;font-size:0.8rem;">🏷️ SpaCy NER</span>
            <span style="color:#3b4d6e;font-size:0.8rem;">🕸️ Entity Linking</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#1e3a5f;font-size:0.78rem;padding:8px 0;">
    NewsPulse AI · NLP Assignment 4 &amp; 5 ·
    POS Tagging · Word Embeddings · Classification · Named Entity Recognition (NER) · Entity Linking
</div>
""", unsafe_allow_html=True)
