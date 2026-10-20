"""
ner_engine.py
-------------
Named Entity Recognition (NER) & Relationship Intelligence Engine:
  1. SpaCy NLP pipeline with 'en_core_web_sm' (with auto-download fallback)
  2. Custom EntityRuler adding domain-specific patterns:
     - TECH_PRODUCT (GPT-4, Kubernetes, RTX, TensorFlow, ONNX, RAG, etc.)
     - SECURITY_THREAT (DDoS, Zero-Day, Phishing, Ransomware, Prompt Injection, etc.)
     - Standard OntoNotes 5 entities (PERSON, ORG, GPE, MONEY, DATE, PERCENT, CARDINAL)
  3. Sentence-level entity co-occurrence extraction (ORG <-> MONEY, PERSON <-> ORG, etc.)
  4. DisplaCy-style inline HTML highlighted renderer
"""

from __future__ import annotations

import re
import sys
import subprocess
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Any, Optional

# Human-readable entity descriptions
ENTITY_DESCRIPTIONS: Dict[str, str] = {
    "PERSON": "People, including fictional characters",
    "ORG": "Companies, agencies, institutions, and syndicates",
    "GPE": "Countries, cities, states, and sovereign territories",
    "MONEY": "Monetary values, including currencies and amounts",
    "DATE": "Absolute or relative dates or time periods",
    "PERCENT": "Percentage figures and basis point values",
    "CARDINAL": "Numerals that do not fall under another type",
    "TECH_PRODUCT": "Hardware, platforms, models, software frameworks & chips",
    "SECURITY_THREAT": "Vulnerabilities, exploit vectors, malware & cyber attacks",
    "LAW": "Named laws, statutory acts, and formal directives",
    "NORP": "Nationalities or religious or political groups",
}

# Color palettes for UI rendering
ENTITY_COLORS: Dict[str, Dict[str, str]] = {
    "PERSON":          {"bg": "#1e3a5f", "text": "#93c5fd", "border": "#3b82f6"},
    "ORG":             {"bg": "#312e81", "text": "#c4b5fd", "border": "#6366f1"},
    "GPE":             {"bg": "#064e3b", "text": "#6ee7b7", "border": "#10b981"},
    "MONEY":           {"bg": "#713f12", "text": "#fde047", "border": "#eab308"},
    "DATE":            {"bg": "#374151", "text": "#e5e7eb", "border": "#9ca3af"},
    "PERCENT":         {"bg": "#701a75", "text": "#f0abfc", "border": "#d946ef"},
    "CARDINAL":        {"bg": "#1c1917", "text": "#d6d3d1", "border": "#78716c"},
    "TECH_PRODUCT":    {"bg": "#0f766e", "text": "#5eead4", "border": "#14b8a6"},
    "SECURITY_THREAT": {"bg": "#7f1d1d", "text": "#fca5a5", "border": "#ef4444"},
    "LAW":             {"bg": "#431407", "text": "#fdba74", "border": "#f97316"},
    "NORP":            {"bg": "#134e4a", "text": "#99f6e4", "border": "#2dd4bf"},
}

# Domain-specific patterns for EntityRuler
CUSTOM_RULER_PATTERNS = [
    # TECH_PRODUCT patterns
    {"label": "TECH_PRODUCT", "pattern": "GPT-4"},
    {"label": "TECH_PRODUCT", "pattern": "GPT-3.5"},
    {"label": "TECH_PRODUCT", "pattern": "ChatGPT"},
    {"label": "TECH_PRODUCT", "pattern": "Claude"},
    {"label": "TECH_PRODUCT", "pattern": "LLaMA"},
    {"label": "TECH_PRODUCT", "pattern": "Mistral"},
    {"label": "TECH_PRODUCT", "pattern": "Hugging Face"},
    {"label": "TECH_PRODUCT", "pattern": "Kubernetes"},
    {"label": "TECH_PRODUCT", "pattern": "Docker"},
    {"label": "TECH_PRODUCT", "pattern": "TensorFlow"},
    {"label": "TECH_PRODUCT", "pattern": "PyTorch"},
    {"label": "TECH_PRODUCT", "pattern": "RTX"},
    {"label": "TECH_PRODUCT", "pattern": "HBM4"},
    {"label": "TECH_PRODUCT", "pattern": "ONNX"},
    {"label": "TECH_PRODUCT", "pattern": "PyPI"},
    {"label": "TECH_PRODUCT", "pattern": "Active Directory"},
    {"label": "TECH_PRODUCT", "pattern": "gRPC"},
    {"label": "TECH_PRODUCT", "pattern": "Slurm"},
    {"label": "TECH_PRODUCT", "pattern": "FedRAMP"},
    {"label": "TECH_PRODUCT", "pattern": "Blockchain"},
    {"label": "TECH_PRODUCT", "pattern": "FinOps"},
    {"label": "TECH_PRODUCT", "pattern": "TIPS"},
    {"label": "TECH_PRODUCT", "pattern": [{"LOWER": "large"}, {"LOWER": "language"}, {"LOWER": "models"}]},
    {"label": "TECH_PRODUCT", "pattern": [{"LOWER": "large"}, {"LOWER": "language"}, {"LOWER": "model"}]},
    {"label": "TECH_PRODUCT", "pattern": [{"LOWER": "transformer"}, {"LOWER": "architectures"}]},
    {"label": "TECH_PRODUCT", "pattern": [{"LOWER": "neural"}, {"LOWER": "inference"}, {"LOWER": "accelerators"}]},
    {"label": "TECH_PRODUCT", "pattern": [{"LOWER": "5g"}, {"LOWER": "private"}, {"LOWER": "network"}]},
    {"label": "TECH_PRODUCT", "pattern": [{"LOWER": "silicon"}, {"LOWER": "photonics"}]},
    {"label": "TECH_PRODUCT", "pattern": [{"LOWER": "rag"}, {"LOWER": "pipelines"}]},
    {"label": "TECH_PRODUCT", "pattern": [{"LOWER": "rag"}, {"LOWER": "pipeline"}]},
    {"label": "TECH_PRODUCT", "pattern": "RAG"},
    {"label": "TECH_PRODUCT", "pattern": "LLM"},
    {"label": "TECH_PRODUCT", "pattern": "LLMs"},
    {"label": "TECH_PRODUCT", "pattern": "GPU"},
    {"label": "TECH_PRODUCT", "pattern": "GPUs"},
    {"label": "TECH_PRODUCT", "pattern": "Bitcoin"},
    {"label": "TECH_PRODUCT", "pattern": "CBDC"},

    # SECURITY_THREAT patterns
    {"label": "SECURITY_THREAT", "pattern": "DDoS"},
    {"label": "SECURITY_THREAT", "pattern": "Zero-Day"},
    {"label": "SECURITY_THREAT", "pattern": "zero-day"},
    {"label": "SECURITY_THREAT", "pattern": "Ransomware"},
    {"label": "SECURITY_THREAT", "pattern": "ransomware"},
    {"label": "SECURITY_THREAT", "pattern": "Malware"},
    {"label": "SECURITY_THREAT", "pattern": "malware"},
    {"label": "SECURITY_THREAT", "pattern": "Phishing"},
    {"label": "SECURITY_THREAT", "pattern": "phishing"},
    {"label": "SECURITY_THREAT", "pattern": "Spear-Phishing"},
    {"label": "SECURITY_THREAT", "pattern": "spear-phishing"},
    {"label": "SECURITY_THREAT", "pattern": "Keylogger"},
    {"label": "SECURITY_THREAT", "pattern": "keylogger"},
    {"label": "SECURITY_THREAT", "pattern": "Trojan"},
    {"label": "SECURITY_THREAT", "pattern": "trojan"},
    {"label": "SECURITY_THREAT", "pattern": "trojanized"},
    {"label": "SECURITY_THREAT", "pattern": "Backdoor"},
    {"label": "SECURITY_THREAT", "pattern": "backdoors"},
    {"label": "SECURITY_THREAT", "pattern": "Deepfake"},
    {"label": "SECURITY_THREAT", "pattern": "deepfake"},
    {"label": "SECURITY_THREAT", "pattern": [{"LOWER": "prompt"}, {"LOWER": "injection"}]},
    {"label": "SECURITY_THREAT", "pattern": [{"LOWER": "prompt-injection"}]},
    {"label": "SECURITY_THREAT", "pattern": [{"LOWER": "model"}, {"LOWER": "theft"}]},
    {"label": "SECURITY_THREAT", "pattern": [{"LOWER": "model"}, {"LOWER": "poisoning"}]},
    {"label": "SECURITY_THREAT", "pattern": [{"LOWER": "poisoned"}, {"LOWER": "gradient"}]},
    {"label": "SECURITY_THREAT", "pattern": [{"LOWER": "adversarial"}, {"LOWER": "gradient"}]},
    {"label": "SECURITY_THREAT", "pattern": [{"LOWER": "supply-chain"}, {"LOWER": "attack"}]},
    {"label": "SECURITY_THREAT", "pattern": [{"LOWER": "kill-chain"}, {"LOWER": "attack"}]},
    {"label": "SECURITY_THREAT", "pattern": [{"LOWER": "steganographic"}, {"LOWER": "payloads"}]},
    {"label": "SECURITY_THREAT", "pattern": [{"LOWER": "c2"}, {"LOWER": "channels"}]},
    {"label": "SECURITY_THREAT", "pattern": [{"LOWER": "c2"}, {"LOWER": "communications"}]},

    # LAW / ACT patterns
    {"label": "LAW", "pattern": "EU AI Act"},
    {"label": "LAW", "pattern": "AI Act"},
    {"label": "LAW", "pattern": "GDPR"},
    {"label": "LAW", "pattern": "Digital Markets Act"},
    {"label": "LAW", "pattern": "Digital Personal Data Protection Act"},
    {"label": "LAW", "pattern": "Children's Online Safety Act"},
    {"label": "LAW", "pattern": "Biometric Information Privacy Act"},
    {"label": "LAW", "pattern": "Stablecoin Transparency and Reliability Act"},
    {"label": "LAW", "pattern": "AI Liability Directive"},
]

# Singleton NLP instance
_SPACY_NLP = None


def _load_spacy_model():
    """Load or download en_core_web_sm with custom EntityRuler."""
    global _SPACY_NLP
    if _SPACY_NLP is not None:
        return _SPACY_NLP

    try:
        import spacy
    except ImportError:
        return None

    try:
        nlp = spacy.load("en_core_web_sm")
    except Exception:
        # Try downloading automatically
        try:
            from spacy.cli import download
            download("en_core_web_sm")
            nlp = spacy.load("en_core_web_sm")
        except Exception:
            try:
                subprocess.run(
                    [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
                    check=True,
                    capture_output=True,
                )
                nlp = spacy.load("en_core_web_sm")
            except Exception:
                return None

    # Add custom EntityRuler before the statistical NER so domain patterns take precedence
    try:
        if "entity_ruler" not in nlp.pipe_names:
            ruler = nlp.add_pipe("entity_ruler", before="ner")
            ruler.add_patterns(CUSTOM_RULER_PATTERNS)
    except Exception:
        pass

    _SPACY_NLP = nlp
    return _SPACY_NLP


# ══════════════════════════════════════════════════════════════════════════════
# FALLBACK REGEX NER (When spaCy is installing or offline)
# ══════════════════════════════════════════════════════════════════════════════

def _fallback_extract_entities(text: str) -> List[Dict[str, Any]]:
    """Heuristic rule-based NER fallback when spaCy model is loading."""
    entities = []

    # Regex rules
    patterns = [
        ("MONEY", r"(\$\s*\d+(?:\.\d+)?(?:\s*(?:million|billion|trillion))?|\d+(?:\.\d+)?\s*(?:million|billion|trillion)\s*dollars|\d+(?:\.\d+)?\s*(?:basis\s*points|crore\s*rupees|euros))", "Monetary figure"),
        ("PERCENT", r"(\d+(?:\.\d+)?\s*(?:percent|%))", "Percentage figure"),
        ("DATE", r"\b(18\s*months|14\s*minutes|72\s*hours|48\s*hours|six\s*weeks|first\s*quarter|three\s*months|five\s*years|12\s*months|90\s*days|quarterly|annually)\b", "Date or timeframe"),
        ("SECURITY_THREAT", r"\b(malware|ransomware|spear-phishing|phishing|zero-day|CVE-\d{4}-\d+|prompt[- ]injection|backdoors?|trojan(?:ized)?|keylogger|deepfake|kill-chain|c2\s*channels?)\b", "Security threat vector"),
        ("TECH_PRODUCT", r"\b(GPT-4|Kubernetes|RTX|TensorFlow|PyTorch|ONNX|Docker|HBM4|PyPI|Active Directory|gRPC|Slurm|FedRAMP|Blockchain|FinOps|TIPS|Bitcoin|CBDC|RAG|LLMs?|GPUs?)\b", "Technology platform or product"),
        ("ORG", r"\b(CrowdStrike|MIT CSAIL|Hugging Face|CISA|NIST|Federal Reserve|Bank of England|IMF|Goldman Sachs|Google Project Zero|Stanford|European Union|European Commission|FTC|Senate Banking Committee|Department of Commerce)\b", "Organization or agency"),
        ("GPE", r"\b(US|British|European Union|India|Brazil|Norway|Hiroshima)\b", "Geopolitical entity"),
        ("LAW", r"\b(EU AI Act|AI Act|GDPR|Digital Markets Act|Digital Personal Data Protection Act|Children's Online Safety Act|Biometric Information Privacy Act)\b", "Statutory law or act"),
    ]

    for label, regex_str, desc in patterns:
        for match in re.finditer(regex_str, text, re.IGNORECASE):
            entities.append({
                "text": match.group(0),
                "label": label,
                "description": ENTITY_DESCRIPTIONS.get(label, desc),
                "start_char": match.start(),
                "end_char": match.end(),
                "confidence_type": "Rule-Based EntityRuler",
            })

    # Sort by start_char and drop overlaps
    entities.sort(key=lambda x: (x["start_char"], -x["end_char"]))
    deduped = []
    last_end = -1
    for ent in entities:
        if ent["start_char"] >= last_end:
            deduped.append(ent)
            last_end = ent["end_char"]

    return deduped


# ══════════════════════════════════════════════════════════════════════════════
# MAIN EXTRACT_ENTITIES FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def extract_entities(text: str) -> Dict[str, Any]:
    """
    Extract Named Entities using spaCy (with custom EntityRuler) or fallback.

    Parameters
    ----------
    text : str  Raw news or incident text snippet.

    Returns
    -------
    dict:
        entities            – list[dict] with keys:
                              {text, label, description, start_char, end_char, confidence_type}
        entity_counts       – dict[label: count]
        co_occurrence_pairs – list[dict] of sentence-level entity links:
                              {entity_1, type_1, entity_2, type_2, pair_type, sentence}
        sentences           – list[str]
        ruler_pattern_count – int
        engine_type         – str ('spaCy en_core_web_sm + EntityRuler' or fallback)
    """
    nlp = _load_spacy_model()

    if nlp is not None:
        doc = nlp(text)
        entities = []
        ruler_matched_texts = set()

        # Check entity ruler hits
        for ent in doc.ents:
            # Check if entity comes from ruler or statistical model
            # Rules in our ruler match our custom domain labels or exact patterns
            is_ruler = (
                ent.label_ in ["TECH_PRODUCT", "SECURITY_THREAT", "LAW"] or
                any(p["pattern"] == ent.text for p in CUSTOM_RULER_PATTERNS if isinstance(p.get("pattern"), str))
            )
            conf_type = "Rule-Based EntityRuler" if is_ruler else "Statistical Model (OntoNotes 5)"

            entities.append({
                "text": ent.text,
                "label": ent.label_,
                "description": ENTITY_DESCRIPTIONS.get(ent.label_, "Named entity"),
                "start_char": ent.start_char,
                "end_char": ent.end_char,
                "confidence_type": conf_type,
            })

        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        engine_type = "spaCy en_core_web_sm + EntityRuler"

        # Sentence-level co-occurrences
        co_occurrence_pairs = []
        for sent in doc.sents:
            sent_ents = [
                {"text": e.text, "label": e.label_}
                for e in sent.ents
            ]
            # Remove duplicate entity texts within the same sentence
            seen_texts = set()
            unique_sent_ents = []
            for e in sent_ents:
                if e["text"].lower() not in seen_texts:
                    seen_texts.add(e["text"].lower())
                    unique_sent_ents.append(e)

            # Generate distinct unordered pairs
            n = len(unique_sent_ents)
            for i in range(n):
                for j in range(i + 1, n):
                    e1 = unique_sent_ents[i]
                    e2 = unique_sent_ents[j]
                    pair_label = f"{e1['label']} ↔ {e2['label']}"
                    co_occurrence_pairs.append({
                        "entity_1": e1["text"],
                        "type_1": e1["label"],
                        "entity_2": e2["text"],
                        "type_2": e2["label"],
                        "pair_type": pair_label,
                        "sentence": sent.text.strip(),
                    })

    else:
        # Fallback
        entities = _fallback_extract_entities(text)
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        engine_type = "Domain Heuristic Rule-Based Fallback"

        co_occurrence_pairs = []
        for sent in sentences:
            sent_ents = [
                ent for ent in entities
                if ent["text"] in sent
            ]
            seen_texts = set()
            unique_sent_ents = []
            for e in sent_ents:
                if e["text"].lower() not in seen_texts:
                    seen_texts.add(e["text"].lower())
                    unique_sent_ents.append(e)

            n = len(unique_sent_ents)
            for i in range(n):
                for j in range(i + 1, n):
                    e1 = unique_sent_ents[i]
                    e2 = unique_sent_ents[j]
                    co_occurrence_pairs.append({
                        "entity_1": e1["text"],
                        "type_1": e1["label"],
                        "entity_2": e2["text"],
                        "type_2": e2["label"],
                        "pair_type": f"{e1['label']} ↔ {e2['label']}",
                        "sentence": sent,
                    })

    # Entity counts
    counts = Counter(e["label"] for e in entities)

    return {
        "entities": entities,
        "entity_counts": dict(counts),
        "co_occurrence_pairs": co_occurrence_pairs,
        "sentences": sentences,
        "ruler_pattern_count": len(CUSTOM_RULER_PATTERNS),
        "engine_type": engine_type,
    }


# ══════════════════════════════════════════════════════════════════════════════
# DISPLACY-STYLE HIGHLIGHTED HTML RENDERER
# ══════════════════════════════════════════════════════════════════════════════

def render_displacy_html(
    text: str,
    entities: List[Dict[str, Any]],
    selected_labels: Optional[List[str]] = None,
) -> str:
    """
    Generate inline DisplaCy-style HTML with colored badge spans.
    """
    if not entities:
        return f"<div style='line-height:2.0;font-size:0.95rem;color:#cbd5e1;'>{text}</div>"

    if selected_labels is not None:
        filtered_ents = [e for e in entities if e["label"] in selected_labels]
    else:
        filtered_ents = entities

    # Sort entities by start_char
    sorted_ents = sorted(filtered_ents, key=lambda x: (x["start_char"], -x["end_char"]))

    # Resolve overlapping spans
    non_overlapping = []
    last_end = -1
    for ent in sorted_ents:
        if ent["start_char"] >= last_end:
            non_overlapping.append(ent)
            last_end = ent["end_char"]

    html_parts = []
    current_idx = 0

    for ent in non_overlapping:
        s = ent["start_char"]
        e = ent["end_char"]
        lbl = ent["label"]
        txt = ent["text"]

        # Append plain text preceding this entity
        if s > current_idx:
            html_parts.append(text[current_idx:s])

        # Get color scheme
        c = ENTITY_COLORS.get(lbl, {"bg": "#1e293b", "text": "#cbd5e1", "border": "#475569"})

        badge_html = (
            f'<mark style="background:{c["bg"]}; color:{c["text"]}; '
            f'border:1px solid {c["border"]}; border-radius:6px; padding:2px 6px; '
            f'margin:0 2px; font-weight:600; display:inline-block; line-height:1.4;">'
            f'{txt}'
            f'<span style="font-size:0.65rem; text-transform:uppercase; '
            f'background:rgba(0,0,0,0.35); border-radius:4px; padding:1px 4px; '
            f'margin-left:5px; letter-spacing:0.05em; font-family:\'JetBrains Mono\', monospace;'
            f'vertical-align:middle; color:#fff;">{lbl}</span>'
            f'</mark>'
        )
        html_parts.append(badge_html)
        current_idx = e

    # Append remaining trailing text
    if current_idx < len(text):
        html_parts.append(text[current_idx:])

    full_html = "".join(html_parts)
    return (
        f'<div style="background:#0d1117; border:1px solid #21262d; border-radius:12px; '
        f'padding:20px 24px; line-height:2.4; font-size:0.95rem; color:#e2e8f0; '
        f'font-family:\'Inter\', sans-serif;">'
        f'{full_html}'
        f'</div>'
    )
