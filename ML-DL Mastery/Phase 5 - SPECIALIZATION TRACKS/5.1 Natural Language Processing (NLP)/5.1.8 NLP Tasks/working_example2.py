"""
Working Example 2: NLP Tasks — NER, POS tagging, text classification pipeline
===============================================================================
Rule-based NER + POS tagging demo + sklearn classification benchmark.

Run:  python working_example2.py
"""
from pathlib import Path
import re

try:
    from sklearn.datasets import fetch_20newsgroups
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report
except ImportError:
    raise SystemExit("pip install scikit-learn")

OUTPUT = Path(__file__).parent / "output"
OUTPUT.mkdir(exist_ok=True)

SENTENCES = [
    "Apple Inc. was founded by Steve Jobs in Cupertino, California.",
    "Google announced its new AI model in San Francisco on Monday.",
    "The European Central Bank raised interest rates by 25 basis points.",
]

ENTITY_PATTERNS = {
    "ORG": [r"\b(Apple Inc\.|Google|European Central Bank)\b"],
    "PER": [r"\b(Steve Jobs)\b"],
    "LOC": [r"\b(Cupertino|California|San Francisco)\b"],
}

def simple_ner(text):
    entities = []
    for label, patterns in ENTITY_PATTERNS.items():
        for pat in patterns:
            for m in re.finditer(pat, text):
                entities.append((m.group(), label, m.start(), m.end()))
    return sorted(entities, key=lambda x: x[2])

def demo_ner():
    print("=== Rule-based NER ===")
    for sent in SENTENCES:
        entities = simple_ner(sent)
        print(f"\n  Text: {sent[:60]}")
        for ent, label, s, e in entities:
            print(f"    [{label}] '{ent}'")

def demo_classification():
    print("\n=== Multi-class Text Classification (4 categories) ===")
    cats = ["sci.med", "sci.space", "rec.sport.hockey", "comp.graphics"]
    tr = fetch_20newsgroups(subset="train", categories=cats, remove=("headers","footers","quotes"))
    te = fetch_20newsgroups(subset="test",  categories=cats, remove=("headers","footers","quotes"))
    vect = TfidfVectorizer(max_features=8000, sublinear_tf=True)
    clf  = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(vect.fit_transform(tr.data), tr.target)
    preds = clf.predict(vect.transform(te.data))
    print(classification_report(te.target, preds, target_names=cats))

def demo_summarisation():
    """Extractive summarisation via TF-IDF sentence scoring."""
    print("\n=== Extractive Summarisation ===")
    doc = (
        "Natural language processing (NLP) is a subfield of linguistics and AI. "
        "NLP focuses on the interaction between computers and human language. "
        "Key tasks include translation, summarisation, and sentiment analysis. "
        "Modern NLP relies heavily on large language models such as BERT and GPT. "
        "These models are pre-trained on vast text corpora and fine-tuned for tasks."
    )
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", doc) if s.strip()]
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    vect = TfidfVectorizer()
    tfidf_mat = vect.fit_transform(sentences)
    sim = cosine_similarity(tfidf_mat)
    scores = sim.mean(axis=1)
    top_k = np.argsort(scores)[::-1][:2]
    summary = " ".join(sentences[i] for i in sorted(top_k))
    print(f"  Summary: {summary}")


def demo_pos_tagging():
    """Simple rule-based POS tagging using suffix patterns."""
    print("\n=== Rule-based POS Tagging ===")
    rules = [
        (r".*ing$",   "VBG"),
        (r".*ed$",    "VBD"),
        (r".*ly$",    "RB"),
        (r".*ion$",   "NN"),
        (r".*ness$",  "NN"),
        (r"^[A-Z]",   "NNP"),
        (r"^\d+$",    "CD"),
    ]
    tokens = ["Apple", "announced", "quickly", "rising", "innovation", "in", "2024"]
    for tok in tokens:
        tag = "NN"
        for pat, t in rules:
            if re.match(pat, tok):
                tag = t; break
        print(f"  {tok:15s} -> {tag}")


if __name__ == "__main__":
    demo_ner()
    demo_classification()
    demo_summarisation()
    demo_pos_tagging()
