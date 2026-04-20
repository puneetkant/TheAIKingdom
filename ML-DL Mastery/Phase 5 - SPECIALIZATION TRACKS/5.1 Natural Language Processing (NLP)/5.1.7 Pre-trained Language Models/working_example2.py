"""
Working Example 2: Pre-trained Language Models — BERT fine-tuning pattern
==========================================================================
Demonstrates HuggingFace Transformers pipeline patterns (API-level).

Run:  python working_example2.py
"""
from pathlib import Path
try:
    from transformers import pipeline
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

try:
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
except ImportError:
    pass

OUTPUT = Path(__file__).parent / "output"
OUTPUT.mkdir(exist_ok=True)

TEXTS = [
    "This movie was absolutely fantastic!",
    "Terrible experience, would not recommend.",
    "Okay film, nothing special.",
    "Best book I have ever read in my life.",
    "Completely boring and disappointing.",
]

def demo_hf():
    print("=== HuggingFace Transformers Pipeline ===")
    clf = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    for text in TEXTS:
        result = clf(text)[0]
        print(f"  [{result['label']:8s} {result['score']:.3f}] {text[:50]}")

def demo_sklearn_proxy():
    print("=== Sklearn proxy (no GPU/HF required) ===")
    labels = [1, 0, 1, 1, 0]
    vect = TfidfVectorizer()
    X = vect.fit_transform(TEXTS).toarray()
    clf = LogisticRegression().fit(X, labels)
    preds = clf.predict(X)
    for text, pred in zip(TEXTS, preds):
        print(f"  [{['NEG','POS'][pred]}] {text[:50]}")

def demo_fine_tune_pattern():
    """Illustrate the typical fine-tuning pattern: freeze base, train head."""
    print("\n=== Fine-tuning Pattern (sklearn proxy) ===")
    from sklearn.datasets import fetch_20newsgroups
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    # Simulate 'frozen pre-trained features' via TF-IDF (feature extractor)
    cats = ["sci.med", "sci.space"]
    data = fetch_20newsgroups(subset="train", categories=cats,
                               remove=("headers", "footers", "quotes"))
    # Phase 1: pretrained extractor (TF-IDF = frozen)
    tfidf = TfidfVectorizer(max_features=5000, sublinear_tf=True)
    X = tfidf.fit_transform(data.data)

    # Phase 2: fine-tune head only (LR = trainable head)
    from sklearn.linear_model import LogisticRegression
    head = LogisticRegression(max_iter=500, C=1.0)
    scores = cross_val_score(head, X, data.target, cv=5)
    print(f"  Fine-tune head (5-fold): {scores.mean():.4f} ± {scores.std():.4f}")

    # Phase 3: full fine-tune (both extractor and head are tuned end-to-end)
    pipe = Pipeline([("tfidf", TfidfVectorizer(max_features=5000, sublinear_tf=True)),
                     ("clf", LogisticRegression(max_iter=500))])
    scores_full = cross_val_score(pipe, data.data, data.target, cv=5)
    print(f"  Full fine-tune pipeline: {scores_full.mean():.4f} ± {scores_full.std():.4f}")


def demo_plm_comparison():
    """Compare key pre-trained LM families."""
    print("\n=== Pre-trained LM Families ===")
    models = [
        ("BERT",        "Encoder", "Masked LM + NSP",     "Classification, NER, QA"),
        ("GPT-2",       "Decoder", "Causal LM",           "Text generation"),
        ("T5",          "Enc-Dec", "Text-to-text",        "Translation, summarisation"),
        ("RoBERTa",     "Encoder", "Masked LM (no NSP)",  "Classification, NLU"),
        ("DistilBERT",  "Encoder", "Knowledge distil.",  "Fast inference"),
        ("LLaMA-3",     "Decoder", "Causal LM (SFT/RLHF)","Chat, reasoning"),
    ]
    print(f"  {'Model':14s} {'Type':10s} {'Objective':25s} {'Use Case'}")
    print("  " + "-"*70)
    for r in models:
        print(f"  {r[0]:14s} {r[1]:10s} {r[2]:25s} {r[3]}")


if __name__ == "__main__":
    if HF_AVAILABLE:
        demo_hf()
    else:
        demo_sklearn_proxy()
    demo_fine_tune_pattern()
    demo_plm_comparison()
