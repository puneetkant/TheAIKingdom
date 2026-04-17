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

if __name__ == "__main__":
    if HF_AVAILABLE:
        demo_hf()
    else:
        demo_sklearn_proxy()
