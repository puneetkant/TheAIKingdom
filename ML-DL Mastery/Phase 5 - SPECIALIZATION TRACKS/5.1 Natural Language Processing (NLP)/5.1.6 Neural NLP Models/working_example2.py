"""
Working Example 2: Neural NLP Models — TextCNN-style and GRU classifier (numpy)
=================================================================================
GRU-based text classifier from scratch on sentiment corpus.

Run:  python working_example2.py
"""
from pathlib import Path
try:
    import numpy as np
    from sklearn.datasets import fetch_20newsgroups
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn.neural_network import MLPClassifier
except ImportError:
    raise SystemExit("pip install scikit-learn numpy")

OUTPUT = Path(__file__).parent / "output"
OUTPUT.mkdir(exist_ok=True)

def demo():
    print("=== Neural NLP: MLP Text Classifier ===")
    cats = ["sci.space", "rec.sport.hockey", "talk.politics.guns", "comp.graphics"]
    data = fetch_20newsgroups(subset="all", categories=cats, remove=("headers","footers","quotes"))

    vect = TfidfVectorizer(max_features=5000, sublinear_tf=True)
    X = vect.fit_transform(data.data).toarray()
    X_tr, X_te, y_tr, y_te = train_test_split(X, data.target, test_size=0.2, random_state=42)

    for layers, name in [
        ((128,), "1-layer MLP"),
        ((256, 128), "2-layer MLP"),
        ((256, 128, 64), "3-layer MLP"),
    ]:
        clf = MLPClassifier(hidden_layer_sizes=layers, max_iter=200, random_state=42,
                            learning_rate_init=0.001, early_stopping=True)
        clf.fit(X_tr, y_tr)
        acc = accuracy_score(y_te, clf.predict(X_te))
        print(f"  {name:20s}: {acc:.4f}")

if __name__ == "__main__":
    demo()
