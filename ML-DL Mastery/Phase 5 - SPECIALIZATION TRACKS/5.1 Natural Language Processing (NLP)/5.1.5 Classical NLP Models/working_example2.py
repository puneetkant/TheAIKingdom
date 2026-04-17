"""
Working Example 2: Classical NLP Models — Naive Bayes, Logistic Regression, SVM text classification
======================================================================================================
Text classification on 20 newsgroups subset with TF-IDF features.

Run:  python working_example2.py
"""
from pathlib import Path
try:
    import numpy as np
    from sklearn.datasets import fetch_20newsgroups
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    from sklearn.metrics import classification_report, accuracy_score
except ImportError:
    raise SystemExit("pip install scikit-learn numpy")

OUTPUT = Path(__file__).parent / "output"
OUTPUT.mkdir(exist_ok=True)

def demo():
    print("=== Classical NLP Text Classification ===")
    cats = ["sci.space", "rec.sport.hockey", "talk.politics.guns", "comp.graphics"]
    train = fetch_20newsgroups(subset="train", categories=cats, remove=("headers","footers","quotes"))
    test  = fetch_20newsgroups(subset="test",  categories=cats, remove=("headers","footers","quotes"))

    vect = TfidfVectorizer(max_features=10000, sublinear_tf=True, ngram_range=(1,2))
    X_tr = vect.fit_transform(train.data)
    X_te = vect.transform(test.data)

    results = {}
    for name, model in [
        ("Naive Bayes", MultinomialNB(alpha=0.1)),
        ("Logistic Regression", LogisticRegression(max_iter=1000, C=1.0)),
        ("LinearSVC", LinearSVC(max_iter=2000, C=0.5)),
    ]:
        model.fit(X_tr, train.target)
        acc = accuracy_score(test.target, model.predict(X_te))
        results[name] = acc
        print(f"\n  {name}: {acc:.4f}")

    best = max(results, key=results.get)
    print(f"\nBest model: {best} ({results[best]:.4f})")
    (OUTPUT / "nlp_classical_results.txt").write_text(
        "\n".join(f"{k}: {v:.4f}" for k,v in results.items()))
    print("Saved nlp_classical_results.txt")

if __name__ == "__main__":
    demo()
