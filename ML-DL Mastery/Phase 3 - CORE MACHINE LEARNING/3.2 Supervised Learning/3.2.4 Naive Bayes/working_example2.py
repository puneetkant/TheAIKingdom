"""
Working Example 2: Naive Bayes — Gaussian NB, Bernoulli NB, Laplace Smoothing
================================================================================
Text-style (Bernoulli) classification and Gaussian NB on Cal Housing.

Run:  python working_example2.py
"""
from pathlib import Path
try:
    import numpy as np
    from sklearn.datasets import fetch_california_housing, fetch_20newsgroups
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import classification_report, accuracy_score
    from sklearn.pipeline import make_pipeline
except ImportError:
    raise SystemExit("pip install numpy scikit-learn")

def demo_gaussian_nb():
    print("=== Gaussian Naive Bayes (Cal Housing binary) ===")
    h = fetch_california_housing()
    X, y = h.data, (h.target > np.median(h.target)).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    y_pred = gnb.predict(X_test)
    print(f"  Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred, digits=4))

def demo_laplace_smoothing():
    print("=== Laplace Smoothing (manual Bernoulli NB) ===")
    # Docs: D1=["cat dog"], D2=["cat fish"], D3=["dog fish"]
    vocab = ["cat", "dog", "fish"]
    # Class 0: D1,D2 ; Class 1: D3
    X = np.array([[1,1,0],[1,0,1],[0,1,1]])
    y = np.array([0,0,1])
    for alpha in [0, 0.5, 1.0]:
        # P(word|class) with smoothing
        for cls in [0, 1]:
            mask = y == cls
            counts = X[mask].sum(axis=0) + alpha
            total  = X[mask].sum() + alpha * len(vocab)
            probs  = counts / total
            print(f"  α={alpha} class={cls}: {dict(zip(vocab, probs.round(3)))}")

def demo_text_classification():
    print("\n=== Text Classification with Multinomial NB (20 Newsgroups) ===")
    cats = ["sci.space", "comp.graphics", "rec.sport.baseball"]
    try:
        train = fetch_20newsgroups(subset="train", categories=cats, remove=("headers","footers","quotes"))
        test  = fetch_20newsgroups(subset="test",  categories=cats, remove=("headers","footers","quotes"))
        pipe  = make_pipeline(TfidfVectorizer(max_features=5000), MultinomialNB())
        pipe.fit(train.data, train.target)
        y_pred = pipe.predict(test.data)
        print(f"  Accuracy: {accuracy_score(test.target, y_pred):.4f}")
        print(classification_report(test.target, y_pred, target_names=cats, digits=3))
    except Exception as e:
        print(f"  Skipped (dataset unavailable): {e}")

if __name__ == "__main__":
    demo_gaussian_nb()
    demo_laplace_smoothing()
    demo_text_classification()
