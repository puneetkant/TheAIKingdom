"""
Working Example 2: Naive Bayes — Gaussian NB, Bernoulli NB, Laplace Smoothing
================================================================================
Text-style (Bernoulli) classification and Gaussian NB on Cal Housing.

Run:  python working_example2.py
"""
from pathlib import Path
try:
    import numpy as np
    from sklearn.datasets import fetch_california_housing, fetch_20newsgroups, load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB, ComplementNB
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import classification_report, accuracy_score
    from sklearn.pipeline import make_pipeline
except ImportError:
    raise SystemExit("pip install numpy scikit-learn")

def demo_gaussian_nb_housing():
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
            print(f"  alpha={alpha} class={cls}: {dict(zip(vocab, probs.round(3)))}")

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


def demo_gaussian_nb():
    """GaussianNB on iris: accuracy and top-1 predicted class probabilities."""
    print("\n=== Gaussian NB on Iris ===")
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    y_pred = gnb.predict(X_test)
    y_proba = gnb.predict_proba(X_test)
    acc = accuracy_score(y_test, y_pred)
    top1 = y_proba.max(axis=1)
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Top-1 prob  mean={top1.mean():.4f}  min={top1.min():.4f}  max={top1.max():.4f}")
    print(f"  First 5 predictions : {y_pred[:5].tolist()}")
    print(f"  First 5 top-1 probs : {[round(float(p), 4) for p in top1[:5]]}")


def demo_multinomial_nb():
    """MultinomialNB on 20newsgroups (4 categories), per-class accuracy."""
    print("\n=== Multinomial NB on 20 Newsgroups (4 categories) ===")
    cats = ["sci.space", "comp.graphics", "rec.sport.baseball", "talk.politics.misc"]
    try:
        train = fetch_20newsgroups(subset="train", categories=cats,
                                   remove=("headers", "footers", "quotes"))
        test  = fetch_20newsgroups(subset="test",  categories=cats,
                                   remove=("headers", "footers", "quotes"))
        pipe = make_pipeline(TfidfVectorizer(max_features=5000), MultinomialNB())
        pipe.fit(train.data, train.target)
        y_pred = pipe.predict(test.data)
        y_true = np.array(test.target)
        print(f"  Overall accuracy: {accuracy_score(y_true, y_pred):.4f}")
        for i, cat in enumerate(cats):
            mask = y_true == i
            per_acc = accuracy_score(y_true[mask], y_pred[mask])
            print(f"  Class {cat}: {per_acc:.4f}")
    except Exception as e:
        print(f"  Skipped (dataset unavailable): {e}")


def demo_complement_nb():
    """ComplementNB vs MultinomialNB on same 20newsgroups data, compare accuracy."""
    print("\n=== Complement NB vs Multinomial NB ===")
    cats = ["sci.space", "comp.graphics", "rec.sport.baseball", "talk.politics.misc"]
    try:
        train = fetch_20newsgroups(subset="train", categories=cats,
                                   remove=("headers", "footers", "quotes"))
        test  = fetch_20newsgroups(subset="test",  categories=cats,
                                   remove=("headers", "footers", "quotes"))
        for name, clf in [("MultinomialNB", MultinomialNB()),
                           ("ComplementNB ", ComplementNB())]:
            pipe = make_pipeline(TfidfVectorizer(max_features=5000), clf)
            pipe.fit(train.data, train.target)
            acc = accuracy_score(test.target, pipe.predict(test.data))
            print(f"  {name}: accuracy={acc:.4f}")
    except Exception as e:
        print(f"  Skipped (dataset unavailable): {e}")


def demo_prior_vs_likelihood():
    """Manually show prior * likelihood drives NB predictions (toy example)."""
    print("\n=== Prior x Likelihood in Naive Bayes (toy example) ===")
    # 2 classes: Ham (0) and Spam (1)
    # Features: [contains_money, contains_urgent, is_short]
    n_ham, n_spam = 40, 10
    total = n_ham + n_spam
    prior_ham  = n_ham  / total   # 0.80
    prior_spam = n_spam / total   # 0.20
    # Training counts (feature=1 given class) with Laplace smoothing alpha=1
    ham_counts  = np.array([8,  4, 20], dtype=float)
    spam_counts = np.array([8,  9,  3], dtype=float)
    alpha = 1.0
    p_feat_ham  = (ham_counts  + alpha) / (n_ham  + 2 * alpha)
    p_feat_spam = (spam_counts + alpha) / (n_spam + 2 * alpha)
    feat_names = ["contains_money", "contains_urgent", "is_short"]
    print(f"  Prior(Ham)={prior_ham:.2f}  Prior(Spam)={prior_spam:.2f}")
    print("  {:<20}  P(f|Ham)  P(f|Spam)".format("Feature"))
    for fn, ph, ps in zip(feat_names, p_feat_ham, p_feat_spam):
        print(f"  {fn:<20}   {ph:.4f}    {ps:.4f}")
    # Test sample: money=1, urgent=1, is_short=0
    x = np.array([1, 1, 0], dtype=float)
    sample = {fn: int(v) for fn, v in zip(feat_names, x)}
    print(f"  Test sample: {sample}")

    def log_posterior(prior, p_f, xi):
        return np.log(prior) + np.sum(xi * np.log(p_f) + (1 - xi) * np.log(1 - p_f))

    lh = log_posterior(prior_ham,  p_feat_ham,  x)
    ls = log_posterior(prior_spam, p_feat_spam, x)
    print(f"  Log-score Ham={lh:.4f}  Spam={ls:.4f}")
    pred = "Spam" if ls > lh else "Ham"
    print(f"  Prediction: {pred}")


if __name__ == "__main__":
    demo_gaussian_nb_housing()
    demo_laplace_smoothing()
    demo_text_classification()
    demo_gaussian_nb()
    demo_multinomial_nb()
    demo_complement_nb()
    demo_prior_vs_likelihood()
