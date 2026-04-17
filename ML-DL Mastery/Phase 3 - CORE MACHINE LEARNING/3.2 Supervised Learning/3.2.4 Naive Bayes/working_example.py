"""
Working Example: Naive Bayes
Covers Bayes' theorem for classification, Gaussian NB, Multinomial NB,
Bernoulli NB, smoothing (Laplace), text classification, and assumptions.
"""
import numpy as np
from scipy.stats import norm
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.datasets import load_iris, make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
import os


# ── 1. Naive Bayes from scratch ────────────────────────────────────────────────
def naive_bayes_from_scratch():
    print("=== Naive Bayes From Scratch (Gaussian) ===")
    print("  P(y|x) ∝ P(y) · Π_j P(x_j|y)    [class-conditional independence]")

    class GaussianNB_scratch:
        def fit(self, X, y):
            self.classes = np.unique(y)
            self.priors  = {}
            self.mu      = {}
            self.sigma   = {}
            for c in self.classes:
                X_c            = X[y == c]
                self.priors[c] = len(X_c) / len(y)
                self.mu[c]     = X_c.mean(0)
                self.sigma[c]  = X_c.var(0) + 1e-9   # avoid zero
            return self

        def _log_likelihood(self, X, c):
            mu, var = self.mu[c], self.sigma[c]
            return -0.5 * np.sum(np.log(2*np.pi*var) + (X - mu)**2 / var, axis=1)

        def predict_proba(self, X):
            log_probs = np.array([
                np.log(self.priors[c]) + self._log_likelihood(X, c)
                for c in self.classes
            ]).T  # (n, n_classes)
            # Normalise (log-sum-exp for stability)
            log_probs -= log_probs.max(axis=1, keepdims=True)
            probs = np.exp(log_probs)
            return probs / probs.sum(axis=1, keepdims=True)

        def predict(self, X):
            return self.classes[self.predict_proba(X).argmax(axis=1)]

    iris = load_iris()
    X, y = iris.data, iris.target
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)

    model = GaussianNB_scratch().fit(X_tr, y_tr)
    acc   = (model.predict(X_te) == y_te).mean()
    sk_model = GaussianNB().fit(X_tr, y_tr)
    sk_acc   = sk_model.score(X_te, y_te)
    print(f"  Scratch accuracy: {acc:.4f}")
    print(f"  sklearn accuracy: {sk_acc:.4f}")

    # Show learned parameters for class 0
    c = 0
    print(f"\n  Class={c} prior={model.priors[c]:.4f}")
    print(f"    μ = {model.mu[c].round(4)}")
    print(f"    σ²= {model.sigma[c].round(4)}")


# ── 2. Gaussian NB on continuous features ────────────────────────────────────
def gaussian_nb_demo():
    print("\n=== Gaussian Naive Bayes (sklearn) ===")
    X, y = make_classification(n_samples=500, n_features=6, n_informative=4,
                                random_state=1)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)
    model = GaussianNB()
    cv    = cross_val_score(model, X, y, cv=5)
    model.fit(X_tr, y_tr)
    print(f"  5-fold CV acc: {cv.mean():.4f} ± {cv.std():.4f}")
    print(f"  Test acc:      {model.score(X_te, y_te):.4f}")
    print(f"  Class priors:  {model.class_prior_.round(4)}")


# ── 3. Multinomial NB for text ───────────────────────────────────────────────
def multinomial_nb_text():
    print("\n=== Multinomial NB for Text Classification ===")
    corpus = [
        # spam
        "free money click here now",
        "you won a prize claim reward",
        "buy cheap pills online discount",
        "limited offer free gift click",
        "earn cash at home fast free",
        # ham
        "meeting tomorrow at 3pm",
        "please review the attached report",
        "lunch with the team on Friday",
        "project deadline is next Monday",
        "can we reschedule the call today",
    ]
    labels = [1,1,1,1,1, 0,0,0,0,0]   # 1=spam, 0=ham

    vec   = CountVectorizer()
    X     = vec.fit_transform(corpus)
    model = MultinomialNB(alpha=1.0)   # Laplace smoothing
    model.fit(X, labels)

    tests = [
        "free cash reward now",
        "project review meeting",
    ]
    X_test = vec.transform(tests)
    preds  = model.predict(X_test)
    probs  = model.predict_proba(X_test)
    for txt, pred, prob in zip(tests, preds, probs):
        print(f"  '{txt}'")
        print(f"    → {'spam' if pred==1 else 'ham'}  P(ham)={prob[0]:.4f}  P(spam)={prob[1]:.4f}")

    print(f"\n  Vocabulary size: {len(vec.vocabulary_)}")
    print(f"  Log-likelihood (spam) top terms:")
    spam_log = model.feature_log_prob_[1]
    top_idx  = spam_log.argsort()[-5:][::-1]
    vocab_inv = {v:k for k,v in vec.vocabulary_.items()}
    for i in top_idx:
        print(f"    '{vocab_inv[i]}': log P = {spam_log[i]:.4f}")


# ── 4. Laplace smoothing ─────────────────────────────────────────────────────
def laplace_smoothing():
    print("\n=== Laplace Smoothing ===")
    print("  Without smoothing: unseen word → P=0 → P(y|x)=0")
    print("  With α smoothing: P(x_j|y) = (count(x_j,y)+α) / (count(y)+α·|V|)")
    print()
    counts = {"word_a": 10, "word_b": 5, "word_c": 0}   # word_c unseen in class
    N, V = sum(counts.values()), len(counts)
    print(f"  Counts in class: {counts}   N={N}  |V|={V}")
    for alpha in [0, 0.1, 0.5, 1.0]:
        probs = {w: (c+alpha)/(N+alpha*V) for w,c in counts.items()}
        print(f"  α={alpha}: {{'a':{probs['word_a']:.4f}, 'b':{probs['word_b']:.4f}, 'c':{probs['word_c']:.4f}}}")


# ── 5. Bernoulli NB (binary features) ────────────────────────────────────────
def bernoulli_nb_demo():
    print("\n=== Bernoulli NB (binary features) ===")
    rng = np.random.default_rng(2)
    # Binary feature matrix
    n_pos, n_neg = 200, 200
    X_pos = rng.binomial(1, [0.8,0.7,0.2,0.1], (n_pos,4))
    X_neg = rng.binomial(1, [0.2,0.3,0.8,0.9], (n_neg,4))
    X     = np.vstack([X_pos, X_neg])
    y     = np.array([1]*n_pos + [0]*n_neg)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)
    model = BernoulliNB(alpha=1.0)
    model.fit(X_tr, y_tr)
    acc = model.score(X_te, y_te)
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Feature probs per class (log scale):")
    for c, name in enumerate(["Negative", "Positive"]):
        probs = np.exp(model.feature_log_prob_[c])
        print(f"    {name}: {probs.round(4)}")


# ── 6. Naive Bayes assumptions analysis ──────────────────────────────────────
def assumption_analysis():
    print("\n=== When Naive Bayes Fails (Correlated Features) ===")
    rng = np.random.default_rng(3)
    n   = 300

    # Case 1: independent features (NB works)
    X_indep = rng.standard_normal((n, 4))
    y_ind   = (X_indep[:,0] + X_indep[:,1] > 0).astype(int)

    # Case 2: highly correlated features (NB struggles)
    X_base  = rng.standard_normal((n, 2))
    X_corr  = np.column_stack([X_base, X_base + 0.1*rng.standard_normal((n,2))])
    y_corr  = (X_base[:,0] + X_base[:,1] > 0).astype(int)

    from sklearn.linear_model import LogisticRegression
    print(f"  {'Scenario':<30} {'GNB acc':<12} {'LR acc'}")
    for name, X, y in [("Independent features", X_indep, y_ind),
                        ("Correlated features",  X_corr,  y_corr)]:
        for clf_name, clf in [("GNB", GaussianNB()), ("LR", LogisticRegression())]:
            cv = cross_val_score(clf, X, y, cv=5).mean()
            if clf_name == "GNB":
                gnb_cv = cv
            else:
                print(f"  {name:<30} {gnb_cv:<12.4f} {cv:.4f}")


if __name__ == "__main__":
    naive_bayes_from_scratch()
    gaussian_nb_demo()
    multinomial_nb_text()
    laplace_smoothing()
    bernoulli_nb_demo()
    assumption_analysis()
