"""
Working Example: Perceptron and Linear Models
Covers the perceptron algorithm, convergence theorem, delta rule (Widrow-Hoff),
linear discriminant analysis, and comparison with logistic regression.
"""
import numpy as np
from sklearn.linear_model import Perceptron, SGDClassifier, LinearDiscriminantAnalysis
from sklearn.datasets import make_classification, load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_perceptron")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── 1. Perceptron from scratch ────────────────────────────────────────────────
def perceptron_scratch():
    print("=== Perceptron Algorithm (from scratch) ===")
    print("  Update: if y_i(wᵀx_i) ≤ 0: w ← w + η·y_i·x_i")
    print()

    class Perceptron_scratch:
        def __init__(self, lr=1.0, max_iter=1000):
            self.lr, self.max_iter = lr, max_iter

        def fit(self, X, y):
            n, p = X.shape
            # Add bias column
            X_b = np.column_stack([np.ones(n), X])
            self.w = np.zeros(p + 1)
            self.errors_per_epoch = []
            for epoch in range(self.max_iter):
                errors = 0
                for xi, yi in zip(X_b, y):
                    if yi * (self.w @ xi) <= 0:
                        self.w += self.lr * yi * xi
                        errors += 1
                self.errors_per_epoch.append(errors)
                if errors == 0:
                    print(f"  Converged at epoch {epoch+1}")
                    break
            return self

        def predict(self, X):
            X_b = np.column_stack([np.ones(len(X)), X])
            return np.sign(X_b @ self.w)

    rng = np.random.default_rng(0)
    # Linearly separable data
    X_pos = rng.multivariate_normal([2, 2], np.eye(2), 50)
    X_neg = rng.multivariate_normal([-2,-2], np.eye(2), 50)
    X = np.vstack([X_pos, X_neg])
    y = np.array([1]*50 + [-1]*50)

    p = Perceptron_scratch(lr=0.1).fit(X, y)
    acc = (p.predict(X) == y).mean()
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Weights: {p.w.round(4)}")

    # Non-separable data (will not converge cleanly)
    X2, y2 = make_classification(n_samples=200, n_features=2, n_redundant=0,
                                  n_informative=2, class_sep=0.5, random_state=1)
    y2 = 2*y2 - 1  # {0,1} → {-1,+1}
    p2 = Perceptron_scratch(lr=0.01, max_iter=200).fit(X2, y2)
    acc2 = (p2.predict(X2) == y2).mean()
    print(f"\n  Non-separable: accuracy={acc2:.4f}  (no convergence guarantee)")


# ── 2. Perceptron convergence theorem ────────────────────────────────────────
def convergence_theorem():
    print("\n=== Perceptron Convergence Theorem ===")
    print("  IF data is linearly separable with margin γ, and ||x_i|| ≤ R:")
    print("  Mistakes ≤ (R/γ)²")
    print()
    print("  Proof sketch:")
    print("  1. Each mistake increases w·w* by at least γ")
    print("  2. Each mistake increases ||w||² by at most R²")
    print("  3. Combining → bounded mistakes")


# ── 3. Delta rule (Widrow-Hoff / LMS) ────────────────────────────────────────
def delta_rule():
    print("\n=== Delta Rule (Widrow-Hoff / LMS) ===")
    print("  Minimise MSE: L(w) = Σ(y_i - wᵀx_i)²")
    print("  Update: w ← w + η·(y_i - ŷ_i)·x_i   [gradient descent on MSE]")
    print()

    rng = np.random.default_rng(2)
    n   = 100
    x   = rng.uniform(0, 5, n)
    y   = 2*x + 1 + rng.normal(0, 0.5, n)
    X   = np.column_stack([np.ones(n), x])

    w    = np.zeros(2)
    lr   = 0.005
    losses = []
    for epoch in range(500):
        for xi, yi in zip(X, y):
            error = yi - w @ xi
            w    += lr * error * xi
        losses.append(np.mean((X@w - y)**2))

    w_ols = np.linalg.solve(X.T @ X, X.T @ y)
    print(f"  Delta rule:  w={w.round(4)}")
    print(f"  OLS:         w={w_ols.round(4)}")
    print(f"  Final MSE:   {losses[-1]:.4f}")


# ── 4. sklearn Perceptron ─────────────────────────────────────────────────────
def sklearn_perceptron():
    print("\n=== sklearn Perceptron ===")
    X, y = make_classification(n_samples=500, n_features=10, n_informative=5, random_state=3)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)
    scaler = StandardScaler().fit(X_tr)
    X_tr_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_te)

    p = Perceptron(max_iter=1000, tol=1e-3, random_state=0)
    p.fit(X_tr_s, y_tr)
    print(f"  Perceptron: test acc={p.score(X_te_s, y_te):.4f}  "
          f"n_iter={p.n_iter_}")

    # SGDClassifier with hinge loss ≈ linear SVM
    sgd = SGDClassifier(loss="hinge", max_iter=1000, tol=1e-3, random_state=0)
    sgd.fit(X_tr_s, y_tr)
    print(f"  SGD(hinge): test acc={sgd.score(X_te_s, y_te):.4f}  "
          f"(linear SVM approximation)")

    # SGDClassifier with log_loss ≈ logistic regression
    sgd_log = SGDClassifier(loss="log_loss", max_iter=1000, tol=1e-3, random_state=0)
    sgd_log.fit(X_tr_s, y_tr)
    print(f"  SGD(log):   test acc={sgd_log.score(X_te_s, y_te):.4f}  "
          f"(logistic regression approximation)")


# ── 5. Linear Discriminant Analysis ─────────────────────────────────────────
def lda_demo():
    print("\n=== Linear Discriminant Analysis (LDA) ===")
    print("  Model: P(x|y=k) = N(μ_k, Σ)   (shared covariance)")
    print("  Decision rule: argmax_k log P(y=k) + log P(x|y=k)")
    print("  Also used for dimensionality reduction: projects to n_classes-1 dims")

    iris = load_iris()
    X, y = iris.data, iris.target
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)

    lda = LinearDiscriminantAnalysis()
    lda.fit(X_tr, y_tr)
    acc = lda.score(X_te, y_te)
    print(f"\n  LDA classifier accuracy: {acc:.4f}")

    # Dimensionality reduction
    X_lda = lda.transform(X)
    print(f"  Original shape: {X.shape}  → LDA shape: {X_lda.shape}")

    # Plot LDA projection
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["blue", "red", "green"]
    for c, name, color in zip([0,1,2], iris.target_names, colors):
        ax.scatter(X_lda[y==c, 0], X_lda[y==c, 1], label=name,
                   color=color, alpha=0.6, s=30, edgecolors='k', lw=0.4)
    ax.set(xlabel="LD1", ylabel="LD2", title="LDA Projection (Iris)")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "lda_projection.png")
    plt.savefig(path, dpi=90); plt.close()
    print(f"  LDA projection saved: {path}")


# ── 6. Comparison of linear classifiers ──────────────────────────────────────
def linear_classifier_comparison():
    print("\n=== Linear Classifier Comparison ===")
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC

    X, y = make_classification(n_samples=500, n_features=10, n_informative=5, random_state=4)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)
    scaler = StandardScaler().fit(X_tr)
    X_tr_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_te)

    classifiers = [
        ("Perceptron",       Perceptron(max_iter=1000, random_state=0)),
        ("SGD(hinge/SVM)",   SGDClassifier(loss="hinge",    random_state=0, max_iter=1000)),
        ("SGD(log/LR)",      SGDClassifier(loss="log_loss", random_state=0, max_iter=1000)),
        ("LogisticRegr",     LogisticRegression(max_iter=500)),
        ("LinearSVC",        LinearSVC(max_iter=2000)),
        ("LDA",              LinearDiscriminantAnalysis()),
    ]

    print(f"  {'Classifier':<22} {'Test acc'}")
    for name, clf in classifiers:
        clf.fit(X_tr_s, y_tr)
        acc = clf.score(X_te_s, y_te)
        print(f"  {name:<22} {acc:.4f}")

    print()
    print("  Key differences:")
    print("    Perceptron: no probabilistic output, only converges if separable")
    print("    LR/SVM:     convex objectives, always converge")
    print("    LDA:        generative (models P(x|y)), optimal when Gaussian assumptions hold")


if __name__ == "__main__":
    perceptron_scratch()
    convergence_theorem()
    delta_rule()
    sklearn_perceptron()
    lda_demo()
    linear_classifier_comparison()
