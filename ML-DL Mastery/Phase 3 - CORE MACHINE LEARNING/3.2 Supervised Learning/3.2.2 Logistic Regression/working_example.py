"""
Working Example: Logistic Regression
Covers sigmoid, binary cross-entropy, gradient descent, multi-class (softmax/OvR),
decision boundaries, regularisation, calibration, and ROC/AUC.
"""
import numpy as np
from scipy.special import expit   # sigmoid
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification, load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
from sklearn.preprocessing import StandardScaler
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_logreg")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -- 1. Sigmoid and odds -------------------------------------------------------
def sigmoid_demo():
    print("=== Sigmoid Function and Log-Odds ===")
    z = np.array([-5, -2, -1, 0, 1, 2, 5], dtype=float)
    p = expit(z)
    odds   = p / (1 - p)
    logodds = np.log(odds)
    print(f"  {'z':>6}  {'sigma(z)':>8}  {'odds':>10}  {'log-odds':>10}")
    for zi, pi, oi, li in zip(z, p, odds, logodds):
        print(f"  {zi:>6.1f}  {pi:>8.4f}  {oi:>10.4f}  {li:>10.4f}")
    print()
    print("  Interpretation: log(p/(1-p)) = wᵀx   (linear in log-odds space)")


# -- 2. Binary cross-entropy and MLE ------------------------------------------
def binary_cross_entropy():
    print("\n=== Binary Cross-Entropy (log-loss) ===")
    print("  L(w) = -Sigma [y_i log(p_i) + (1-y_i) log(1-p_i)]")
    print("  dL/dw = Sigma (p_i - y_i) x_i   (gradient)")
    print()

    # Example
    y_true = np.array([1, 1, 0, 0, 1])
    y_prob = np.array([0.9, 0.7, 0.3, 0.2, 0.6])
    bce = -np.mean(y_true*np.log(y_prob + 1e-15) + (1-y_true)*np.log(1-y_prob+1e-15))
    print(f"  y_true={y_true}  y_prob={y_prob}")
    print(f"  BCE = {bce:.4f}")

    # Gradient descent from scratch
    rng = np.random.default_rng(0)
    n   = 200
    X   = rng.standard_normal((n, 2))
    y   = (X[:,0] + 0.5*X[:,1] > 0).astype(float)
    X   = np.column_stack([np.ones(n), X])

    w    = np.zeros(3)
    lr   = 0.1
    for _ in range(500):
        p    = expit(X @ w)
        grad = X.T @ (p - y) / n
        w   -= lr * grad

    print(f"\n  GD weights (500 steps): {w.round(4)}")
    acc = ((expit(X@w) > 0.5).astype(float) == y).mean()
    print(f"  Accuracy: {acc:.4f}")


# -- 3. Logistic regression with sklearn --------------------------------------
def sklearn_logistic():
    print("\n=== sklearn Logistic Regression ===")
    rng  = np.random.default_rng(1)
    X, y = make_classification(n_samples=500, n_features=5, n_informative=3,
                                random_state=1)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)

    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr)
    X_te_s = scaler.transform(X_te)

    for penalty, C in [("l2", 1), ("l2", 0.01), ("l1", 1)]:
        solver = "liblinear" if penalty=="l1" else "lbfgs"
        model  = LogisticRegression(penalty=penalty, C=C, solver=solver, max_iter=500)
        model.fit(X_tr_s, y_tr)
        proba = model.predict_proba(X_te_s)[:,1]
        auc   = roc_auc_score(y_te, proba)
        acc   = model.score(X_te_s, y_te)
        print(f"  {penalty.upper()}, C={C:<6}: acc={acc:.4f}  AUC={auc:.4f}  "
              f"||w||2={np.linalg.norm(model.coef_):.4f}")


# -- 4. Multi-class: OvR and Softmax -----------------------------------------
def multiclass_logistic():
    print("\n=== Multi-class Logistic Regression ===")
    iris = load_iris()
    X, y = iris.data, iris.target
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)
    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr)
    X_te_s = scaler.transform(X_te)

    for mc_str, solver, multi in [
        ("OvR (one-vs-rest)",  "lbfgs",     "ovr"),
        ("Softmax (multinomial)", "lbfgs",   "multinomial"),
    ]:
        model = LogisticRegression(multi_class=multi, solver=solver, max_iter=500)
        model.fit(X_tr_s, y_tr)
        acc   = model.score(X_te_s, y_te)
        print(f"  {mc_str:<30}: acc={acc:.4f}")

    # Softmax function
    print("\n  Softmax: P(y=k|x) = exp(wₖᵀx) / Sigma_j exp(wⱼᵀx)")
    z   = np.array([2.0, 1.0, 0.5])
    sm  = np.exp(z - z.max()) / np.exp(z - z.max()).sum()  # numerically stable
    print(f"  z={z} -> softmax={sm.round(4)}")


# -- 5. Decision boundary visualisation ---------------------------------------
def decision_boundary():
    rng   = np.random.default_rng(2)
    X, y  = make_classification(n_samples=300, n_features=2, n_redundant=0,
                                 n_informative=2, random_state=2)
    model = LogisticRegression().fit(X, y)

    xx, yy = np.meshgrid(np.linspace(X[:,0].min()-1, X[:,0].max()+1, 200),
                         np.linspace(X[:,1].min()-1, X[:,1].max()+1, 200))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(7,5))
    ax.contourf(xx, yy, Z, alpha=0.3, cmap="RdBu")
    ax.scatter(X[:,0], X[:,1], c=y, cmap="RdBu", edgecolors='k', s=20, lw=0.4)
    ax.set(xlabel="x1", ylabel="x2", title="Logistic Regression Decision Boundary")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "decision_boundary.png")
    plt.savefig(path, dpi=90); plt.close()
    print(f"\n  Decision boundary saved: {path}")


# -- 6. ROC curve and AUC ----------------------------------------------------
def roc_auc_demo():
    print("\n=== ROC Curve and AUC ===")
    rng  = np.random.default_rng(3)
    X, y = make_classification(n_samples=500, n_features=10, random_state=3)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)

    model = LogisticRegression(max_iter=500).fit(X_tr, y_tr)
    proba = model.predict_proba(X_te)[:,1]
    auc   = roc_auc_score(y_te, proba)
    fpr, tpr, thresholds = roc_curve(y_te, proba)

    print(f"  AUC = {auc:.4f}")
    print(f"  Optimal threshold (Youden's J): "
          f"{thresholds[(tpr - fpr).argmax()]:.4f}")
    print(f"\n  Classification report (threshold=0.5):")
    print(classification_report(y_te, model.predict(X_te), indent=4))

    # Plot ROC
    fig, ax = plt.subplots(figsize=(6,5))
    ax.plot(fpr, tpr, lw=2, label=f"Logistic (AUC={auc:.3f})")
    ax.plot([0,1],[0,1], 'k--', lw=1, label="Random")
    ax.set(xlabel="FPR", ylabel="TPR", title="ROC Curve")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "roc_curve.png")
    plt.savefig(path, dpi=90); plt.close()
    print(f"  ROC curve saved: {path}")


if __name__ == "__main__":
    sigmoid_demo()
    binary_cross_entropy()
    sklearn_logistic()
    multiclass_logistic()
    decision_boundary()
    roc_auc_demo()
