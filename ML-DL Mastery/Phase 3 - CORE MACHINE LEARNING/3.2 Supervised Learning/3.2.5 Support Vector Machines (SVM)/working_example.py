"""
Working Example: Support Vector Machines (SVM)
Covers maximal margin, dual problem, kernel trick, soft margin (C),
multi-class SVM, SVR, and hyperparameter tuning.
"""
import numpy as np
from sklearn.svm import SVC, SVR, LinearSVC
from sklearn.datasets import make_classification, make_moons, make_blobs
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_svm")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── 1. Hard-margin SVM intuition ─────────────────────────────────────────────
def svm_intuition():
    print("=== SVM: Maximum Margin Classifier ===")
    print("  Primal problem:")
    print("    min ||w||²/2   s.t.  y_i(wᵀx_i + b) ≥ 1  ∀i")
    print()
    print("  Dual problem:")
    print("    max Σα_i - ½ Σ_ij α_i α_j y_i y_j (xᵢᵀxⱼ)")
    print("    s.t. Σ α_i y_i = 0,  α_i ≥ 0")
    print()
    print("  Support vectors: training points with α_i > 0  (on the margin)")
    print("  Margin width = 2 / ||w||")
    print()
    print("  Kernel trick: replace xᵢᵀxⱼ → K(xᵢ,xⱼ)")
    print("  Common kernels: Linear, RBF (Gaussian), Polynomial, Sigmoid")


# ── 2. Linear SVM on linearly separable data ─────────────────────────────────
def linear_svm():
    print("\n=== Linear SVM ===")
    rng  = np.random.default_rng(0)
    X, y = make_classification(n_samples=200, n_features=2, n_redundant=0,
                                n_informative=2, random_state=0, class_sep=2.0)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)
    scaler = StandardScaler().fit(X_tr)
    X_tr_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_te)

    for C in [0.01, 0.1, 1.0, 10.0, 100.0]:
        model = SVC(kernel="linear", C=C)
        model.fit(X_tr_s, y_tr)
        acc = model.score(X_te_s, y_te)
        n_sv = model.n_support_.sum()
        margin = 2 / np.linalg.norm(model.coef_)
        print(f"  C={C:<8} acc={acc:.4f}  support_vectors={n_sv}  margin={margin:.4f}")


# ── 3. Kernel SVM ────────────────────────────────────────────────────────────
def kernel_svm():
    print("\n=== Kernel SVM on Non-Linearly Separable Data ===")
    X, y = make_moons(n_samples=400, noise=0.2, random_state=1)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)
    scaler = StandardScaler().fit(X_tr)
    X_tr_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_te)

    kernels = [
        ("linear",  SVC(kernel="linear",  C=1)),
        ("poly d=3",SVC(kernel="poly",    C=1, degree=3, coef0=1)),
        ("rbf",     SVC(kernel="rbf",     C=1, gamma="scale")),
        ("sigmoid", SVC(kernel="sigmoid", C=1, coef0=1)),
    ]
    print(f"  {'Kernel':<12} {'Test acc':<12} {'n_support_vectors'}")
    for name, model in kernels:
        model.fit(X_tr_s, y_tr)
        acc  = model.score(X_te_s, y_te)
        n_sv = model.n_support_.sum()
        print(f"  {name:<12} {acc:<12.4f} {n_sv}")


# ── 4. Soft-margin (C parameter) ─────────────────────────────────────────────
def soft_margin_C():
    print("\n=== Soft-Margin SVM: C Hyperparameter ===")
    print("  min  ½||w||² + C·Σξ_i   s.t.  y_i(wᵀx_i+b) ≥ 1-ξ_i,  ξ_i ≥ 0")
    print("  C → ∞: hard margin (low bias, high variance)")
    print("  C → 0: wide margin, more violations (high bias, low variance)")
    X, y = make_moons(n_samples=400, noise=0.3, random_state=2)
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    X_tr, X_te, y_tr, y_te = train_test_split(X_s, y, test_size=0.3, random_state=0)

    print(f"\n  {'C':<10} {'Train acc':<12} {'Test acc'}")
    for C in [0.001, 0.01, 0.1, 1, 10, 100]:
        model = SVC(kernel="rbf", C=C, gamma="scale")
        model.fit(X_tr, y_tr)
        tr_acc = model.score(X_tr, y_tr)
        te_acc = model.score(X_te, y_te)
        print(f"  {C:<10} {tr_acc:<12.4f} {te_acc:.4f}")


# ── 5. RBF kernel and gamma ───────────────────────────────────────────────────
def rbf_gamma_effect():
    print("\n=== RBF Kernel: γ Effect ===")
    print("  K(x,x') = exp(-γ||x-x'||²)")
    print("  γ large → narrow Gaussian → more complex boundary (overfit)")
    print("  γ small → wide Gaussian → smoother boundary (underfit)")

    X, y = make_moons(n_samples=400, noise=0.2, random_state=3)
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    X_tr, X_te, y_tr, y_te = train_test_split(X_s, y, test_size=0.3, random_state=0)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, gamma in zip(axes, [0.01, 0.1, 1.0, 10.0]):
        model = SVC(kernel="rbf", C=1.0, gamma=gamma).fit(X_tr, y_tr)
        acc   = model.score(X_te, y_te)
        xx, yy = np.meshgrid(np.linspace(-2.5, 3, 150), np.linspace(-1.5, 2, 150))
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        ax.contourf(xx, yy, Z, alpha=0.3, cmap="RdBu")
        ax.scatter(X_s[:,0], X_s[:,1], c=y, cmap="RdBu", edgecolors='k', s=15, lw=0.4)
        ax.set(title=f"γ={gamma}  acc={acc:.3f}")
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "rbf_gamma.png")
    plt.savefig(path, dpi=90); plt.close()
    print(f"  RBF gamma plot saved: {path}")


# ── 6. GridSearch for C and gamma ────────────────────────────────────────────
def grid_search_svm():
    print("\n=== Grid Search for SVM Hyperparameters ===")
    X, y = make_moons(n_samples=400, noise=0.2, random_state=4)
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    param_grid = {"C": [0.1, 1, 10], "gamma": [0.1, 1, "scale"]}
    gs = GridSearchCV(SVC(kernel="rbf"), param_grid, cv=5, n_jobs=-1)
    gs.fit(X_s, y)
    print(f"  Best params: {gs.best_params_}")
    print(f"  Best CV acc: {gs.best_score_:.4f}")


# ── 7. Support Vector Regression ────────────────────────────────────────────
def svr_demo():
    print("\n=== Support Vector Regression (SVR) ===")
    print("  ε-insensitive loss: loss=0 if |y-ŷ|<ε, else |y-ŷ|-ε")
    rng = np.random.default_rng(5)
    x   = rng.uniform(-3, 3, 100)
    y   = np.sin(x) + rng.normal(0, 0.3, 100)
    X   = x.reshape(-1,1)
    x_t = np.linspace(-3, 3, 200).reshape(-1,1)
    y_t = np.sin(x_t.ravel())

    scaler = StandardScaler().fit(X)
    X_s, X_t_s = scaler.transform(X), scaler.transform(x_t)

    print(f"  {'ε':<8} {'C':<8} {'RMSE'}")
    for epsilon in [0.01, 0.1, 0.5]:
        for C in [0.1, 1, 10]:
            model = SVR(kernel="rbf", epsilon=epsilon, C=C, gamma="scale")
            model.fit(X_s, y)
            rmse = np.sqrt(np.mean((model.predict(X_t_s) - y_t)**2))
            print(f"  {epsilon:<8} {C:<8} {rmse:.4f}")


# ── 8. Multi-class SVM ───────────────────────────────────────────────────────
def multiclass_svm():
    print("\n=== Multi-class SVM ===")
    X, y = make_blobs(n_samples=300, centers=4, random_state=6, cluster_std=1.5)
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    X_tr, X_te, y_tr, y_te = train_test_split(X_s, y, test_size=0.3, random_state=0)

    for strategy, model in [
        ("OvO (default SVC)", SVC(kernel="rbf", C=10, gamma="scale")),
        ("OvR (LinearSVC)",   LinearSVC(max_iter=2000, C=10)),
    ]:
        model.fit(X_tr, y_tr)
        acc = model.score(X_te, y_te)
        print(f"  {strategy:<25}: acc={acc:.4f}")


if __name__ == "__main__":
    svm_intuition()
    linear_svm()
    kernel_svm()
    soft_margin_C()
    rbf_gamma_effect()
    grid_search_svm()
    svr_demo()
    multiclass_svm()
