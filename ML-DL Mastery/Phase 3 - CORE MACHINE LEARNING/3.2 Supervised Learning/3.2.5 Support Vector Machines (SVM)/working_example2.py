"""
Working Example 2: Support Vector Machines — SVC, SVR, Kernels, Cal Housing
=============================================================================
Linear SVC, RBF kernel, C vs margin tradeoff, SVR regression.

Run:  python working_example2.py
"""
from pathlib import Path
try:
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC, SVR, LinearSVC
    from sklearn.metrics import mean_squared_error, roc_auc_score, classification_report
    from sklearn.pipeline import make_pipeline
except ImportError:
    raise SystemExit("pip install numpy matplotlib scikit-learn")

OUTPUT = Path(__file__).parent / "output"
OUTPUT.mkdir(exist_ok=True)

def load_binary(n=5000):
    h = fetch_california_housing()
    X, y = h.data[:n], (h.target[:n] > np.median(h.target[:n])).astype(int)
    return train_test_split(X, y, test_size=0.2, random_state=42)

def demo_linear_svc(X_train, X_test, y_train, y_test):
    print("=== Linear SVC (Cal Housing binary) ===")
    for C in [0.01, 0.1, 1.0, 10.0]:
        pipe = make_pipeline(StandardScaler(), LinearSVC(C=C, max_iter=5000))
        pipe.fit(X_train, y_train)
        acc = pipe.score(X_test, y_test)
        print(f"  C={C:>5}: accuracy={acc:.4f}")

def demo_rbf_kernel(X_train, X_test, y_train, y_test):
    print("\n=== RBF Kernel SVC ===")
    # Use subset for speed
    pipe = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, gamma="scale", probability=True))
    pipe.fit(X_train, y_train)
    y_prob = pipe.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    acc = pipe.score(X_test, y_test)
    print(f"  RBF SVC: accuracy={acc:.4f}  AUC={auc:.4f}")

def demo_svr():
    print("\n=== SVR (Support Vector Regression) ===")
    h = fetch_california_housing()
    X, y = h.data[:3000], h.target[:3000]   # smaller for SVR speed
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    for kernel in ["linear", "rbf"]:
        pipe = make_pipeline(StandardScaler(), SVR(kernel=kernel, C=1.0, epsilon=0.1))
        pipe.fit(X_train, y_train)
        rmse = mean_squared_error(y_test, pipe.predict(X_test))**0.5
        print(f"  SVR({kernel}): RMSE={rmse:.4f}")

def demo_kernel_comparison(X_train, X_test, y_train, y_test):
    """Compare multiple kernels side-by-side."""
    print("\n=== Kernel Comparison ===")
    results = {}
    for kernel in ["linear", "poly", "rbf", "sigmoid"]:
        pipe = make_pipeline(
            StandardScaler(),
            SVC(kernel=kernel, C=1.0, gamma="scale", degree=3, probability=True),
        )
        pipe.fit(X_train, y_train)
        auc = roc_auc_score(y_test, pipe.predict_proba(X_test)[:, 1])
        acc = pipe.score(X_test, y_test)
        results[kernel] = (acc, auc)
        print(f"  {kernel:8s}: accuracy={acc:.4f}  AUC={auc:.4f}")

    kernels = list(results.keys())
    accs = [results[k][0] for k in kernels]
    aucs = [results[k][1] for k in kernels]
    x = np.arange(len(kernels))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - 0.2, accs, 0.4, label="Accuracy")
    ax.bar(x + 0.2, aucs, 0.4, label="AUC")
    ax.set_xticks(x); ax.set_xticklabels(kernels)
    ax.set_ylim(0.7, 1.0); ax.legend(); ax.set_title("SVM Kernel Comparison")
    fig.savefig(OUTPUT / "svm_kernels.png", dpi=120, bbox_inches="tight")
    plt.close(fig); print("  Saved: svm_kernels.png")


def demo_c_gamma_grid():
    """Heatmap of accuracy over C x gamma grid for RBF SVM."""
    print("\n=== C vs Gamma Grid (RBF SVM) ===")
    h = fetch_california_housing()
    X, y = h.data[:2000], (h.target[:2000] > np.median(h.target[:2000])).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)
    Cs     = [0.1, 1.0, 10.0, 100.0]
    gammas = [0.001, 0.01, 0.1, 1.0]
    grid = np.zeros((len(Cs), len(gammas)))
    for i, C in enumerate(Cs):
        for j, g in enumerate(gammas):
            svc = SVC(C=C, gamma=g, kernel="rbf")
            svc.fit(X_tr, y_train)
            grid[i, j] = svc.score(X_te, y_test)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(grid, vmin=grid.min(), vmax=grid.max(), aspect="auto")
    ax.set_xticks(range(len(gammas))); ax.set_xticklabels(gammas)
    ax.set_yticks(range(len(Cs))); ax.set_yticklabels(Cs)
    ax.set_xlabel("gamma"); ax.set_ylabel("C")
    ax.set_title("RBF SVM Accuracy Heatmap")
    plt.colorbar(im, ax=ax)
    for i in range(len(Cs)):
        for j in range(len(gammas)):
            ax.text(j, i, f"{grid[i,j]:.2f}", ha="center", va="center", fontsize=8, color="white")
    fig.savefig(OUTPUT / "svm_grid.png", dpi=120, bbox_inches="tight")
    plt.close(fig); print("  Saved: svm_grid.png")
    best_i, best_j = np.unravel_index(grid.argmax(), grid.shape)
    print(f"  Best: C={Cs[best_i]}  gamma={gammas[best_j]}  accuracy={grid[best_i,best_j]:.4f}")


def demo_support_vectors():
    """Demonstrate margin width and support vector count vs C on 2-D data."""
    print("\n=== Margin Width vs C (2D linearly separable) ===")
    from sklearn.datasets import make_classification
    np.random.seed(0)
    X2, y2 = make_classification(n_samples=200, n_features=2, n_redundant=0,
                                   n_informative=2, n_clusters_per_class=1, random_state=7)
    X_tr, X_te, y_tr, y_te = train_test_split(X2, y2, test_size=0.25, random_state=42)
    scaler = StandardScaler(); X_tr = scaler.fit_transform(X_tr); X_te = scaler.transform(X_te)
    for C in [0.01, 0.1, 1.0, 10.0, 100.0]:
        svc = SVC(kernel="linear", C=C)
        svc.fit(X_tr, y_tr)
        n_sv = svc.n_support_.sum()
        acc  = svc.score(X_te, y_te)
        # margin width = 2 / ||w||
        w_norm = np.linalg.norm(svc.coef_)
        margin = 2.0 / w_norm if w_norm > 0 else float("inf")
        print(f"  C={C:>6}: support_vectors={n_sv:>3}  margin={margin:.4f}  accuracy={acc:.4f}")


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_binary()
    demo_linear_svc(X_train, X_test, y_train, y_test)
    demo_rbf_kernel(X_train, X_test, y_train, y_test)
    demo_svr()
    demo_kernel_comparison(X_train, X_test, y_train, y_test)
    demo_c_gamma_grid()
    demo_support_vectors()
