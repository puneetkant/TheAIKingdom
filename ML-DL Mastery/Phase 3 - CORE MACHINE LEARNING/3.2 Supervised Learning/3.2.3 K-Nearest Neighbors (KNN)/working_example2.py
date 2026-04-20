"""
Working Example 2: K-Nearest Neighbors — Regression & Classification, k sweep
==============================================================================
KNN regression and binary classification on Cal Housing.
Effect of k, distance weighting, curse of dimensionality.

Run:  python working_example2.py
"""
from pathlib import Path
try:
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
    from sklearn.metrics import mean_squared_error, roc_auc_score
    from sklearn.pipeline import make_pipeline
except ImportError:
    raise SystemExit("pip install numpy matplotlib scikit-learn")

OUTPUT = Path(__file__).parent / "output"
OUTPUT.mkdir(exist_ok=True)

def demo_knn_regression():
    print("=== KNN Regression (Cal Housing) ===")
    h = fetch_california_housing()
    X, y = h.data, h.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    results = []
    for k in [1, 3, 5, 10, 20, 50]:
        pipe = make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=k, weights="distance"))
        pipe.fit(X_train, y_train)
        rmse = mean_squared_error(y_test, pipe.predict(X_test))**0.5
        results.append((k, rmse))
        print(f"  k={k:>3}: RMSE={rmse:.4f}")

    ks, rmses = zip(*results)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ks, rmses, "o-"); ax.set_xlabel("k"); ax.set_ylabel("RMSE")
    ax.set_title("KNN Regression: RMSE vs k")
    fig.savefig(OUTPUT / "knn_k_sweep.png", dpi=120, bbox_inches="tight")
    plt.close(fig); print("  Saved: knn_k_sweep.png")

def demo_knn_classification():
    print("\n=== KNN Classification (binary) ===")
    h = fetch_california_housing()
    X, y = h.data, (h.target > np.median(h.target)).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    for k in [3, 5, 10, 20]:
        pipe = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k))
        pipe.fit(X_train, y_train)
        acc = pipe.score(X_test, y_test)
        auc = roc_auc_score(y_test, pipe.predict_proba(X_test)[:, 1])
        print(f"  k={k:>3}: accuracy={acc:.4f}  AUC={auc:.4f}")

def demo_curse_dimensionality():
    print("\n=== Curse of Dimensionality ===")
    np.random.seed(42)
    # Fraction of data in hypercube side s in d dimensions
    # To capture r fraction: s = r^(1/d)
    r = 0.01   # want 1% of data
    for d in [1, 2, 5, 10, 20, 100]:
        s = r**(1/d)
        print(f"  d={d:>3}: need side fraction {s:.4f} to capture {100*r}% of data")

def demo_distance_metrics():
    """Compare Euclidean, Manhattan and Chebyshev distance metrics for KNN."""
    print("\n=== Distance Metric Comparison ===")
    h = fetch_california_housing()
    X, y = h.data, (h.target > np.median(h.target)).astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    # p=2 -> Euclidean, p=1 -> Manhattan
    for metric, p in [("euclidean", 2), ("manhattan", 1), ("chebyshev", None)]:
        if metric == "chebyshev":
            knn = KNeighborsClassifier(n_neighbors=10, metric="chebyshev")
        else:
            knn = KNeighborsClassifier(n_neighbors=10, p=p)
        pipe = make_pipeline(StandardScaler(), knn)
        pipe.fit(X_tr, y_tr)
        acc = pipe.score(X_te, y_te)
        auc = roc_auc_score(y_te, pipe.predict_proba(X_te)[:, 1])
        print(f"  {metric:12s}: accuracy={acc:.4f}  AUC={auc:.4f}")


def demo_radius_neighbors():
    """RadiusNeighborsClassifier — classify only within a fixed radius."""
    print("\n=== Radius Neighbors Classifier ===")
    from sklearn.neighbors import RadiusNeighborsClassifier
    h = fetch_california_housing()
    X, y = h.data[:3000], (h.target[:3000] > np.median(h.target[:3000])).astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s  = scaler.transform(X_te)
    for r in [1.5, 2.0, 3.0]:
        rn = RadiusNeighborsClassifier(radius=r, outlier_label=0)
        rn.fit(X_tr_s, y_tr)
        acc = rn.score(X_te_s, y_te)
        print(f"  radius={r}: accuracy={acc:.4f}")


def demo_knn_cv_selection():
    """Cross-validation to pick the best k."""
    print("\n=== Cross-Validation K Selection ===")
    h = fetch_california_housing()
    X, y = h.data, h.target
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    best_k, best_cv = None, np.inf
    for k in [3, 5, 10, 15, 20, 30]:
        pipe = make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=k, weights="distance"))
        scores = cross_val_score(pipe, X_tr, y_tr, cv=5, scoring="neg_root_mean_squared_error")
        cv_rmse = -scores.mean()
        print(f"  k={k:>2}: CV RMSE={cv_rmse:.4f}")
        if cv_rmse < best_cv:
            best_cv, best_k = cv_rmse, k
    print(f"  -> Best k={best_k} (CV RMSE={best_cv:.4f})")


if __name__ == "__main__":
    demo_knn_regression()
    demo_knn_classification()
    demo_curse_dimensionality()
    demo_distance_metrics()
    demo_radius_neighbors()
    demo_knn_cv_selection()
