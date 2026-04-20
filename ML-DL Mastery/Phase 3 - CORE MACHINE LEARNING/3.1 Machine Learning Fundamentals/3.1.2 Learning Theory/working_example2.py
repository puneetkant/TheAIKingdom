"""
Working Example 2: Learning Theory — PAC Bounds, VC Dimension, Regularisation
===============================================================================
Hoeffding's inequality, empirical risk minimisation, regularisation path,
generalisation bound illustration with Cal Housing.

Run:  python working_example2.py
"""
from pathlib import Path
try:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.datasets import (fetch_california_housing, make_blobs,
                                  make_circles, make_classification)
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge, Lasso, LogisticRegression
    from sklearn.svm import LinearSVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import mean_squared_error
except ImportError:
    raise SystemExit("pip install numpy matplotlib scikit-learn")

OUTPUT = Path(__file__).parent / "output"
OUTPUT.mkdir(exist_ok=True)

def demo_hoeffding():
    print("=== Hoeffding Bound (Generalisation) ===")
    import math
    # With n samples and delta confidence, generalisation error <= train_error + sqrt(log(2/delta)/(2n))
    delta = 0.05
    for n in [100, 500, 1000, 5000, 10000]:
        bound = math.sqrt(math.log(2/delta) / (2*n))
        print(f"  n={n:>6}: gen gap <= {bound:.4f}  (95% confidence)")

def demo_regularisation_path():
    print("\n=== Regularisation Path (Ridge vs Lasso) ===")
    housing = fetch_california_housing()
    X, y = housing.data, housing.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    alphas = np.logspace(-3, 3, 30)
    ridge_mse, lasso_mse = [], []
    for a in alphas:
        r = make_pipeline(StandardScaler(), Ridge(alpha=a))
        r.fit(X_train, y_train)
        ridge_mse.append(mean_squared_error(y_test, r.predict(X_test)))

        try:
            l = make_pipeline(StandardScaler(), Lasso(alpha=a, max_iter=5000))
            l.fit(X_train, y_train)
            lasso_mse.append(mean_squared_error(y_test, l.predict(X_test)))
        except Exception:
            lasso_mse.append(np.nan)

    best_ridge = alphas[np.argmin(ridge_mse)]
    best_lasso = alphas[np.nanargmin(lasso_mse)]
    print(f"  Best Ridge alpha: {best_ridge:.4f}  MSE: {min(ridge_mse):.4f}")
    print(f"  Best Lasso alpha: {best_lasso:.4f}  MSE: {np.nanmin(lasso_mse):.4f}")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogx(alphas, ridge_mse, "o-", label="Ridge")
    ax.semilogx(alphas, lasso_mse, "s-", label="Lasso")
    ax.axvline(best_ridge, ls="--", color="C0", alpha=0.5)
    ax.axvline(best_lasso, ls="--", color="C1", alpha=0.5)
    ax.set_xlabel("alpha (regularisation strength)"); ax.set_ylabel("Test MSE")
    ax.set_title("Regularisation Path"); ax.legend()
    fig.savefig(OUTPUT / "regularisation_path.png", dpi=120, bbox_inches="tight")
    plt.close(fig); print(f"  Saved: regularisation_path.png")

def demo_cross_validation():
    print("\n=== Cross-Validation (k-fold) ===")
    housing = fetch_california_housing()
    X, y = housing.data, housing.target
    pipe = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    for k in [3, 5, 10]:
        scores = cross_val_score(pipe, X, y, cv=k, scoring="neg_mean_squared_error")
        mse = -scores.mean(); std = scores.std()
        print(f"  {k}-fold CV: MSE={mse:.4f} ± {std:.4f}")

def demo_vc_dimension():
    print("\n=== VC Dimension: Can LinearSVC Shatter 3 Points? ===")
    import warnings
    collinear     = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    non_collinear = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]])
    for desc, pts in [("Collinear    ", collinear),
                      ("Non-collinear", non_collinear)]:
        perfect = 0
        for bits in range(8):
            y = np.array([(bits >> j) & 1 for j in range(3)])
            if len(np.unique(y)) < 2:
                continue  # skip degenerate all-same-class labelings
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    clf = LinearSVC(C=1e8, max_iter=20000, dual=False)
                    clf.fit(pts, y)
                if np.all(clf.predict(pts) == y):
                    perfect += 1
            except Exception:
                pass
        shattered = "YES" if perfect == 6 else "NO"
        print(f"  {desc}: {perfect}/6 distinct-class labelings classified perfectly  "
              f"=> shattered: {shattered}")


def demo_no_free_lunch():
    print("\n=== No Free Lunch: Each Model Wins on a Different Dataset ===")
    rng = np.random.RandomState(42)
    X_xor = rng.randn(300, 2)
    y_xor = ((X_xor[:, 0] > 0) ^ (X_xor[:, 1] > 0)).astype(int)
    X_blobs,   y_blobs   = make_blobs(n_samples=300, centers=3, random_state=42)
    X_circles, y_circles = make_circles(n_samples=300, noise=0.1, factor=0.4, random_state=42)
    datasets = [
        ("Blobs",   X_blobs,   y_blobs),
        ("Circles", X_circles, y_circles),
        ("XOR",     X_xor,     y_xor),
    ]
    models = [
        ("LogisticReg",  LogisticRegression(max_iter=1000)),
        ("DecisionTree", DecisionTreeClassifier(max_depth=4)),
        ("KNN-5",        KNeighborsClassifier(n_neighbors=5)),
    ]
    header = f"  {'Dataset':10s}"
    for m_name, _ in models:
        header += f"  {m_name:14s}"
    print(header)
    for ds_name, X, y in datasets:
        row = f"  {ds_name:10s}"
        for _, model in models:
            acc = cross_val_score(model, X, y, cv=5).mean()
            row += f"  {acc:14.4f}"
        print(row)


def demo_sample_complexity():
    print("\n=== Sample Complexity: Accuracy Grows with Training Set Size ===")
    X_full, y_full = make_classification(
        n_samples=2000, n_features=20, random_state=42
    )
    print(f"  {'n_train':>8}  {'mean_cv_acc':>12}")
    for n in [20, 50, 100, 200, 500, 1000]:
        X_sub, _, y_sub, _ = train_test_split(
            X_full, y_full, train_size=n, random_state=42
        )
        scores = cross_val_score(LogisticRegression(max_iter=1000), X_sub, y_sub, cv=5)
        print(f"  {n:8d}  {scores.mean():12.4f}")


if __name__ == "__main__":
    demo_hoeffding()
    demo_regularisation_path()
    demo_cross_validation()
    demo_vc_dimension()
    demo_no_free_lunch()
    demo_sample_complexity()
