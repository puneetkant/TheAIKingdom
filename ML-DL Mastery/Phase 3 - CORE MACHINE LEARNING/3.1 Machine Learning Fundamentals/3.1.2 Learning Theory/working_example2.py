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
    import matplotlib.pyplot as plt
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge, Lasso
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

if __name__ == "__main__":
    demo_hoeffding()
    demo_regularisation_path()
    demo_cross_validation()
