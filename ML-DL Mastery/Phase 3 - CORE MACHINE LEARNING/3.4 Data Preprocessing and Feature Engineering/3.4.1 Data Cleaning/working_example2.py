"""
Working Example 2: Data Cleaning — Missing Values, Outliers, Cal Housing
=========================================================================
SimpleImputer, KNNImputer, outlier detection and removal.

Run:  python working_example2.py
"""
from pathlib import Path
try:
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split
    from sklearn.impute import SimpleImputer, KNNImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_squared_error
    from sklearn.pipeline import make_pipeline, Pipeline
except ImportError:
    raise SystemExit("pip install numpy matplotlib scikit-learn")

OUTPUT = Path(__file__).parent / "output"
OUTPUT.mkdir(exist_ok=True)

def demo_missing_data():
    print("=== Imputation Strategies ===")
    h = fetch_california_housing()
    X, y = h.data.copy(), h.target
    rng = np.random.default_rng(42)

    # Inject 10% missing values
    mask = rng.random(X.shape) < 0.1
    X_miss = X.copy().astype(float)
    X_miss[mask] = np.nan
    print(f"  Missing values per feature: {np.isnan(X_miss).sum(0)}")

    X_train, X_test, y_train, y_test = train_test_split(X_miss, y, test_size=0.2, random_state=42)

    for strategy in ["mean", "median", "most_frequent"]:
        pipe = make_pipeline(SimpleImputer(strategy=strategy), StandardScaler(), Ridge(1.0))
        pipe.fit(X_train, y_train)
        rmse = mean_squared_error(y_test, pipe.predict(X_test))**0.5
        print(f"  SimpleImputer({strategy:14s}): RMSE={rmse:.4f}")

    # KNN imputer
    pipe_knn = make_pipeline(KNNImputer(n_neighbors=5), StandardScaler(), Ridge(1.0))
    pipe_knn.fit(X_train, y_train)
    rmse = mean_squared_error(y_test, pipe_knn.predict(X_test))**0.5
    print(f"  KNNImputer(k=5):              RMSE={rmse:.4f}")

    # Baseline (no missing)
    pipe_full = make_pipeline(StandardScaler(), Ridge(1.0))
    pipe_full.fit(*train_test_split(X, y, test_size=0.2, random_state=42)[:2])
    Xt, Xv, yt, yv = train_test_split(X, y, test_size=0.2, random_state=42)
    pipe_full.fit(Xt, yt)
    rmse_full = mean_squared_error(yv, pipe_full.predict(Xv))**0.5
    print(f"  No missing (baseline):        RMSE={rmse_full:.4f}")

def demo_outlier_removal():
    print("\n=== Outlier Removal (IQR) ===")
    h = fetch_california_housing()
    X, y = h.data, h.target
    print(f"  Before: {X.shape[0]} rows")

    # Remove rows where any feature is > 3 IQR from median
    Q1 = np.percentile(X, 25, axis=0)
    Q3 = np.percentile(X, 75, axis=0)
    IQR = Q3 - Q1
    mask = ((X >= Q1 - 3*IQR) & (X <= Q3 + 3*IQR)).all(axis=1)
    X_clean, y_clean = X[mask], y[mask]
    print(f"  After IQR(3x) removal: {X_clean.shape[0]} rows  (removed {(~mask).sum()})")

    X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size=0.2, random_state=42)
    pipe = make_pipeline(StandardScaler(), Ridge(1.0))
    pipe.fit(X_train, y_train)
    rmse = mean_squared_error(y_test, pipe.predict(X_test))**0.5
    print(f"  Ridge on cleaned data: RMSE={rmse:.4f}")

if __name__ == "__main__":
    demo_missing_data()
    demo_outlier_removal()
