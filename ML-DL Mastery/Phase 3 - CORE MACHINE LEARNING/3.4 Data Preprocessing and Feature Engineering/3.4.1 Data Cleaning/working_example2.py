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

def demo_duplicate_detection():
    """Detect and remove duplicate rows — a common real-world issue."""
    print("\n=== Duplicate Row Detection ===")
    h = fetch_california_housing()
    X, y = h.data.copy(), h.target.copy()
    # Inject duplicates
    dup_idx = np.arange(50)
    X_dup = np.vstack([X, X[dup_idx]])
    y_dup = np.concatenate([y, y[dup_idx]])
    print(f"  Before dedup: {len(X_dup)} rows")
    # Use pandas-style with numpy: identify duplicates by finding unique rows
    _, unique_idx = np.unique(X_dup, axis=0, return_index=True)
    X_dedup = X_dup[np.sort(unique_idx)]
    y_dedup = y_dup[np.sort(unique_idx)]
    print(f"  After dedup:  {len(X_dedup)} rows  (removed {len(X_dup)-len(X_dedup)})")


def demo_schema_validation():
    """Check data schema: ranges, types, and unexpected values."""
    print("\n=== Schema / Range Validation ===")
    h = fetch_california_housing()
    X = h.data
    # Expected non-negative columns
    for i, name in enumerate(h.feature_names):
        neg_count = (X[:, i] < 0).sum()
        min_v, max_v = X[:, i].min(), X[:, i].max()
        print(f"  {name:15s}: min={min_v:.3f}  max={max_v:.3f}  negatives={neg_count}")


if __name__ == "__main__":
    demo_missing_data()
    demo_outlier_removal()
    demo_duplicate_detection()
    demo_schema_validation()
