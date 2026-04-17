"""
Working Example 2: Feature Engineering — Polynomial, Interaction, Log Transform
=================================================================================
PolynomialFeatures, log/sqrt transforms, interaction terms, domain features.

Run:  python working_example2.py
"""
from pathlib import Path
try:
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, PolynomialFeatures, FunctionTransformer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline, Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.metrics import mean_squared_error, r2_score
except ImportError:
    raise SystemExit("pip install numpy matplotlib scikit-learn")

OUTPUT = Path(__file__).parent / "output"
OUTPUT.mkdir(exist_ok=True)

def demo_baseline(X_train, X_test, y_train, y_test):
    pipe = make_pipeline(StandardScaler(), Ridge(1.0))
    pipe.fit(X_train, y_train)
    rmse = mean_squared_error(y_test, pipe.predict(X_test))**0.5
    r2   = r2_score(y_test, pipe.predict(X_test))
    return rmse, r2

def demo_polynomial():
    print("=== Polynomial Features ===")
    h = fetch_california_housing()
    X, y = h.data, h.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    base_rmse, base_r2 = demo_baseline(X_train, X_test, y_train, y_test)
    print(f"  Baseline Ridge:         RMSE={base_rmse:.4f}  R²={base_r2:.4f}")

    for d in [2]:
        pipe = make_pipeline(StandardScaler(), PolynomialFeatures(d, include_bias=False),
                             StandardScaler(), Ridge(1.0))
        pipe.fit(X_train, y_train)
        rmse = mean_squared_error(y_test, pipe.predict(X_test))**0.5
        r2   = r2_score(y_test, pipe.predict(X_test))
        n_features = pipe.named_steps["polynomialfeatures"].n_output_features_
        print(f"  PolynomialFeatures(d={d}): RMSE={rmse:.4f}  R²={r2:.4f}  n_features={n_features}")

def demo_log_transform():
    print("\n=== Log / Sqrt Transforms ===")
    h = fetch_california_housing()
    X, y = h.data, h.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Log transform all features (clip to avoid log(0))
    log_pipe = make_pipeline(
        FunctionTransformer(lambda X: np.log1p(np.clip(X, 0, None)), validate=True),
        StandardScaler(), Ridge(1.0)
    )
    log_pipe.fit(X_train, y_train)
    rmse_log = mean_squared_error(y_test, log_pipe.predict(X_test))**0.5
    print(f"  Log1p + Ridge: RMSE={rmse_log:.4f}")

def demo_domain_features():
    print("\n=== Domain Feature Engineering (Cal Housing) ===")
    h = fetch_california_housing()
    X, y = h.data, h.target
    # Feature engineering: rooms/household, bedrooms/rooms, population density
    Xf = np.column_stack([
        X,
        X[:, 2] / (X[:, 5] + 1),   # rooms_per_household
        X[:, 3] / (X[:, 2] + 1),   # bedrooms_per_room
        X[:, 4] / (X[:, 5] + 1),   # pop_per_household
    ])
    X_train, X_test, y_train, y_test = train_test_split(Xf, y, test_size=0.2, random_state=42)
    pipe = make_pipeline(StandardScaler(), Ridge(1.0))
    pipe.fit(X_train, y_train)
    rmse = mean_squared_error(y_test, pipe.predict(X_test))**0.5
    r2   = r2_score(y_test, pipe.predict(X_test))
    print(f"  Domain features + Ridge: RMSE={rmse:.4f}  R²={r2:.4f}  (from 8 → {Xf.shape[1]} features)")

if __name__ == "__main__":
    demo_polynomial()
    demo_log_transform()
    demo_domain_features()
