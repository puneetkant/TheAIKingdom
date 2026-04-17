"""
Working Example 2: Feature Scaling — StandardScaler, MinMax, RobustScaler
==========================================================================
Effect of scaling on KNN, SVM, Ridge regression performance.

Run:  python working_example2.py
"""
from pathlib import Path
try:
    import numpy as np
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.svm import LinearSVR
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_squared_error
    from sklearn.pipeline import make_pipeline
except ImportError:
    raise SystemExit("pip install numpy scikit-learn")

def demo_scaling_comparison():
    print("=== Scaling Comparison (Cal Housing) ===")
    h = fetch_california_housing()
    X, y = h.data, h.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scalers = {
        "None":     None,
        "Standard": StandardScaler(),
        "MinMax":   MinMaxScaler(),
        "Robust":   RobustScaler(),
    }
    models = {
        "Ridge":   Ridge(1.0),
        "KNN(10)": KNeighborsRegressor(10),
    }

    for model_name, model in models.items():
        print(f"\n  Model: {model_name}")
        for scaler_name, scaler in scalers.items():
            if scaler is None:
                import sklearn; from sklearn.base import clone
                m = sklearn.base.clone(model)
                m.fit(X_train, y_train)
                rmse = mean_squared_error(y_test, m.predict(X_test))**0.5
            else:
                pipe = make_pipeline(scaler, model)
                pipe.fit(X_train, y_train)
                rmse = mean_squared_error(y_test, pipe.predict(X_test))**0.5
            print(f"    {scaler_name:10s}: RMSE={rmse:.4f}")

def demo_feature_range():
    print("\n=== Feature Value Ranges (Cal Housing) ===")
    h = fetch_california_housing()
    X = h.data
    print(f"  {'Feature':20s}  {'min':>10}  {'max':>10}  {'std':>10}")
    for name, col in zip(h.feature_names, X.T):
        print(f"  {name:20s}  {col.min():10.2f}  {col.max():10.2f}  {col.std():10.2f}")

if __name__ == "__main__":
    demo_feature_range()
    demo_scaling_comparison()
