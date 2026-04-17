"""
Working Example 2: Feature Selection — Filter, Wrapper, Embedded Methods
=========================================================================
SelectKBest, RFE, RFECV, permutation_importance, SelectFromModel.

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
    from sklearn.feature_selection import (SelectKBest, f_regression,
                                           RFE, RFECV, SelectFromModel)
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import mean_squared_error
    from sklearn.pipeline import make_pipeline
except ImportError:
    raise SystemExit("pip install numpy matplotlib scikit-learn")

OUTPUT = Path(__file__).parent / "output"
OUTPUT.mkdir(exist_ok=True)

def demo_filter():
    print("=== Filter: SelectKBest (f_regression) ===")
    h = fetch_california_housing()
    X, y = h.data, h.target
    for k in [2, 4, 6, 8]:
        pipe = make_pipeline(StandardScaler(), SelectKBest(f_regression, k=k), Ridge(1.0))
        scores = cross_val_score(pipe, X, y, cv=5, scoring="neg_root_mean_squared_error")
        print(f"  k={k}: CV RMSE={-scores.mean():.4f} ± {scores.std():.4f}")

    selector = SelectKBest(f_regression, k=4)
    selector.fit(X, y)
    selected = np.array(h.feature_names)[selector.get_support()]
    print(f"  Top-4 features: {list(selected)}")

def demo_rfe():
    print("\n=== Wrapper: RFE with Ridge ===")
    h = fetch_california_housing()
    X, y = h.data, h.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_ts_s = scaler.transform(X_test)

    rfe = RFE(Ridge(1.0), n_features_to_select=4)
    rfe.fit(X_tr_s, y_train)
    selected = np.array(h.feature_names)[rfe.support_]
    print(f"  RFE selected: {list(selected)}")
    rmse = mean_squared_error(y_test, rfe.predict(X_ts_s))**0.5
    print(f"  Test RMSE (4 features): {rmse:.4f}")

def demo_embedded():
    print("\n=== Embedded: Lasso (SelectFromModel) ===")
    h = fetch_california_housing()
    X, y = h.data, h.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    for alpha in [0.01, 0.05, 0.1, 0.5]:
        pipe = make_pipeline(StandardScaler(), Lasso(alpha=alpha, max_iter=5000))
        pipe.fit(X_train, y_train)
        coefs = pipe.named_steps["lasso"].coef_
        nonzero = (coefs != 0).sum()
        rmse = mean_squared_error(y_test, pipe.predict(X_test))**0.5
        print(f"  Lasso(α={alpha:.2f}): {nonzero}/8 features kept  RMSE={rmse:.4f}")

def demo_permutation():
    print("\n=== Permutation Importance (RF) ===")
    h = fetch_california_housing()
    X, y = h.data, h.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestRegressor(50, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    result = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42)
    order = result.importances_mean.argsort()[::-1]
    for i in order:
        print(f"  {h.feature_names[i]:15s}: {result.importances_mean[i]:.4f} ± {result.importances_std[i]:.4f}")

if __name__ == "__main__":
    demo_filter()
    demo_rfe()
    demo_embedded()
    demo_permutation()
