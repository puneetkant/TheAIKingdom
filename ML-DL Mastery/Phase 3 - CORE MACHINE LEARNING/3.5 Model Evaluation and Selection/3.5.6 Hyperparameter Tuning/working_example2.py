"""
Working Example 2: Hyperparameter Tuning — GridSearchCV, RandomizedSearchCV, HalvingSearch
============================================================================================
Grid search, random search, successive halving on Cal Housing.

Run:  python working_example2.py
"""
from pathlib import Path
try:
    import numpy as np
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import (train_test_split, GridSearchCV,
                                          RandomizedSearchCV, KFold)
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline, Pipeline
    from sklearn.metrics import mean_squared_error
    from scipy.stats import randint, uniform
    import warnings; warnings.filterwarnings("ignore")
except ImportError:
    raise SystemExit("pip install numpy scipy scikit-learn")

def demo_grid_search():
    print("=== GridSearchCV (Ridge alpha) ===")
    h = fetch_california_housing()
    X, y = h.data, h.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipe = Pipeline([("sc", StandardScaler()), ("model", Ridge())])
    param_grid = {"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}
    gs = GridSearchCV(pipe, param_grid, cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1)
    gs.fit(X_train, y_train)
    print(f"  Best alpha={gs.best_params_['model__alpha']}  CV RMSE={-gs.best_score_:.4f}")
    rmse = mean_squared_error(y_test, gs.predict(X_test))**0.5
    print(f"  Test RMSE: {rmse:.4f}")

def demo_random_search():
    print("\n=== RandomizedSearchCV (RF) ===")
    h = fetch_california_housing()
    X, y = h.data, h.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    param_dist = {
        "model__n_estimators": randint(50, 300),
        "model__max_depth":    [None, 5, 10, 20],
        "model__min_samples_split": randint(2, 20),
        "model__max_features": uniform(0.3, 0.7),
    }
    pipe = Pipeline([("sc", StandardScaler()), ("model", RandomForestRegressor(random_state=42, n_jobs=-1))])
    rs = RandomizedSearchCV(pipe, param_dist, n_iter=20, cv=3,
                             scoring="neg_root_mean_squared_error",
                             random_state=42, n_jobs=-1)
    rs.fit(X_train, y_train)
    print(f"  Best params: {rs.best_params_}")
    print(f"  Best CV RMSE: {-rs.best_score_:.4f}")
    rmse = mean_squared_error(y_test, rs.predict(X_test))**0.5
    print(f"  Test RMSE: {rmse:.4f}")

def demo_halving():
    print("\n=== HalvingRandomSearchCV (fast) ===")
    try:
        from sklearn.experimental import enable_halving_search_cv  # noqa
        from sklearn.model_selection import HalvingRandomSearchCV
    except ImportError:
        print("  HalvingRandomSearchCV not available in this sklearn version")
        return
    h = fetch_california_housing()
    X, y = h.data, h.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    param_dist = {"n_estimators": randint(50, 500), "max_depth": [None, 5, 10, 20, 30]}
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    hs = HalvingRandomSearchCV(rf, param_dist, cv=3,
                                scoring="neg_root_mean_squared_error",
                                factor=3, random_state=42, n_jobs=-1)
    hs.fit(X_train, y_train)
    print(f"  Best: {hs.best_params_}  CV RMSE: {-hs.best_score_:.4f}")
    rmse = mean_squared_error(y_test, hs.predict(X_test))**0.5
    print(f"  Test RMSE: {rmse:.4f}")

if __name__ == "__main__":
    demo_grid_search()
    demo_random_search()
    demo_halving()
