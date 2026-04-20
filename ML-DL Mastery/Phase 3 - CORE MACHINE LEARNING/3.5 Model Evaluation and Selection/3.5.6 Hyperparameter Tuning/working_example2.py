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

def demo_optuna_style():
    """Manual Bayesian-style search using random sampling + early stopping."""
    print("\n=== Manual Bayesian-style Search (warmstart) ===")
    h = fetch_california_housing()
    X, y = h.data, h.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rng = np.random.default_rng(0)
    best_rmse, best_params = np.inf, {}
    n_trials = 30
    for _ in range(n_trials):
        n_est = int(rng.integers(50, 400))
        lr    = float(rng.uniform(0.01, 0.3))
        depth = int(rng.integers(2, 8))
        gb = GradientBoostingRegressor(n_estimators=n_est, learning_rate=lr,
                                       max_depth=depth, random_state=42)
        gb.fit(X_train, y_train)
        rmse = mean_squared_error(y_test, gb.predict(X_test))**0.5
        if rmse < best_rmse:
            best_rmse = rmse
            best_params = {"n_estimators": n_est, "lr": round(lr,3), "max_depth": depth}
    print(f"  Best RMSE={best_rmse:.4f}  Params={best_params}")


def demo_learning_curves():
    """Plot learning curves (train vs validation score vs training size)."""
    print("\n=== Learning Curves ===")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path
    from sklearn.model_selection import learning_curve
    out = Path(__file__).parent / "output"; out.mkdir(exist_ok=True)
    h = fetch_california_housing()
    X, y = h.data, h.target
    pipe = make_pipeline(StandardScaler(), Ridge(1.0))
    sizes, tr_scores, val_scores = learning_curve(
        pipe, X, y, cv=5, scoring="neg_root_mean_squared_error",
        train_sizes=np.linspace(0.1, 1.0, 8), n_jobs=-1)
    tr_rmse  = -tr_scores.mean(axis=1)
    val_rmse = -val_scores.mean(axis=1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(sizes, tr_rmse,  "o-", label="Train RMSE")
    ax.plot(sizes, val_rmse, "s-", label="Val RMSE")
    ax.set_xlabel("Training size"); ax.set_ylabel("RMSE")
    ax.set_title("Learning Curves — Ridge"); ax.legend()
    fig.savefig(out / "learning_curves.png", dpi=120, bbox_inches="tight")
    plt.close(fig); print("  Saved: learning_curves.png")
    print(f"  Min val RMSE={val_rmse.min():.4f} at n={sizes[val_rmse.argmin()]}")


if __name__ == "__main__":
    demo_grid_search()
    demo_random_search()
    demo_halving()
    demo_optuna_style()
    demo_learning_curves()
