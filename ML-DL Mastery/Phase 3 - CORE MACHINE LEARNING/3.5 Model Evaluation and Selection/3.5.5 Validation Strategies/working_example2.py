"""
Working Example 2: Validation Strategies — KFold, StratifiedKFold, TimeSeriesSplit
===================================================================================
Cross-validation strategies and their effect on score estimates.

Run:  python working_example2.py
"""
from pathlib import Path
try:
    import numpy as np
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import (KFold, StratifiedKFold, RepeatedKFold,
                                          TimeSeriesSplit, cross_val_score,
                                          cross_validate, ShuffleSplit)
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge, LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import make_pipeline
    import warnings; warnings.filterwarnings("ignore")
except ImportError:
    raise SystemExit("pip install numpy scikit-learn")

def demo_cv_strategies():
    print("=== Cross-Validation Strategies ===")
    h = fetch_california_housing()
    X, y = h.data, h.target
    pipe = make_pipeline(StandardScaler(), Ridge(1.0))

    strategies = {
        "KFold(5)":          KFold(5, shuffle=True, random_state=42),
        "KFold(10)":         KFold(10, shuffle=True, random_state=42),
        "RepeatedKFold(5x3)":RepeatedKFold(n_splits=5, n_repeats=3, random_state=42),
        "ShuffleSplit(10)":  ShuffleSplit(10, test_size=0.2, random_state=42),
    }
    print(f"  {'Strategy':25s}  {'Mean RMSE':>12}  {'Std':>8}  {'n_evals':>8}")
    for name, cv in strategies.items():
        scores = cross_val_score(pipe, X, y, cv=cv,
                                  scoring="neg_root_mean_squared_error")
        print(f"  {name:25s}  {-scores.mean():>12.4f}  {scores.std():>8.4f}  {len(scores):>8}")

def demo_time_series_cv():
    print("\n=== TimeSeriesSplit (temporal data) ===")
    h = fetch_california_housing()
    X, y = h.data, h.target
    pipe = make_pipeline(StandardScaler(), Ridge(1.0))

    tscv = TimeSeriesSplit(n_splits=5)
    scores = cross_val_score(pipe, X, y, cv=tscv,
                              scoring="neg_root_mean_squared_error")
    print(f"  TimeSeriesSplit(5): RMSE per fold = {[-s.round(4) for s in scores]}")
    print(f"  Mean: {-scores.mean():.4f}  Std: {scores.std():.4f}")

def demo_cross_validate_detail():
    print("\n=== cross_validate (train+test scores) ===")
    h = fetch_california_housing()
    X, y = h.data, h.target
    pipe = make_pipeline(StandardScaler(), Ridge(1.0))
    cv_results = cross_validate(pipe, X, y, cv=5,
                                 scoring="neg_root_mean_squared_error",
                                 return_train_score=True)
    train_rmse = -cv_results["train_score"]
    test_rmse  = -cv_results["test_score"]
    print(f"  Train RMSE: {train_rmse.mean():.4f} ± {train_rmse.std():.4f}")
    print(f"  Test  RMSE: {test_rmse.mean():.4f} ± {test_rmse.std():.4f}")
    gap = test_rmse.mean() - train_rmse.mean()
    print(f"  Generalization gap: {gap:.4f}  ({'overfit' if gap>0.05 else 'ok'})")

def demo_stratified_cv():
    """StratifiedKFold preserves class distribution across folds."""
    print("\n=== StratifiedKFold vs plain KFold ===")
    h = fetch_california_housing()
    X, y = h.data, (h.target > np.median(h.target)).astype(int)
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path
    out = Path(__file__).parent / "output"; out.mkdir(exist_ok=True)
    for CVClass, name in [(KFold, "KFold"), (StratifiedKFold, "StratifiedKFold")]:
        cv = CVClass(5, shuffle=True, random_state=42)
        scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
        print(f"  {name:20s}: AUC={scores.mean():.4f} ± {scores.std():.4f}")


def demo_nested_cv():
    """Nested CV: outer loop estimates generalisation, inner loop selects model."""
    print("\n=== Nested Cross-Validation ===")
    from sklearn.model_selection import GridSearchCV
    h = fetch_california_housing()
    X, y = h.data, h.target
    pipe = make_pipeline(StandardScaler(), Ridge())
    inner_cv  = KFold(3, shuffle=True, random_state=1)
    outer_cv  = KFold(5, shuffle=True, random_state=2)
    grid = {"ridge__alpha": [0.01, 0.1, 1.0, 10.0]}
    gs   = GridSearchCV(pipe, grid, cv=inner_cv, scoring="neg_root_mean_squared_error")
    scores = cross_val_score(gs, X, y, cv=outer_cv,
                              scoring="neg_root_mean_squared_error")
    print(f"  Nested CV RMSE: {-scores.mean():.4f} ± {scores.std():.4f}")
    print("  (Inner loop picks alpha; outer loop estimates true generalisation)")


def demo_leave_one_group_out():
    """GroupKFold — keep entire groups (e.g. geographic blocks) together."""
    print("\n=== GroupKFold (geographic blocks) ===")
    from sklearn.model_selection import GroupKFold
    h = fetch_california_housing()
    X, y = h.data, h.target
    # Use latitude quintiles as groups
    groups = np.digitize(h.data[:, 6], np.percentile(h.data[:, 6], [20, 40, 60, 80]))
    pipe  = make_pipeline(StandardScaler(), Ridge(1.0))
    gkf   = GroupKFold(n_splits=5)
    scores = cross_val_score(pipe, X, y, cv=gkf, groups=groups,
                              scoring="neg_root_mean_squared_error")
    print(f"  GroupKFold RMSE: {-scores.mean():.4f} ± {scores.std():.4f}")


if __name__ == "__main__":
    demo_cv_strategies()
    demo_time_series_cv()
    demo_cross_validate_detail()
    demo_stratified_cv()
    demo_nested_cv()
    demo_leave_one_group_out()
