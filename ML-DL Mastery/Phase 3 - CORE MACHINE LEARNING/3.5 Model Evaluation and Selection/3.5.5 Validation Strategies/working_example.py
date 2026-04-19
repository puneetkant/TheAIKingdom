"""
Working Example: Validation Strategies
Covers hold-out, k-fold, stratified k-fold, leave-one-out, time-series CV,
group CV, nested CV, and the bias-variance tradeoff of validation choices.
"""
import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import (KFold, StratifiedKFold, LeaveOneOut,
                                      cross_val_score, cross_validate,
                                      train_test_split, GroupKFold,
                                      TimeSeriesSplit, ShuffleSplit,
                                      RepeatedStratifiedKFold, GridSearchCV)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_validation")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -- 1. Hold-out validation ----------------------------------------------------
def holdout():
    print("=== Hold-out Validation ===")
    X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, random_state=0)

    # Various split ratios
    print(f"  {'Test size':<12} {'Train n':<10} {'Test n':<10} {'Acc (single run)'}")
    for test_size in [0.1, 0.2, 0.3, 0.4]:
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=42)
        model = Pipeline([("sc", StandardScaler()),
                          ("lr", LogisticRegression(max_iter=500))])
        model.fit(X_tr, y_tr)
        acc = model.score(X_te, y_te)
        print(f"  {test_size:<12} {len(X_tr):<10} {len(X_te):<10} {acc:.4f}")

    print(f"\n  Pros: Fast; Cons: High variance (single evaluation)")


# -- 2. K-Fold cross-validation ------------------------------------------------
def kfold_cv():
    print("\n=== K-Fold Cross-Validation ===")
    X, y = make_classification(n_samples=500, n_features=10, n_informative=5, random_state=0)
    pipe  = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(max_iter=500))])

    print(f"\n  {'k':<5} {'Mean Acc':>12} {'Std':>10} {'Spread = 2×Std'}")
    for k in [3, 5, 10, 20]:
        cv = KFold(n_splits=k, shuffle=True, random_state=0)
        scores = cross_val_score(pipe, X, y, cv=cv)
        print(f"  {k:<5} {scores.mean():>12.4f} {scores.std():>10.4f} {2*scores.std():>14.4f}")

    # Fold-by-fold for k=5
    print(f"\n  5-fold detailed:")
    cv = KFold(n_splits=5, shuffle=True, random_state=0)
    for fold, (tr_idx, te_idx) in enumerate(cv.split(X), 1):
        pipe.fit(X[tr_idx], y[tr_idx])
        acc = pipe.score(X[te_idx], y[te_idx])
        print(f"    Fold {fold}: train={len(tr_idx)}  test={len(te_idx)}  acc={acc:.4f}")


# -- 3. Stratified K-Fold ------------------------------------------------------
def stratified_kfold():
    print("\n=== Stratified K-Fold ===")
    print("  Preserves class proportion in each fold (essential for imbalanced)")
    X, y = make_classification(n_samples=300, n_features=5, n_informative=3,
                                random_state=0, weights=[0.8, 0.2])
    print(f"  Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    print(f"\n  {'Fold':<6} {'Standard KF (class 1%)':>24} {'Stratified KF (class 1%)':>26}")
    kf   = KFold(n_splits=5, shuffle=True, random_state=0)
    skf  = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    for (_, te1), (_, te2) in zip(kf.split(X), skf.split(X, y)):
        pct1 = y[te1].mean()
        pct2 = y[te2].mean()
        print(f"  {' ':<6} {pct1:>24.4f} {pct2:>26.4f}")


# -- 4. Leave-One-Out (LOO) ----------------------------------------------------
def loo_cv():
    print("\n=== Leave-One-Out Cross-Validation ===")
    print("  k = n; each sample is a test set once")
    print("  Pros: Unbiased for small n; Cons: Very slow for large n")

    X, y = make_classification(n_samples=80, n_features=5, n_informative=3, random_state=0)
    pipe  = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(max_iter=500))])
    loo   = LeaveOneOut()
    scores = cross_val_score(pipe, X, y, cv=loo)
    print(f"\n  n={len(X)}, n_evaluations={loo.get_n_splits(X)}")
    print(f"  Mean acc = {scores.mean():.4f}  Std = {scores.std():.4f}")
    print(f"  (LOO has high variance; prefer 5-10 fold for n>100)")


# -- 5. Repeated K-Fold --------------------------------------------------------
def repeated_kfold():
    print("\n=== Repeated K-Fold ===")
    print("  Repeat k-fold R times with different shuffles -> more stable estimate")
    X, y = make_classification(n_samples=300, n_features=10, n_informative=5, random_state=0)
    pipe  = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(max_iter=500))])

    print(f"\n  {'Method':<30} {'Mean Acc':>12} {'Std':>10}")
    kf  = KFold(n_splits=5, shuffle=True, random_state=0)
    rkf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=0)

    for name, cv in [("5-Fold (1 repeat)",    kf),
                     ("5-Fold (10 repeats)",   rkf)]:
        sc = cross_val_score(pipe, X, y, cv=cv)
        print(f"  {name:<30} {sc.mean():>12.4f} {sc.std():>10.4f}")


# -- 6. Time Series CV ---------------------------------------------------------
def time_series_cv():
    print("\n=== Time Series Cross-Validation ===")
    print("  No future data leakage: test always after training period")
    rng = np.random.default_rng(0)
    n   = 200
    t   = np.arange(n)
    y   = np.sin(0.1 * t) + 0.5 * rng.standard_normal(n)
    X   = np.column_stack([t, t**2, np.sin(0.1*t)])

    tss = TimeSeriesSplit(n_splits=5)
    print(f"\n  TimeSeriesSplit (n_splits=5):")
    for fold, (tr_idx, te_idx) in enumerate(tss.split(X), 1):
        print(f"    Fold {fold}: train=[{tr_idx[0]}:{tr_idx[-1]}]  test=[{te_idx[0]}:{te_idx[-1]}]")

    sc = cross_val_score(Ridge(), X, y, cv=tss, scoring="r2")
    print(f"\n  R² scores: {sc.round(4)}")
    print(f"  Mean R²: {sc.mean():.4f}")


# -- 7. Group K-Fold -----------------------------------------------------------
def group_kfold():
    print("\n=== Group K-Fold ===")
    print("  Ensures all samples from a group are in the same fold")
    print("  Use when samples from same patient/user/subject must not leak")
    X, y = make_classification(n_samples=300, n_features=5, n_informative=3, random_state=0)
    groups = np.repeat(np.arange(30), 10)  # 30 subjects, 10 samples each

    gkf = GroupKFold(n_splits=5)
    print(f"\n  Fold group assignments:")
    for fold, (tr_idx, te_idx) in enumerate(gkf.split(X, y, groups), 1):
        tr_groups = set(groups[tr_idx])
        te_groups = set(groups[te_idx])
        overlap   = tr_groups & te_groups
        print(f"    Fold {fold}: test_groups={sorted(te_groups)}  overlap={overlap or 'none'}")


# -- 8. Nested CV (hyperparameter + model selection) --------------------------
def nested_cv():
    print("\n=== Nested Cross-Validation ===")
    print("  Outer loop: unbiased performance estimate")
    print("  Inner loop: hyperparameter selection (avoids test-set leakage)")

    X, y = make_classification(n_samples=300, n_features=10, n_informative=5, random_state=1)
    pipe  = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(max_iter=500))])
    param_grid = {"lr__C": [0.01, 0.1, 1.0, 10.0]}

    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)

    outer_scores = []
    for fold, (tr_idx, te_idx) in enumerate(outer.split(X, y), 1):
        gs = GridSearchCV(pipe, param_grid, cv=inner, scoring="accuracy")
        gs.fit(X[tr_idx], y[tr_idx])
        score = gs.score(X[te_idx], y[te_idx])
        outer_scores.append(score)
        print(f"  Fold {fold}: best_C={gs.best_params_['lr__C']}  test_acc={score:.4f}")

    print(f"\n  Nested CV estimate: {np.mean(outer_scores):.4f} ± {np.std(outer_scores):.4f}")
    print(f"  (This is an unbiased estimate of the model-selection pipeline)")


if __name__ == "__main__":
    holdout()
    kfold_cv()
    stratified_kfold()
    loo_cv()
    repeated_kfold()
    time_series_cv()
    group_kfold()
    nested_cv()
