"""
Working Example: Feature Scaling and Normalisation
Covers StandardScaler, MinMaxScaler, RobustScaler, Normalizer,
MaxAbsScaler, PowerTransformer, QuantileTransformer, and when to use each.
"""
import numpy as np
from sklearn.preprocessing import (StandardScaler, MinMaxScaler, RobustScaler,
                                   Normalizer, MaxAbsScaler, PowerTransformer,
                                   QuantileTransformer)
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification, load_boston
from sklearn.model_selection import train_test_split, cross_val_score
import os


# -- 1. Why scaling matters ----------------------------------------------------
def why_scaling():
    print("=== Why Feature Scaling Matters ===")
    rng = np.random.default_rng(0)
    n   = 200
    # Feature 1: age [20,80], Feature 2: salary [20000,100000]
    X = np.column_stack([
        rng.uniform(20, 80, n),
        rng.uniform(20000, 100000, n),
    ])
    y = ((X[:,0]/60 + X[:,1]/100000) > 1).astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)

    print("  Unscaled features: large scale difference")
    print(f"    age: [{X[:,0].min():.0f}, {X[:,0].max():.0f}]")
    print(f"    salary: [{X[:,1].min():.0f}, {X[:,1].max():.0f}]")

    print(f"\n  {'Model':<20} {'Unscaled acc':<16} {'Scaled acc'}")
    X_tr_s = StandardScaler().fit_transform(X_tr)
    X_te_s = StandardScaler().fit(X_tr).transform(X_te)

    for name, clf in [("KNN(k=5)",  KNeighborsClassifier()),
                      ("SVM(rbf)",  SVC()),
                      ("Logistic",  LogisticRegression(max_iter=500))]:
        acc_us = cross_val_score(clf, np.vstack([X_tr, X_te]),
                                 np.concatenate([y_tr, y_te]), cv=5).mean()
        X_s_all = StandardScaler().fit_transform(np.vstack([X_tr, X_te]))
        acc_s  = cross_val_score(clf, X_s_all, np.concatenate([y_tr, y_te]), cv=5).mean()
        print(f"  {name:<20} {acc_us:<16.4f} {acc_s:.4f}")


# -- 2. Standard Scaler (Z-score normalisation) -------------------------------
def standard_scaler_demo():
    print("\n=== StandardScaler (Z-score) ===")
    print("  X' = (X - mu) / sigma   ->   mean=0, std=1")
    rng = np.random.default_rng(1)
    X   = rng.normal(50, 20, (100, 3))

    scaler = StandardScaler().fit(X)
    X_s    = scaler.transform(X)
    print(f"  Before: mean={X.mean(0).round(2)}  std={X.std(0).round(2)}")
    print(f"  After:  mean={X_s.mean(0).round(6)}  std={X_s.std(0).round(6)}")
    print(f"  Learned mean: {scaler.mean_.round(2)}")
    print(f"  Learned std:  {scaler.scale_.round(2)}")


# -- 3. MinMax Scaler ---------------------------------------------------------
def minmax_scaler_demo():
    print("\n=== MinMaxScaler ===")
    print("  X' = (X - X_min) / (X_max - X_min)  ->  range [0,1]")
    rng = np.random.default_rng(2)
    X   = rng.normal(100, 30, (50, 2))
    # Inject outlier
    X[0, 0] = 500

    scaler = MinMaxScaler().fit(X)
    X_s    = scaler.transform(X)
    print(f"  Original range: [{X[:,0].min():.1f}, {X[:,0].max():.1f}]")
    print(f"  Scaled range:   [{X_s[:,0].min():.4f}, {X_s[:,0].max():.4f}]")
    print(f"  Issue: outlier (500) compresses all other values!")
    print(f"  Mean of scaled (with outlier): {X_s[:,0].mean():.4f}  (expect ~0.5)")

    # Custom range
    scaler_range = MinMaxScaler(feature_range=(-1, 1)).fit(X)
    X_sr = scaler_range.transform(X)
    print(f"\n  Custom range [-1,1]: [{X_sr[:,0].min():.4f}, {X_sr[:,0].max():.4f}]")


# -- 4. Robust Scaler ---------------------------------------------------------
def robust_scaler_demo():
    print("\n=== RobustScaler (median + IQR) ===")
    print("  X' = (X - median) / IQR  ->  robust to outliers")
    rng = np.random.default_rng(3)
    X   = rng.normal(0, 1, (100, 2))
    X   = np.vstack([X, [[10, 10], [-10, -10]]])   # outliers

    for name, scaler in [("StandardScaler", StandardScaler()),
                          ("RobustScaler",  RobustScaler())]:
        X_s = scaler.fit_transform(X)
        print(f"  {name}: mean={X_s.mean(0).round(4)}  "
              f"std={X_s.std(0).round(4)}  "
              f"max={X_s.max(0).round(2)}")


# -- 5. Normalizer (L1, L2 per sample) ---------------------------------------
def normalizer_demo():
    print("\n=== Normalizer (per-sample norm) ===")
    print("  Scales each sample to unit norm (NOT each feature)")
    print("  Use when: magnitude doesn't matter, only direction (e.g. text, cosine sim)")
    X = np.array([[1, 2, 3], [4, 0, 2], [0, 5, 1]], dtype=float)
    for norm in ["l1", "l2", "max"]:
        X_n = Normalizer(norm=norm).fit_transform(X)
        norms = np.linalg.norm(X_n, ord={"l1":1,"l2":2,"max":np.inf}[norm], axis=1)
        print(f"  norm={norm}: row norms={norms.round(4)}  X[0]={X_n[0].round(4)}")


# -- 6. Power and Quantile transformers ---------------------------------------
def power_quantile_transform():
    print("\n=== Power and Quantile Transforms (for Gaussianising skewed data) ===")
    rng = np.random.default_rng(4)
    # Highly skewed distribution
    X_skewed = rng.exponential(scale=5, size=(500, 1))

    from scipy.stats import skew
    print(f"  Original skewness: {skew(X_skewed[:,0]):.4f}")

    for name, transformer in [
        ("Box-Cox",        PowerTransformer(method="box-cox")),
        ("Yeo-Johnson",    PowerTransformer(method="yeo-johnson")),
        ("Quantile(norm)", QuantileTransformer(output_distribution="normal", random_state=0)),
        ("Quantile(unif)", QuantileTransformer(output_distribution="uniform", random_state=0)),
    ]:
        X_t = transformer.fit_transform(X_skewed)
        print(f"  {name:<22}: skewness={skew(X_t[:,0]):.4f}  "
              f"range=[{X_t.min():.2f}, {X_t.max():.2f}]")


# -- 7. Scaler comparison on ML performance -----------------------------------
def scaler_comparison():
    print("\n=== Scaler Comparison on ML Accuracy ===")
    rng = np.random.default_rng(5)
    n   = 300
    # Mix of very different scales
    X   = np.column_stack([rng.normal(0, 1, n),
                           rng.normal(1000, 500, n),
                           rng.exponential(5, n)])
    y   = (X[:,0] + X[:,1]/500 + np.log1p(X[:,2]) > 4).astype(int)

    scalers = {
        "None":            None,
        "StandardScaler":  StandardScaler(),
        "MinMaxScaler":    MinMaxScaler(),
        "RobustScaler":    RobustScaler(),
        "PowerTransformer":PowerTransformer(method="yeo-johnson"),
        "QuantileTransf":  QuantileTransformer(output_distribution="normal", random_state=0),
    }

    from sklearn.svm import SVC
    print(f"  Model: SVM(rbf)")
    print(f"  {'Scaler':<22} {'5-fold CV acc'}")
    for name, scaler in scalers.items():
        if scaler is None:
            cv = cross_val_score(SVC(), X, y, cv=5).mean()
        else:
            pipe = Pipeline([("scaler", scaler), ("clf", SVC())])
            cv   = cross_val_score(pipe, X, y, cv=5).mean()
        print(f"  {name:<22} {cv:.4f}")


# -- 8. When to scale / not scale ---------------------------------------------
def when_to_scale():
    print("\n=== When to Scale (and When Not To) ===")
    print("  SCALE:")
    print("    - KNN, SVM, Neural Networks (distance/gradient sensitive)")
    print("    - PCA, k-Means (variance/distance based)")
    print("    - Logistic/Linear Regression (regularisation interacts with scale)")
    print("    - Lasso/Ridge (penalty affected by scale)")
    print()
    print("  DO NOT SCALE (or optional):")
    print("    - Decision Trees, Random Forests, Gradient Boosting (threshold-based)")
    print("    - Naive Bayes (density-based, not distance-based)")
    print()
    print("  CRITICAL: fit scaler on TRAIN set only, then transform train and test!")
    print("    -> Prevents data leakage from test into scaler statistics")


if __name__ == "__main__":
    why_scaling()
    standard_scaler_demo()
    minmax_scaler_demo()
    robust_scaler_demo()
    normalizer_demo()
    power_quantile_transform()
    scaler_comparison()
    when_to_scale()
