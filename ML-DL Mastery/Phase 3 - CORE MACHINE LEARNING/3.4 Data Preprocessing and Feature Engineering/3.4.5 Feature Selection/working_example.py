"""
Working Example: Feature Selection
Covers filter methods (correlation, chi-squared, ANOVA), wrapper methods
(RFE, sequential), embedded methods (L1, tree importance), and mutual information.
"""
import numpy as np
from sklearn.datasets import make_classification, load_breast_cancer
from sklearn.feature_selection import (SelectKBest, f_classif, chi2,
                                        mutual_info_classif, RFE, RFECV,
                                        SelectFromModel, VarianceThreshold,
                                        SequentialFeatureSelector)
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
import os


# ── 1. Why feature selection ──────────────────────────────────────────────────
def why_feature_selection():
    print("=== Why Feature Selection ===")
    print("  Benefits: reduces overfitting, speeds up training, improves interpretability")
    print("  Three main approaches:")
    print("    Filter:   score features independently (fast, model-agnostic)")
    print("    Wrapper:  use model performance to select subset (slow, model-specific)")
    print("    Embedded: regularisation / tree importance during training")


# ── 2. Variance threshold (remove near-constant) ─────────────────────────────
def variance_threshold():
    print("\n=== Variance Threshold ===")
    rng = np.random.default_rng(0)
    X   = rng.standard_normal((100, 5))
    X   = np.column_stack([X,
                           np.zeros(100),              # constant
                           np.full(100, 0.001) + rng.normal(0, 0.0001, 100)])  # near-constant

    vt = VarianceThreshold(threshold=0.01)
    X_sel = vt.fit_transform(X)
    print(f"  Original features: {X.shape[1]}")
    print(f"  After threshold=0.01: {X_sel.shape[1]}")
    print(f"  Variances: {X.var(0).round(5)}")
    print(f"  Selected:  {vt.get_support()}")


# ── 3. Correlation filter ─────────────────────────────────────────────────────
def correlation_filter():
    print("\n=== Correlation-Based Filter ===")
    rng = np.random.default_rng(1)
    n   = 300
    X   = rng.standard_normal((n, 10))
    # Add correlated feature
    X   = np.column_stack([X, X[:,0] + 0.01*rng.standard_normal(n)])  # almost identical to col 0
    y   = (X[:,0] + X[:,2] + X[:,5] > 0).astype(int)

    # Feature-target correlation
    corr_with_y = np.abs(np.array([np.corrcoef(X[:,j], y)[0,1] for j in range(X.shape[1])]))
    idx_sorted  = corr_with_y.argsort()[::-1]
    print(f"  Feature-target |correlation|:")
    for i in idx_sorted[:8]:
        print(f"    f_{i}: {corr_with_y[i]:.4f}")

    # Remove highly inter-correlated features (correlation > 0.9)
    C        = np.corrcoef(X.T)
    to_drop  = set()
    for i in range(X.shape[1]):
        for j in range(i+1, X.shape[1]):
            if abs(C[i,j]) > 0.90:
                to_drop.add(j)   # drop the later one
    print(f"\n  Dropping {len(to_drop)} highly correlated features: {sorted(to_drop)}")


# ── 4. Filter: SelectKBest (ANOVA F-test, Chi-squared, MI) ──────────────────
def select_k_best():
    print("\n=== SelectKBest Filter ===")
    X, y = make_classification(n_samples=400, n_features=20, n_informative=5,
                                n_redundant=3, random_state=0)
    scaler = StandardScaler()
    X_s    = scaler.fit_transform(X)

    methods = {
        "ANOVA F-test":    f_classif,
        "Mutual Info":     mutual_info_classif,
    }
    for name, scorer in methods.items():
        sel = SelectKBest(scorer, k=5)
        sel.fit(X_s, y)
        scores  = sel.scores_
        top5    = scores.argsort()[-5:][::-1]
        cv      = cross_val_score(
            LogisticRegression(max_iter=500), sel.transform(X_s), y, cv=5).mean()
        print(f"\n  {name}:")
        print(f"    Top 5 features: {top5}  scores={scores[top5].round(3)}")
        print(f"    CV acc (k=5): {cv:.4f}")


# ── 5. Recursive Feature Elimination (RFE) ────────────────────────────────────
def recursive_feature_elimination():
    print("\n=== Recursive Feature Elimination (RFE) ===")
    X, y = make_classification(n_samples=300, n_features=15, n_informative=5, random_state=1)
    scaler = StandardScaler()
    X_s    = scaler.fit_transform(X)

    estimator = LogisticRegression(max_iter=500)
    rfe = RFE(estimator=estimator, n_features_to_select=5, step=1)
    rfe.fit(X_s, y)

    print(f"  Selected features: {np.where(rfe.support_)[0]}")
    print(f"  Feature ranking:   {rfe.ranking_}")

    # RFECV: automatically find optimal n
    rfecv = RFECV(estimator=estimator, step=1, cv=5, scoring="accuracy", n_jobs=-1)
    rfecv.fit(X_s, y)
    print(f"\n  RFECV optimal n_features: {rfecv.n_features_}")
    print(f"  RFECV selected: {np.where(rfecv.support_)[0]}")
    print(f"  CV acc with RFECV: {rfecv.cv_results_['mean_test_score'][rfecv.n_features_-1]:.4f}")


# ── 6. Embedded: L1 (Lasso) selection ────────────────────────────────────────
def lasso_selection():
    print("\n=== L1 (Lasso) Feature Selection ===")
    X, y = make_classification(n_samples=300, n_features=20, n_informative=5, random_state=2)
    scaler = StandardScaler()
    X_s    = scaler.fit_transform(X)

    print(f"  {'C (inv. reg.)':<16} {'Non-zero features':<20} {'CV acc'}")
    for C in [0.001, 0.01, 0.1, 1.0]:
        lasso = LogisticRegression(penalty="l1", C=C, solver="liblinear", max_iter=1000)
        sfm   = SelectFromModel(lasso).fit(X_s, y)
        n_sel = sfm.get_support().sum()
        X_sel = sfm.transform(X_s)
        cv    = cross_val_score(LogisticRegression(max_iter=500), X_sel, y, cv=5).mean()
        print(f"  {C:<16} {n_sel:<20} {cv:.4f}")


# ── 7. Embedded: Tree feature importance ─────────────────────────────────────
def tree_importance():
    print("\n=== Tree-Based Feature Importance ===")
    X, y = make_classification(n_samples=400, n_features=15, n_informative=5, random_state=3)

    rf = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)
    rf.fit(X, y)
    importances = rf.feature_importances_
    idx = importances.argsort()[::-1]

    print(f"  {'Feature':<12} {'Importance'}")
    for i in idx[:10]:
        print(f"  f_{i:<9} {importances[i]:.4f}")

    # Permutation importance (more reliable)
    from sklearn.inspection import permutation_importance
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)
    rf.fit(X_tr, y_tr)
    perm = permutation_importance(rf, X_te, y_te, n_repeats=10, random_state=0, n_jobs=-1)
    print(f"\n  Permutation importance (top 5):")
    perm_idx = perm.importances_mean.argsort()[::-1]
    for i in perm_idx[:5]:
        print(f"  f_{i:<9} {perm.importances_mean[i]:.4f} ± {perm.importances_std[i]:.4f}")


# ── 8. Feature selection comparison ──────────────────────────────────────────
def feature_selection_comparison():
    print("\n=== Feature Selection Method Comparison ===")
    cancer = load_breast_cancer()
    X, y   = cancer.data, cancer.target
    scaler = StandardScaler()
    X_s    = scaler.fit_transform(X)

    base_cv = cross_val_score(LogisticRegression(max_iter=1000), X_s, y, cv=5).mean()
    print(f"  All {X.shape[1]} features: CV acc = {base_cv:.4f}")

    k = 10  # select top 10
    methods = [
        ("ANOVA F-test (k=10)",      SelectKBest(f_classif, k=k)),
        ("Mutual Info (k=10)",       SelectKBest(mutual_info_classif, k=k)),
        ("L1 SelectFromModel",       SelectFromModel(LogisticRegression(penalty="l1", C=0.1, solver="liblinear", max_iter=1000))),
        ("RF SelectFromModel",       SelectFromModel(RandomForestClassifier(n_estimators=50, random_state=0, n_jobs=-1))),
        ("RFE (k=10)",               RFE(LogisticRegression(max_iter=1000), n_features_to_select=k, step=2)),
    ]

    for name, selector in methods:
        X_sel = selector.fit_transform(X_s, y)
        cv    = cross_val_score(LogisticRegression(max_iter=1000), X_sel, y, cv=5).mean()
        print(f"  {name:<35}: n_features={X_sel.shape[1]}  CV acc={cv:.4f}")


if __name__ == "__main__":
    why_feature_selection()
    variance_threshold()
    correlation_filter()
    select_k_best()
    recursive_feature_elimination()
    lasso_selection()
    tree_importance()
    feature_selection_comparison()
