"""
Working Example: Ensemble Methods
Covers bagging, random forests, extra trees, boosting (AdaBoost, Gradient Boosting,
XGBoost), stacking, and voting classifiers.
"""
import numpy as np
from sklearn.datasets import make_classification, load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import (BaggingClassifier, RandomForestClassifier, ExtraTreesClassifier,
                               AdaBoostClassifier, GradientBoostingClassifier,
                               VotingClassifier, StackingClassifier,
                               RandomForestRegressor, GradientBoostingRegressor)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import os


# ── 1. Wisdom of crowds: why ensembles work ───────────────────────────────────
def ensemble_theory():
    print("=== Why Ensembles Work ===")
    print("  Error of ensemble (averaging) with M independent classifiers:")
    print("  Each has error ε, if ε < 0.5 and models are independent:")
    print("  P(majority wrong) = Σ_{k>M/2} C(M,k) ε^k (1-ε)^{M-k}")
    print()
    from scipy.stats import binom
    for eps in [0.4, 0.3, 0.2]:
        for M in [3, 5, 11, 21]:
            # P(majority vote wrong) = P(binomial(M, eps) > M/2)
            p_err = 1 - binom.cdf(M//2, M, eps)
            print(f"  ε={eps}  M={M:2d}: P(ensemble wrong)={p_err:.6f}  (single={eps})")
    print()
    print("  Key: models should be diverse (low correlation in errors).")


# ── 2. Bagging ────────────────────────────────────────────────────────────────
def bagging_demo():
    print("\n=== Bagging (Bootstrap Aggregating) ===")
    X, y = make_classification(n_samples=500, n_features=10, random_state=0)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)

    single = DecisionTreeClassifier(random_state=0)
    bagging = BaggingClassifier(
        estimator=DecisionTreeClassifier(),
        n_estimators=100,
        max_samples=0.8,
        bootstrap=True,
        random_state=0,
        n_jobs=-1,
    )

    single.fit(X_tr, y_tr)
    bagging.fit(X_tr, y_tr)

    print(f"  Single tree: train={single.score(X_tr,y_tr):.4f}  test={single.score(X_te,y_te):.4f}")
    print(f"  Bagging(100): train={bagging.score(X_tr,y_tr):.4f}  test={bagging.score(X_te,y_te):.4f}")


# ── 3. Random Forest ─────────────────────────────────────────────────────────
def random_forest():
    print("\n=== Random Forest ===")
    print("  Extension of bagging: at each split, consider random subset of features")
    print("  max_features: controls feature randomness (reduces correlation between trees)")
    X, y = make_classification(n_samples=600, n_features=15, n_informative=6, random_state=1)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)

    print(f"\n  {'n_estimators':<15} {'max_features':<14} {'Test acc'}")
    for n_est in [10, 50, 100, 200]:
        for mf in ["sqrt", "log2", None]:
            rf = RandomForestClassifier(n_estimators=n_est, max_features=mf,
                                        random_state=0, n_jobs=-1)
            rf.fit(X_tr, y_tr)
            print(f"  {n_est:<15} {str(mf):<14} {rf.score(X_te, y_te):.4f}")
    print()

    # OOB score (out-of-bag)
    rf_oob = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=0, n_jobs=-1)
    rf_oob.fit(X_tr, y_tr)
    print(f"  OOB score: {rf_oob.oob_score_:.4f}  test acc: {rf_oob.score(X_te, y_te):.4f}")

    # Feature importances
    importances = rf_oob.feature_importances_
    top_idx = importances.argsort()[-5:][::-1]
    print(f"  Top 5 features: {[(f'f{i}', round(importances[i],4)) for i in top_idx]}")


# ── 4. Extra Trees ────────────────────────────────────────────────────────────
def extra_trees():
    print("\n=== Extremely Randomised Trees (Extra Trees) ===")
    print("  Splits chosen at random (not optimally) → even lower variance")
    X, y = make_classification(n_samples=500, n_features=10, random_state=2)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)

    for name, model in [
        ("RandomForest",  RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)),
        ("ExtraTrees",    ExtraTreesClassifier(n_estimators=100,   random_state=0, n_jobs=-1)),
    ]:
        model.fit(X_tr, y_tr)
        print(f"  {name:<16}: test acc={model.score(X_te, y_te):.4f}")


# ── 5. AdaBoost ───────────────────────────────────────────────────────────────
def adaboost_demo():
    print("\n=== AdaBoost ===")
    print("  Sequentially fit weak learners, upweight misclassified samples")
    print("  Final prediction: weighted majority vote")
    X, y = make_classification(n_samples=500, n_features=10, random_state=3)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)

    print(f"\n  {'n_estimators':<15} {'learning_rate':<16} {'Test acc'}")
    for n_est in [50, 100, 200]:
        for lr in [0.5, 1.0, 2.0]:
            ada = AdaBoostClassifier(
                estimator=DecisionTreeClassifier(max_depth=1),
                n_estimators=n_est, learning_rate=lr, random_state=0)
            ada.fit(X_tr, y_tr)
            print(f"  {n_est:<15} {lr:<16} {ada.score(X_te, y_te):.4f}")


# ── 6. Gradient Boosting ──────────────────────────────────────────────────────
def gradient_boosting_demo():
    print("\n=== Gradient Boosting (GBDT) ===")
    print("  Fit each tree to residuals of previous ensemble (gradient of loss)")
    X, y = make_classification(n_samples=600, n_features=10, random_state=4)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)

    # Compare training evolution
    gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1,
                                     max_depth=3, random_state=0)
    gb.fit(X_tr, y_tr)
    staged_test = [gb.score(X_te, y_te) for gb in gb.staged_predict(X_te)]  # not ideal
    # Proper staged score
    staged_acc = [np.mean(sp == y_te) for sp in gb.staged_predict(X_te)]

    print(f"  n=1:   acc={staged_acc[0]:.4f}")
    print(f"  n=10:  acc={staged_acc[9]:.4f}")
    print(f"  n=50:  acc={staged_acc[49]:.4f}")
    print(f"  n=100: acc={staged_acc[99]:.4f}")
    print(f"  n=200: acc={staged_acc[199]:.4f}")

    print(f"\n  {'learning_rate':<15} {'max_depth':<12} {'Test acc'}")
    for lr in [0.01, 0.05, 0.1, 0.3]:
        for md in [2, 3, 5]:
            model = GradientBoostingClassifier(n_estimators=100, learning_rate=lr,
                                               max_depth=md, random_state=0)
            model.fit(X_tr, y_tr)
            print(f"  {lr:<15} {md:<12} {model.score(X_te, y_te):.4f}")


# ── 7. Voting and Stacking ───────────────────────────────────────────────────
def voting_and_stacking():
    print("\n=== Voting and Stacking ===")
    X, y = make_classification(n_samples=500, n_features=10, random_state=5)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)

    base = [
        ("lr",  LogisticRegression(max_iter=500)),
        ("rf",  RandomForestClassifier(n_estimators=50, random_state=0)),
        ("svc", SVC(probability=True, kernel="rbf")),
    ]

    for voting in ["hard", "soft"]:
        vc = VotingClassifier(estimators=base, voting=voting, n_jobs=-1)
        vc.fit(X_tr, y_tr)
        print(f"  VotingClassifier ({voting}): {vc.score(X_te, y_te):.4f}")

    # Stacking
    stack = StackingClassifier(
        estimators=base,
        final_estimator=LogisticRegression(max_iter=500),
        cv=5,
        n_jobs=-1,
    )
    stack.fit(X_tr, y_tr)
    print(f"  StackingClassifier:         {stack.score(X_te, y_te):.4f}")


if __name__ == "__main__":
    ensemble_theory()
    bagging_demo()
    random_forest()
    extra_trees()
    adaboost_demo()
    gradient_boosting_demo()
    voting_and_stacking()
