"""
Working Example: Hyperparameter Tuning
Covers manual search, grid search, random search, Bayesian optimisation,
Halving searches, and practical tuning strategies.
"""
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
                                      HalvingGridSearchCV, HalvingRandomSearchCV,
                                      StratifiedKFold, cross_val_score)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import loguniform, randint, uniform
import os, time


# ── 1. Why hyperparameter tuning ──────────────────────────────────────────────
def intro():
    print("=== Hyperparameter Tuning ===")
    print("  Hyperparameters: parameters set BEFORE training (not learned from data)")
    print("  Examples: learning rate, n_estimators, max_depth, C, kernel, ...")
    print()
    print("  Tuning objectives:")
    print("    1. Maximise generalisation (CV performance)")
    print("    2. Avoid overfitting to validation data")
    print("    3. Balance accuracy vs compute cost")


# ── 2. Manual / grid search intuition ────────────────────────────────────────
def manual_search():
    print("\n=== Manual Parameter Sweep ===")
    X, y = make_classification(n_samples=400, n_features=10, n_informative=5, random_state=0)
    cv   = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    print(f"  LR: C sweep")
    best_c, best_acc = None, 0
    for C in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
        pipe  = Pipeline([("sc", StandardScaler()),
                          ("lr", LogisticRegression(C=C, max_iter=500))])
        acc   = cross_val_score(pipe, X, y, cv=cv).mean()
        marker = " ← best" if acc > best_acc else ""
        print(f"    C={C:<8} acc={acc:.4f}{marker}")
        if acc > best_acc:
            best_acc, best_c = acc, C
    print(f"  Best: C={best_c}  CV acc={best_acc:.4f}")


# ── 3. Grid Search ────────────────────────────────────────────────────────────
def grid_search():
    print("\n=== Grid Search (GridSearchCV) ===")
    X, y  = make_classification(n_samples=400, n_features=10, n_informative=5, random_state=1)
    pipe  = Pipeline([("sc", StandardScaler()), ("svc", SVC(probability=True))])
    param_grid = {
        "svc__C":      [0.1, 1.0, 10.0],
        "svc__kernel": ["linear", "rbf"],
        "svc__gamma":  ["scale", "auto"],
    }
    total = 3 * 2 * 2
    print(f"  Grid size: {total} combinations × 5 folds = {total*5} fits")

    t0 = time.time()
    gs = GridSearchCV(pipe, param_grid, cv=5, scoring="accuracy",
                      n_jobs=-1, verbose=0)
    gs.fit(X, y)
    elapsed = time.time() - t0
    print(f"  Elapsed: {elapsed:.2f}s")
    print(f"  Best params:  {gs.best_params_}")
    print(f"  Best CV acc:  {gs.best_score_:.4f}")

    # Top 5 results
    res = gs.cv_results_
    idx = np.argsort(res["mean_test_score"])[::-1]
    print(f"\n  Top 5 results:")
    for i in idx[:5]:
        print(f"    {res['params'][i]}  → acc={res['mean_test_score'][i]:.4f}")


# ── 4. Random Search ─────────────────────────────────────────────────────────
def random_search():
    print("\n=== Random Search (RandomizedSearchCV) ===")
    print("  Samples randomly from distributions — efficient for large spaces")
    X, y  = make_classification(n_samples=500, n_features=15, n_informative=7, random_state=2)

    param_dist = {
        "n_estimators":      randint(50, 500),
        "max_depth":         randint(3, 20),
        "min_samples_split": randint(2, 20),
        "min_samples_leaf":  randint(1, 10),
        "max_features":      uniform(0.1, 0.9),
    }
    n_iter = 30
    print(f"  RandomForest: {n_iter} random samples × 5 folds = {n_iter*5} fits")

    t0 = time.time()
    rs = RandomizedSearchCV(
        RandomForestClassifier(random_state=0, n_jobs=-1),
        param_dist, n_iter=n_iter, cv=5, scoring="accuracy",
        random_state=0, n_jobs=-1
    )
    rs.fit(X, y)
    elapsed = time.time() - t0
    print(f"  Elapsed: {elapsed:.2f}s")
    print(f"  Best params:  {rs.best_params_}")
    print(f"  Best CV acc:  {rs.best_score_:.4f}")

    # Compare with default RF
    cv_default = cross_val_score(
        RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1), X, y, cv=5).mean()
    print(f"  Default RF CV acc: {cv_default:.4f}")
    print(f"  Improvement: {(rs.best_score_ - cv_default)*100:+.2f}%")


# ── 5. Halving Search (successive halving) ───────────────────────────────────
def halving_search():
    print("\n=== Halving Search (Successive Halving) ===")
    print("  Allocates more compute to promising configurations iteratively")
    X, y  = make_classification(n_samples=600, n_features=12, n_informative=6, random_state=3)

    param_grid = {
        "n_estimators": [50, 100, 200, 300],
        "max_depth":    [3, 5, 7, None],
        "max_features": [0.3, 0.5, 0.7, 1.0],
    }
    t0 = time.time()
    hgs = HalvingGridSearchCV(
        RandomForestClassifier(random_state=0, n_jobs=-1),
        param_grid, cv=5, scoring="accuracy", factor=3, n_jobs=-1
    )
    hgs.fit(X, y)
    elapsed = time.time() - t0
    print(f"  HalvingGridSearch: {elapsed:.2f}s")
    print(f"  Best params:  {hgs.best_params_}")
    print(f"  Best CV acc:  {hgs.best_score_:.4f}")


# ── 6. Bayesian Optimisation (conceptual + optuna if available) ───────────────
def bayesian_optimisation():
    print("\n=== Bayesian Optimisation ===")
    print("  Models the objective function as a Gaussian Process surrogate")
    print("  Uses acquisition function (EI, UCB) to choose next trial")
    print("  More efficient than random search for expensive evaluations")
    print()
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        X, y = make_classification(n_samples=400, n_features=10, n_informative=5, random_state=4)
        cv   = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

        def objective(trial):
            n_est  = trial.suggest_int("n_estimators", 50, 300)
            depth  = trial.suggest_int("max_depth", 2, 15)
            mf     = trial.suggest_float("max_features", 0.1, 1.0)
            rf     = RandomForestClassifier(n_estimators=n_est, max_depth=depth,
                                            max_features=mf, random_state=0, n_jobs=-1)
            return cross_val_score(rf, X, y, cv=cv).mean()

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=20, show_progress_bar=False)

        print(f"  Optuna (20 trials):")
        print(f"  Best params: {study.best_params}")
        print(f"  Best value:  {study.best_value:.4f}")

    except ImportError:
        print("  optuna not installed. Install: pip install optuna")
        print("  Algorithm:")
        print("    1. Fit GP on (params → score) observations")
        print("    2. Use acquisition function to pick next params")
        print("    3. Evaluate; update GP; repeat")
        print("    4. Return params with highest predicted score")


# ── 7. Early stopping in ensembles ───────────────────────────────────────────
def early_stopping_gbm():
    print("\n=== Early Stopping (Gradient Boosting) ===")
    X, y = make_classification(n_samples=500, n_features=10, n_informative=5, random_state=5)
    from sklearn.model_selection import train_test_split
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=0)

    gb = GradientBoostingClassifier(n_estimators=500, learning_rate=0.05,
                                     max_depth=3, random_state=0)
    gb.fit(X_tr, y_tr)

    val_scores = [gb.estimators_[:i+1][-1].predict(X_val).mean()
                  if i > 0 else 0
                  for i in range(0, len(gb.estimators_), 10)]

    staged = list(gb.staged_score(X_val, y_val))
    best_n = np.argmax(staged) + 1
    print(f"  Max 500 estimators")
    print(f"  Best validation accuracy at n_estimators={best_n}: {staged[best_n-1]:.4f}")
    print(f"  Final (500):  {staged[-1]:.4f}")


# ── 8. Tuning tips ───────────────────────────────────────────────────────────
def tuning_tips():
    print("\n=== Practical Tuning Tips ===")
    tips = [
        ("Start coarse",       "Use log scale for C, LR: [0.001, 0.01, 0.1, 1, 10, 100]"),
        ("Random > Grid",      "Random search for many hyperparams; grid for 1-2"),
        ("Order of importance","LR/n_estimators >> max_depth >> regularisation"),
        ("Warm start",         "Reuse fit for incremental n_estimators checks"),
        ("Pipeline tuning",    "Tune preprocessing + model together via Pipeline"),
        ("No test set",        "Never tune on test set; use held-out validation or CV"),
        ("Early stopping",     "For iterative algorithms; prevents overfitting to train"),
        ("Optuna/Hyperopt",    "Use BO for expensive models (deep nets, XGBoost)"),
    ]
    for tip, desc in tips:
        print(f"  {tip:<22}: {desc}")


if __name__ == "__main__":
    intro()
    manual_search()
    grid_search()
    random_search()
    halving_search()
    bayesian_optimisation()
    early_stopping_gbm()
    tuning_tips()
