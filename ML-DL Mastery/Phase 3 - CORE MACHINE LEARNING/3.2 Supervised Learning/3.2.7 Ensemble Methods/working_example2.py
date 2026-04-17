"""
Working Example 2: Ensemble Methods — Random Forest, Gradient Boosting, XGBoost
================================================================================
Random Forest regression, GradientBoostingRegressor, feature importances,
OOB score, learning rate sweep.

Run:  python working_example2.py
"""
from pathlib import Path
try:
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                                   RandomForestClassifier, AdaBoostClassifier)
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import mean_squared_error, roc_auc_score
except ImportError:
    raise SystemExit("pip install numpy matplotlib scikit-learn")

OUTPUT = Path(__file__).parent / "output"
OUTPUT.mkdir(exist_ok=True)

def demo_random_forest():
    print("=== Random Forest Regressor ===")
    h = fetch_california_housing()
    X, y = h.data, h.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    for n in [10, 50, 100, 200]:
        rf = RandomForestRegressor(n_estimators=n, oob_score=True, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        rmse = mean_squared_error(y_test, rf.predict(X_test))**0.5
        oob_rmse = ((y_train - rf.oob_prediction_)**2).mean()**0.5
        print(f"  n={n:>3}: test RMSE={rmse:.4f}  OOB RMSE={oob_rmse:.4f}")

    # Feature importances
    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    importances = rf.feature_importances_
    names = h.feature_names
    print("\n  Feature importances:")
    for name, imp in sorted(zip(names, importances), key=lambda x: x[1], reverse=True):
        print(f"    {name:20s}: {imp:.4f}")

    fig, ax = plt.subplots(figsize=(7, 4))
    idx = np.argsort(importances)[::-1]
    ax.bar(range(len(importances)), importances[idx])
    ax.set_xticks(range(len(importances))); ax.set_xticklabels([names[i] for i in idx], rotation=45, ha="right")
    ax.set_title("Random Forest Feature Importances")
    fig.savefig(OUTPUT / "rf_importances.png", dpi=120, bbox_inches="tight")
    plt.close(fig); print("  Saved: rf_importances.png")

def demo_gradient_boosting():
    print("\n=== Gradient Boosting Regressor ===")
    h = fetch_california_housing()
    X, y = h.data, h.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    for lr in [0.01, 0.05, 0.1, 0.3]:
        gb = GradientBoostingRegressor(n_estimators=200, learning_rate=lr, max_depth=3,
                                        subsample=0.8, random_state=42)
        gb.fit(X_train, y_train)
        rmse = mean_squared_error(y_test, gb.predict(X_test))**0.5
        print(f"  lr={lr}: RMSE={rmse:.4f}")

def demo_adaboost():
    print("\n=== AdaBoost Classifier ===")
    h = fetch_california_housing()
    X, y = h.data, (h.target > np.median(h.target)).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    ab = AdaBoostClassifier(n_estimators=100, random_state=42)
    ab.fit(X_train, y_train)
    auc = roc_auc_score(y_test, ab.predict_proba(X_test)[:,1])
    print(f"  AdaBoost (100 trees): AUC={auc:.4f}")

if __name__ == "__main__":
    demo_random_forest()
    demo_gradient_boosting()
    demo_adaboost()
