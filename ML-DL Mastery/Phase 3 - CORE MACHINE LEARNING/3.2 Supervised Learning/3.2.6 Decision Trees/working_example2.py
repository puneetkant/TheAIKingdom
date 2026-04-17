"""
Working Example 2: Decision Trees — Classification & Regression, Depth Sweep
=============================================================================
DecisionTreeClassifier and Regressor on Cal Housing.
Max depth sweep, feature importances, tree visualisation.

Run:  python working_example2.py
"""
from pathlib import Path
try:
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text
    from sklearn.metrics import mean_squared_error, roc_auc_score
except ImportError:
    raise SystemExit("pip install numpy matplotlib scikit-learn")

OUTPUT = Path(__file__).parent / "output"
OUTPUT.mkdir(exist_ok=True)

def demo_tree_classifier():
    print("=== Decision Tree Classifier (Cal Housing binary) ===")
    h = fetch_california_housing()
    X, y = h.data, (h.target > np.median(h.target)).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    results = []
    for d in [1, 2, 3, 5, 8, 15, None]:
        dt = DecisionTreeClassifier(max_depth=d, random_state=42)
        dt.fit(X_train, y_train)
        acc  = dt.score(X_test, y_test)
        auc  = roc_auc_score(y_test, dt.predict_proba(X_test)[:,1])
        tr_acc = dt.score(X_train, y_train)
        print(f"  depth={str(d):>4}: train_acc={tr_acc:.4f}  test_acc={acc:.4f}  AUC={auc:.4f}")
        results.append((d if d else 20, acc, tr_acc))

    depths = [r[0] for r in results]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(depths, [r[2] for r in results], "o-", label="Train acc")
    ax.plot(depths, [r[1] for r in results], "s-", label="Test acc")
    ax.set_xlabel("Max depth"); ax.set_ylabel("Accuracy")
    ax.set_title("Decision Tree depth vs accuracy"); ax.legend()
    fig.savefig(OUTPUT / "dt_depth_sweep.png", dpi=120, bbox_inches="tight")
    plt.close(fig); print("  Saved: dt_depth_sweep.png")

    # Feature importances
    best = DecisionTreeClassifier(max_depth=5, random_state=42)
    best.fit(X_train, y_train)
    importances = best.feature_importances_
    names = h.feature_names
    print("\n  Feature importances (depth=5):")
    for name, imp in sorted(zip(names, importances), key=lambda x: x[1], reverse=True):
        print(f"    {name:20s}: {imp:.4f}")

def demo_tree_regressor():
    print("\n=== Decision Tree Regressor ===")
    h = fetch_california_housing()
    X, y = h.data, h.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    for d in [2, 4, 6, 10, None]:
        dt = DecisionTreeRegressor(max_depth=d, random_state=42)
        dt.fit(X_train, y_train)
        rmse = mean_squared_error(y_test, dt.predict(X_test))**0.5
        print(f"  depth={str(d):>4}: RMSE={rmse:.4f}")

if __name__ == "__main__":
    demo_tree_classifier()
    demo_tree_regressor()
