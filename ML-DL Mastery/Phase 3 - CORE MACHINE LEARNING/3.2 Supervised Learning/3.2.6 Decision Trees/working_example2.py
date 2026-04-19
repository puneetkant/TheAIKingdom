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
    import matplotlib
    matplotlib.use("Agg")
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


def demo_max_depth_sweep():
    """DecisionTreeClassifier on iris, sweep max_depth 1..10, plot train/test curves."""
    print("\n=== Max Depth Sweep (Iris) ===")
    from sklearn.datasets import load_iris
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)
    depths = list(range(1, 11))
    train_accs, test_accs = [], []
    for d in depths:
        dt = DecisionTreeClassifier(max_depth=d, random_state=42)
        dt.fit(X_train, y_train)
        train_accs.append(dt.score(X_train, y_train))
        test_accs.append(dt.score(X_test, y_test))
        print(f"  depth={d:>2}: train={train_accs[-1]:.4f}  test={test_accs[-1]:.4f}")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(depths, train_accs, "o-", label="Train")
    ax.plot(depths, test_accs,  "s-", label="Test")
    ax.set_xlabel("max_depth")
    ax.set_ylabel("Accuracy")
    ax.set_title("Decision Tree Depth vs Accuracy (Iris)")
    ax.legend()
    fig.savefig(OUTPUT / "decision_tree_depth.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: decision_tree_depth.png")


def demo_regression_tree():
    """DecisionTreeRegressor on Cal Housing (first 3000 samples), RMSE for depths [2,4,6,8]."""
    print("\n=== Regression Tree (Cal Housing 3000 samples) ===")
    h = fetch_california_housing()
    X, y = h.data[:3000], h.target[:3000]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    for d in [2, 4, 6, 8]:
        dt = DecisionTreeRegressor(max_depth=d, random_state=42)
        dt.fit(X_train, y_train)
        rmse = mean_squared_error(y_test, dt.predict(X_test)) ** 0.5
        print(f"  depth={d}: RMSE={rmse:.4f}")


def demo_feature_importance():
    """Random Forest (50 trees) on iris, bar chart of feature importances."""
    print("\n=== Feature Importance - Random Forest on Iris ===")
    from sklearn.datasets import load_iris
    from sklearn.ensemble import RandomForestClassifier
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.3, random_state=42)
    rf = RandomForestClassifier(n_estimators=50, random_state=42)
    rf.fit(X_train, y_train)
    importances = rf.feature_importances_
    names = list(iris.feature_names)
    idx = np.argsort(importances)[::-1]
    for rank, i in enumerate(idx):
        print(f"  {rank+1}. {names[i]:<25}: {importances[i]:.4f}")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(len(names)), importances[idx])
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([names[i] for i in idx], rotation=20, ha="right")
    ax.set_ylabel("Importance")
    ax.set_title("Random Forest Feature Importances (Iris)")
    fig.tight_layout()
    fig.savefig(OUTPUT / "dt_feature_importance.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: dt_feature_importance.png")


def demo_ccp_pruning():
    """Cost-complexity pruning (ccp_alpha sweep) on iris classification, accuracy vs alpha."""
    print("\n=== CCP Pruning Sweep (Iris) ===")
    from sklearn.datasets import load_iris
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)
    dt0 = DecisionTreeClassifier(random_state=42)
    path = dt0.cost_complexity_pruning_path(X_train, y_train)
    alphas = path.ccp_alphas[:-1]   # drop trivial last entry (single-node tree)
    print(f"  Alpha candidates: {len(alphas)}")
    train_accs, test_accs = [], []
    for a in alphas:
        dt = DecisionTreeClassifier(ccp_alpha=a, random_state=42)
        dt.fit(X_train, y_train)
        train_accs.append(dt.score(X_train, y_train))
        test_accs.append(dt.score(X_test, y_test))
    step = max(1, len(alphas) // 8)
    print("  {:<12}  {:>7}  {:>7}".format("alpha", "train", "test"))
    for i in range(0, len(alphas), step):
        print(f"  {alphas[i]:<12.6f}  {train_accs[i]:>7.4f}  {test_accs[i]:>7.4f}")
    best_i = int(np.argmax(test_accs))
    print(f"  Best: alpha={alphas[best_i]:.6f}  test_acc={test_accs[best_i]:.4f}")


if __name__ == "__main__":
    demo_tree_classifier()
    demo_tree_regressor()
    demo_max_depth_sweep()
    demo_regression_tree()
    demo_feature_importance()
    demo_ccp_pruning()
