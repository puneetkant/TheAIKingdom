"""
Working Example: Decision Trees
Covers information gain, Gini impurity, splitting criteria, tree construction
from scratch, pruning, regression trees, and visualisation.
"""
import numpy as np
from collections import Counter
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text, plot_tree
from sklearn.datasets import load_iris, make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_dt")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── 1. Impurity measures ──────────────────────────────────────────────────────
def impurity_measures():
    print("=== Impurity Measures ===")

    def entropy(p):
        p = np.array(p, dtype=float)
        p /= p.sum()
        return -np.sum(p[p > 0] * np.log2(p[p > 0]))

    def gini(p):
        p = np.array(p, dtype=float) / sum(p)
        return 1 - np.sum(p**2)

    distributions = [(1,0), (1,1), (3,1), (1,3), (1,1,1)]
    print(f"  {'Distribution':<20} {'Entropy':<12} {'Gini'}")
    for d in distributions:
        print(f"  {str(d):<20} {entropy(d):<12.4f} {gini(d):.4f}")

    print()
    # Information gain example
    parent = [10, 10]   # 10 class-0, 10 class-1
    split1 = ([8,2], [2,8])   # good split
    split2 = ([5,5], [5,5])   # useless split

    def info_gain(parent, children):
        n = sum(parent)
        h_parent = entropy(parent)
        h_children = sum((sum(c)/n) * entropy(c) for c in children)
        return h_parent - h_children

    print(f"  Parent {parent}: H={entropy(parent):.4f}")
    print(f"  Split1 {split1}: IG={info_gain(parent, split1):.4f}")
    print(f"  Split2 {split2}: IG={info_gain(parent, split2):.4f}")


# ── 2. Decision tree from scratch ────────────────────────────────────────────
def decision_tree_scratch():
    print("\n=== Decision Tree From Scratch ===")

    def gini(y):
        n = len(y)
        if n == 0: return 0
        counts = Counter(y)
        return 1 - sum((c/n)**2 for c in counts.values())

    def best_split(X, y):
        best_gain, best_feat, best_thresh = -1, None, None
        G_parent = gini(y)
        n = len(y)
        for feat in range(X.shape[1]):
            thresholds = np.unique(X[:, feat])
            for thresh in thresholds:
                left  = y[X[:,feat] <= thresh]
                right = y[X[:,feat] >  thresh]
                if len(left)==0 or len(right)==0: continue
                G_children = (len(left)/n)*gini(left) + (len(right)/n)*gini(right)
                gain = G_parent - G_children
                if gain > best_gain:
                    best_gain, best_feat, best_thresh = gain, feat, thresh
        return best_feat, best_thresh, best_gain

    def build_tree(X, y, depth=0, max_depth=4):
        if len(set(y)) == 1 or depth == max_depth or len(y) < 5:
            return {"leaf": True, "class": Counter(y).most_common(1)[0][0]}
        feat, thresh, gain = best_split(X, y)
        if feat is None:
            return {"leaf": True, "class": Counter(y).most_common(1)[0][0]}
        left  = X[:,feat] <= thresh
        return {
            "feat": feat, "thresh": thresh, "gain": gain,
            "left":  build_tree(X[left],  y[left],  depth+1, max_depth),
            "right": build_tree(X[~left], y[~left], depth+1, max_depth),
        }

    def predict_tree(tree, x):
        if tree.get("leaf"):
            return tree["class"]
        return predict_tree(
            tree["left"] if x[tree["feat"]] <= tree["thresh"] else tree["right"], x)

    iris = load_iris()
    X, y = iris.data, iris.target
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)
    tree = build_tree(X_tr, y_tr, max_depth=4)
    preds = np.array([predict_tree(tree, x) for x in X_te])
    acc = (preds == y_te).mean()
    print(f"  Scratch decision tree accuracy: {acc:.4f}")

    # Compare with sklearn
    sk = DecisionTreeClassifier(max_depth=4, random_state=0).fit(X_tr, y_tr)
    print(f"  sklearn DecisionTree accuracy:  {sk.score(X_te, y_te):.4f}")


# ── 3. sklearn Decision Tree ─────────────────────────────────────────────────
def sklearn_decision_tree():
    print("\n=== sklearn Decision Tree ===")
    iris = load_iris()
    X, y = iris.data, iris.target
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)

    print(f"  {'max_depth':<12} {'Train acc':<12} {'Test acc'}")
    for d in [1, 2, 3, 4, 6, None]:
        model = DecisionTreeClassifier(max_depth=d, random_state=0)
        model.fit(X_tr, y_tr)
        print(f"  {str(d):<12} {model.score(X_tr, y_tr):<12.4f} {model.score(X_te, y_te):.4f}")

    # Best model: print tree structure
    model = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X_tr, y_tr)
    print("\n  Tree structure (depth=3):")
    print(export_text(model, feature_names=iris.feature_names))

    # Visualise
    fig, ax = plt.subplots(figsize=(14, 6))
    plot_tree(model, feature_names=iris.feature_names, class_names=iris.target_names,
              filled=True, ax=ax, fontsize=9)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "decision_tree.png")
    plt.savefig(path, dpi=90); plt.close()
    print(f"\n  Tree visualisation saved: {path}")


# ── 4. Feature importance ────────────────────────────────────────────────────
def feature_importance():
    print("\n=== Feature Importance ===")
    X, y = make_classification(n_samples=500, n_features=10, n_informative=4,
                                n_redundant=2, random_state=1)
    model = DecisionTreeClassifier(max_depth=5, random_state=0)
    model.fit(X, y)
    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1]
    print(f"  {'Feature':<12} {'Importance'}")
    for i in idx:
        print(f"  feature_{i:<4}  {importances[i]:.4f}")


# ── 5. Pruning (max_depth, min_samples_split, ccp_alpha) ─────────────────────
def pruning_demo():
    print("\n=== Decision Tree Pruning ===")
    X, y = make_classification(n_samples=300, n_features=10, random_state=2)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)

    # Cost-complexity pruning path
    path = DecisionTreeClassifier(random_state=0).cost_complexity_pruning_path(X_tr, y_tr)
    ccp_alphas = path.ccp_alphas[::max(1, len(path.ccp_alphas)//8)]

    print(f"  {'ccp_alpha':<14} {'Train acc':<14} {'Test acc'}")
    for alpha in ccp_alphas:
        model = DecisionTreeClassifier(ccp_alpha=alpha, random_state=0)
        model.fit(X_tr, y_tr)
        print(f"  {alpha:<14.6f} {model.score(X_tr,y_tr):<14.4f} {model.score(X_te,y_te):.4f}")


# ── 6. Decision Tree Regression ──────────────────────────────────────────────
def regression_tree():
    print("\n=== Regression Tree ===")
    rng = np.random.default_rng(3)
    x   = rng.uniform(-3, 3, 100)
    y   = np.sin(x) + 0.3*rng.standard_normal(100)
    X   = x.reshape(-1,1)
    x_t = np.linspace(-3, 3, 200).reshape(-1,1)
    y_t = np.sin(x_t.ravel())

    print(f"  {'max_depth':<12} {'RMSE'}")
    for d in [1, 2, 3, 5, 8, None]:
        model = DecisionTreeRegressor(max_depth=d, random_state=0).fit(X, y)
        rmse  = np.sqrt(np.mean((model.predict(x_t) - y_t)**2))
        print(f"  {str(d):<12} {rmse:.4f}")


if __name__ == "__main__":
    impurity_measures()
    decision_tree_scratch()
    sklearn_decision_tree()
    feature_importance()
    pruning_demo()
    regression_tree()
