"""
Working Example 2: Perceptron and Linear Models — SGD, Elastic Net, Perceptron
===============================================================================
Perceptron convergence on linearly separable data, SGD classifier/regressor,
Elastic Net regularisation.

Run:  python working_example2.py
"""
from pathlib import Path
try:
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.datasets import fetch_california_housing, make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import (Perceptron, SGDClassifier, SGDRegressor,
                                       ElasticNet, ElasticNetCV)
    from sklearn.metrics import mean_squared_error, accuracy_score
    from sklearn.pipeline import make_pipeline
except ImportError:
    raise SystemExit("pip install numpy matplotlib scikit-learn")

OUTPUT = Path(__file__).parent / "output"
OUTPUT.mkdir(exist_ok=True)

def demo_perceptron():
    print("=== Perceptron (linearly separable) ===")
    np.random.seed(42)
    X, y = make_classification(n_samples=500, n_features=2, n_redundant=0,
                                 n_informative=2, random_state=42, n_clusters_per_class=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipe = make_pipeline(StandardScaler(), Perceptron(max_iter=1000, eta0=0.1, random_state=42))
    pipe.fit(X_train, y_train)
    acc = pipe.score(X_test, y_test)
    print(f"  Perceptron accuracy: {acc:.4f}")

def demo_sgd():
    print("\n=== SGD Classifier (Cal Housing binary) ===")
    h = fetch_california_housing()
    X, y = h.data, (h.target > np.median(h.target)).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    for loss in ["log_loss", "hinge", "modified_huber"]:
        pipe = make_pipeline(StandardScaler(), SGDClassifier(loss=loss, max_iter=1000, random_state=42))
        pipe.fit(X_train, y_train)
        acc = pipe.score(X_test, y_test)
        print(f"  SGD(loss={loss:15s}): accuracy={acc:.4f}")

def demo_elastic_net():
    print("\n=== Elastic Net Regression ===")
    h = fetch_california_housing()
    X, y = h.data, h.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    for l1r in [0.0, 0.25, 0.5, 0.75, 1.0]:
        pipe = make_pipeline(StandardScaler(), ElasticNet(alpha=0.1, l1_ratio=l1r, max_iter=5000))
        pipe.fit(X_train, y_train)
        rmse = mean_squared_error(y_test, pipe.predict(X_test))**0.5
        print(f"  l1_ratio={l1r}: RMSE={rmse:.4f}")

    # CV to find best l1_ratio
    enet_cv = make_pipeline(StandardScaler(), ElasticNetCV(l1_ratio=[.1,.5,.9,.95,1.],
                                                            cv=5, max_iter=5000))
    enet_cv.fit(X_train, y_train)
    best = enet_cv.named_steps["elasticnetcv"]
    rmse = mean_squared_error(y_test, enet_cv.predict(X_test))**0.5
    print(f"\n  ElasticNetCV best l1_ratio={best.l1_ratio_:.3f}  alpha={best.alpha_:.4f}  RMSE={rmse:.4f}")

def demo_passive_aggressive():
    """Passive-Aggressive classifiers: update only on mistakes."""
    print("\n=== Passive-Aggressive Classifier ===")
    from sklearn.linear_model import PassiveAggressiveClassifier
    h = fetch_california_housing()
    X, y = h.data, (h.target > np.median(h.target)).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)
    for C in [0.01, 0.1, 1.0]:
        pa = PassiveAggressiveClassifier(C=C, max_iter=1000, random_state=42)
        pa.fit(X_tr, y_train)
        acc = pa.score(X_te, y_test)
        print(f"  C={C}: accuracy={acc:.4f}")


def demo_online_learning_curve():
    """Simulate online (incremental) SGD learning on streaming data."""
    print("\n=== Online Learning Curve (SGD partial_fit) ===")
    h = fetch_california_housing()
    X_all, y_all = h.data, (h.target > np.median(h.target)).astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(X_all, y_all, test_size=0.2, random_state=42)
    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr)
    X_te_s  = scaler.transform(X_te)
    clf = SGDClassifier(loss="log_loss", random_state=42)
    batch = 200
    accs = []
    for start in range(0, len(X_tr_s), batch):
        clf.partial_fit(X_tr_s[start:start+batch], y_tr[start:start+batch], classes=[0,1])
        accs.append(clf.score(X_te_s, y_te))
    print(f"  Start acc={accs[0]:.4f}  End acc={accs[-1]:.4f}  Batches={len(accs)}")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(accs)
    ax.set_xlabel("Mini-batch"); ax.set_ylabel("Test accuracy")
    ax.set_title("SGD Online Learning Curve")
    fig.savefig(OUTPUT / "sgd_online.png", dpi=120, bbox_inches="tight")
    plt.close(fig); print("  Saved: sgd_online.png")


def demo_ridge_lasso_paths():
    """Compare Ridge and Lasso regularisation paths (coefficient shrinkage)."""
    print("\n=== Ridge vs Lasso Regularisation Paths ===")
    from sklearn.linear_model import Ridge, Lasso
    h = fetch_california_housing()
    X, y = h.data, h.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s  = scaler.transform(X_test)
    alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    ridge_rmses, lasso_rmses = [], []
    for a in alphas:
        for model, store in [(Ridge(alpha=a), ridge_rmses), (Lasso(alpha=a, max_iter=5000), lasso_rmses)]:
            model.fit(X_tr_s, y_train)
            rmse = mean_squared_error(y_test, model.predict(X_te_s))**0.5
            store.append(rmse)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogx(alphas, ridge_rmses, "o-", label="Ridge")
    ax.semilogx(alphas, lasso_rmses, "s--", label="Lasso")
    ax.set_xlabel("alpha"); ax.set_ylabel("RMSE")
    ax.set_title("Ridge vs Lasso RMSE vs alpha"); ax.legend()
    fig.savefig(OUTPUT / "ridge_lasso_paths.png", dpi=120, bbox_inches="tight")
    plt.close(fig); print("  Saved: ridge_lasso_paths.png")


if __name__ == "__main__":
    demo_perceptron()
    demo_sgd()
    demo_elastic_net()
    demo_passive_aggressive()
    demo_online_learning_curve()
    demo_ridge_lasso_paths()
