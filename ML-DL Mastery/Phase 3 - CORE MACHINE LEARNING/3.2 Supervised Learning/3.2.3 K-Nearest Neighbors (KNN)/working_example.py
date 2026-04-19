"""
Working Example: K-Nearest Neighbors (KNN)
Covers the algorithm, distance metrics, k selection via cross-validation,
weighted KNN, regression, curse of dimensionality, and KD-tree.
"""
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.datasets import make_classification, make_moons, load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_knn")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -- 1. KNN from scratch -------------------------------------------------------
def knn_from_scratch():
    print("=== KNN From Scratch ===")

    class KNN_scratch:
        def __init__(self, k=3, metric="euclidean"):
            self.k, self.metric = k, metric

        def fit(self, X, y):
            self.X_train, self.y_train = X, y
            return self

        def predict(self, X):
            dists = cdist(X, self.X_train, metric=self.metric)
            preds = []
            for row in dists:
                idx  = row.argsort()[:self.k]
                nbrs = self.y_train[idx]
                preds.append(np.bincount(nbrs.astype(int)).argmax())
            return np.array(preds)

    rng  = np.random.default_rng(0)
    X, y = make_classification(n_samples=300, n_features=2, n_redundant=0,
                                n_informative=2, random_state=0)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)

    for k in [1, 3, 5, 11]:
        model = KNN_scratch(k=k).fit(X_tr, y_tr)
        acc   = (model.predict(X_te) == y_te).mean()
        print(f"  k={k:<3}: test acc={acc:.4f}")


# -- 2. Distance metrics -------------------------------------------------------
def distance_metrics():
    print("\n=== Distance Metrics ===")
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 0.0, 3.0])
    print(f"  a={a}  b={b}")

    euclidean  = np.sqrt(np.sum((a-b)**2))
    manhattan  = np.sum(np.abs(a-b))
    chebyshev  = np.max(np.abs(a-b))
    cosine_sim = np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))
    minkowski2 = np.sum(np.abs(a-b)**2)**(1/2)
    minkowski3 = np.sum(np.abs(a-b)**3)**(1/3)

    print(f"  Euclidean (L2): {euclidean:.4f}")
    print(f"  Manhattan (L1): {manhattan:.4f}")
    print(f"  Chebyshev (Linf): {chebyshev:.4f}")
    print(f"  Cosine sim:     {cosine_sim:.4f}")
    print(f"  Minkowski p=2:  {minkowski2:.4f}")
    print(f"  Minkowski p=3:  {minkowski3:.4f}")
    print()
    print("  Choice guide:")
    print("    Euclidean  -> continuous, similar scale features")
    print("    Manhattan  -> robust to outliers, high-dim")
    print("    Cosine     -> text, angle matters more than magnitude")
    print("    Hamming    -> binary/categorical features")


# -- 3. Choosing k via cross-validation ---------------------------------------
def choose_k():
    print("\n=== Choosing k via Cross-Validation ===")
    X, y = make_moons(n_samples=400, noise=0.25, random_state=1)
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    print(f"  {'k':<6} {'CV acc (mean±std)'}")
    best_k, best_acc = 1, 0
    for k in range(1, 31):
        model   = KNeighborsClassifier(n_neighbors=k)
        cv_accs = cross_val_score(model, X_s, y, cv=5)
        mean, std = cv_accs.mean(), cv_accs.std()
        marker = " <- best" if mean > best_acc else ""
        if mean > best_acc:
            best_k, best_acc = k, mean
        if k <= 15 or k == best_k:   # only print first 15 and best
            print(f"  {k:<6} {mean:.4f} ± {std:.4f}{marker}")


# -- 4. Weighted KNN -----------------------------------------------------------
def weighted_knn():
    print("\n=== Weighted KNN (inverse distance) ===")
    X, y = make_moons(n_samples=400, noise=0.25, random_state=2)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)
    scaler = StandardScaler().fit(X_tr)
    X_tr_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_te)

    for weights in ["uniform", "distance"]:
        model = KNeighborsClassifier(n_neighbors=7, weights=weights)
        model.fit(X_tr_s, y_tr)
        acc = model.score(X_te_s, y_te)
        print(f"  weights={weights:<10}: acc={acc:.4f}")


# -- 5. KNN Regression --------------------------------------------------------
def knn_regression():
    print("\n=== KNN Regression ===")
    rng = np.random.default_rng(3)
    x   = rng.uniform(-3, 3, 150)
    y   = np.sin(x) + rng.normal(0, 0.2, 150)

    X_all  = x.reshape(-1,1)
    x_test = np.linspace(-3, 3, 300).reshape(-1,1)
    y_true = np.sin(x_test.ravel())

    print(f"  {'k':<6} {'RMSE'}")
    for k in [1, 3, 7, 15, 30]:
        model = KNeighborsRegressor(n_neighbors=k)
        model.fit(X_all, y)
        rmse  = np.sqrt(np.mean((model.predict(x_test) - y_true)**2))
        print(f"  {k:<6} {rmse:.4f}")


# -- 6. Curse of dimensionality -----------------------------------------------
def curse_of_dimensionality():
    print("\n=== Curse of Dimensionality ===")
    print("  In high dimensions, all points become equidistant.")
    rng = np.random.default_rng(4)
    n   = 500
    for d in [2, 5, 10, 50, 100, 500]:
        X = rng.uniform(0, 1, (n, d))
        dists = cdist(X[:10], X, metric="euclidean")
        dmax  = dists.max(axis=1)
        dmin  = dists[dists > 0].reshape(10,-1).min(axis=1)
        ratio = (dmax - dmin) / (dmin + 1e-10)
        print(f"  d={d:>4}: mean(d_max/d_min - 1) = {ratio.mean():.4f}  "
              f"(-> 0 means less informative nearest neighbour)")


# -- 7. Decision boundary visualisation ---------------------------------------
def plot_boundaries():
    X, y = make_moons(n_samples=400, noise=0.25, random_state=5)
    scaler = StandardScaler().fit(X)
    X_s = scaler.transform(X)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, k in zip(axes, [1, 5, 15, 50]):
        model = KNeighborsClassifier(n_neighbors=k).fit(X_s, y)
        xx, yy = np.meshgrid(np.linspace(X_s[:,0].min()-0.5, X_s[:,0].max()+0.5, 150),
                             np.linspace(X_s[:,1].min()-0.5, X_s[:,1].max()+0.5, 150))
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        ax.contourf(xx, yy, Z, alpha=0.3, cmap="RdBu")
        ax.scatter(X_s[:,0], X_s[:,1], c=y, cmap="RdBu", edgecolors='k', s=15, lw=0.4)
        acc = model.score(X_s, y)
        ax.set(title=f"k={k}  acc={acc:.3f}")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "knn_boundaries.png")
    plt.savefig(path, dpi=90); plt.close()
    print(f"\n  KNN boundary plot saved: {path}")


if __name__ == "__main__":
    knn_from_scratch()
    distance_metrics()
    choose_k()
    weighted_knn()
    knn_regression()
    curse_of_dimensionality()
    plot_boundaries()
