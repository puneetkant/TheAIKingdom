"""
Working Example: Clustering
Covers K-Means, K-Means++, DBSCAN, hierarchical clustering, GMM,
evaluation metrics (silhouette, Davies-Bouldin), and choosing k.
"""
import numpy as np
from sklearn.datasets import make_blobs, make_moons
from sklearn.cluster import (KMeans, DBSCAN, AgglomerativeClustering,
                              MeanShift, SpectralClustering)
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                              calinski_harabasz_score, adjusted_rand_score)
from sklearn.preprocessing import StandardScaler
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_clustering")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── 1. K-Means from scratch ───────────────────────────────────────────────────
def kmeans_scratch():
    print("=== K-Means From Scratch ===")

    class KMeans_scratch:
        def __init__(self, k=3, max_iter=100, tol=1e-4, random_state=0):
            self.k, self.max_iter, self.tol = k, max_iter, tol
            self.rng = np.random.default_rng(random_state)

        def fit(self, X):
            idx = self.rng.choice(len(X), self.k, replace=False)
            self.centers = X[idx].copy()
            for i in range(self.max_iter):
                # Assign
                dists  = np.linalg.norm(X[:, None] - self.centers[None], axis=2)
                labels = dists.argmin(axis=1)
                # Update
                new_c  = np.array([X[labels==j].mean(0) if (labels==j).any()
                                   else self.centers[j] for j in range(self.k)])
                if np.linalg.norm(new_c - self.centers) < self.tol:
                    print(f"  Converged at iteration {i+1}")
                    break
                self.centers = new_c
            self.labels_ = labels
            self.inertia_ = np.sum((X - self.centers[labels])**2)
            return self

    X, y_true = make_blobs(n_samples=300, centers=4, random_state=0, cluster_std=0.9)
    model = KMeans_scratch(k=4).fit(X)
    ari   = adjusted_rand_score(y_true, model.labels_)
    print(f"  Inertia: {model.inertia_:.2f}  ARI: {ari:.4f}")


# ── 2. K-Means++ initialisation and sklearn ───────────────────────────────────
def kmeans_sklearn():
    print("\n=== K-Means++ and sklearn ===")
    X, y_true = make_blobs(n_samples=500, centers=5, random_state=1)

    for init in ["random", "k-means++"]:
        arises = []
        for _ in range(5):
            km = KMeans(n_clusters=5, init=init, n_init=1).fit(X)
            arises.append(adjusted_rand_score(y_true, km.labels_))
        print(f"  {init:<12}: ARI mean={np.mean(arises):.4f}  "
              f"std={np.std(arises):.4f}  inertia={km.inertia_:.0f}")


# ── 3. Choosing k: elbow and silhouette ───────────────────────────────────────
def choose_k():
    print("\n=== Choosing k: Elbow + Silhouette ===")
    X, _ = make_blobs(n_samples=300, centers=4, random_state=2)

    print(f"  {'k':<5} {'Inertia':<12} {'Silhouette':<14} {'Davies-Bouldin'}")
    for k in range(2, 9):
        km = KMeans(n_clusters=k, random_state=0, n_init=10).fit(X)
        sil = silhouette_score(X, km.labels_)
        db  = davies_bouldin_score(X, km.labels_)
        print(f"  {k:<5} {km.inertia_:<12.2f} {sil:<14.4f} {db:.4f}")


# ── 4. DBSCAN ────────────────────────────────────────────────────────────────
def dbscan_demo():
    print("\n=== DBSCAN (Density-Based Spatial Clustering) ===")
    print("  Parameters: eps (radius), min_samples (core point threshold)")
    print("  Handles noise (label=-1) and arbitrary shapes")

    X, y = make_moons(n_samples=400, noise=0.08, random_state=3)

    print(f"\n  {'eps':<8} {'min_samples':<14} {'n_clusters':<13} {'n_noise':<10} {'ARI'}")
    for eps in [0.1, 0.2, 0.3]:
        for ms in [3, 5, 10]:
            db = DBSCAN(eps=eps, min_samples=ms).fit(X)
            labels = db.labels_
            n_clust = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = (labels == -1).sum()
            ari     = adjusted_rand_score(y, labels) if n_clust > 0 else 0
            print(f"  {eps:<8} {ms:<14} {n_clust:<13} {n_noise:<10} {ari:.4f}")


# ── 5. Hierarchical clustering ───────────────────────────────────────────────
def hierarchical_clustering():
    print("\n=== Hierarchical Clustering ===")
    from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
    X, y = make_blobs(n_samples=150, centers=4, random_state=4)

    print(f"  {'Linkage':<12} {'n_clusters=4 ARI'}")
    for link in ["single", "complete", "average", "ward"]:
        model = AgglomerativeClustering(n_clusters=4, linkage=link)
        labels = model.fit_predict(X)
        ari   = adjusted_rand_score(y, labels)
        print(f"  {link:<12} {ari:.4f}")

    # Dendrogram
    Z = linkage(X, method="ward")
    fig, ax = plt.subplots(figsize=(12, 4))
    dendrogram(Z, ax=ax, no_labels=True, color_threshold=Z[-4, 2])
    ax.set(xlabel="Samples", ylabel="Distance", title="Dendrogram (Ward linkage)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "dendrogram.png")
    plt.savefig(path, dpi=90); plt.close()
    print(f"  Dendrogram saved: {path}")


# ── 6. Gaussian Mixture Model ────────────────────────────────────────────────
def gmm_demo():
    print("\n=== Gaussian Mixture Model (GMM) ===")
    print("  Soft clustering: P(x) = Σ_k π_k N(x; μ_k, Σ_k)")
    print("  Trained via EM algorithm")

    X, y = make_blobs(n_samples=400, centers=4, random_state=5)

    print(f"\n  {'n_components':<15} {'covariance_type':<18} {'ARI':<10} {'BIC'}")
    for n_comp in [2, 3, 4, 5]:
        for cov_type in ["spherical", "diag", "full"]:
            gmm = GaussianMixture(n_components=n_comp, covariance_type=cov_type,
                                  random_state=0)
            gmm.fit(X)
            labels = gmm.predict(X)
            ari    = adjusted_rand_score(y, labels)
            bic    = gmm.bic(X)
            print(f"  {n_comp:<15} {cov_type:<18} {ari:<10.4f} {bic:.1f}")


# ── 7. Clustering algorithm comparison ───────────────────────────────────────
def clustering_comparison():
    print("\n=== Clustering Algorithm Comparison ===")
    # Moons: non-convex clusters
    X_moons, y_moons = make_moons(n_samples=300, noise=0.05, random_state=6)
    # Blobs: convex clusters
    X_blobs, y_blobs = make_blobs(n_samples=300, centers=4, random_state=6)

    print(f"  {'Algorithm':<25} {'Moons ARI':<12} {'Blobs ARI'}")
    for name, algo_m, algo_b in [
        ("KMeans(k=2)",   KMeans(n_clusters=2, random_state=0, n_init=10),
                          KMeans(n_clusters=4, random_state=0, n_init=10)),
        ("DBSCAN",        DBSCAN(eps=0.15, min_samples=5),
                          DBSCAN(eps=1.0,  min_samples=5)),
        ("Aggl.(ward)",   AgglomerativeClustering(n_clusters=2, linkage="ward"),
                          AgglomerativeClustering(n_clusters=4, linkage="ward")),
        ("GMM",           GaussianMixture(n_components=2, random_state=0),
                          GaussianMixture(n_components=4, random_state=0)),
        ("SpectralClust", SpectralClustering(n_clusters=2, affinity="nearest_neighbors", random_state=0, n_jobs=-1),
                          SpectralClustering(n_clusters=4, affinity="rbf",              random_state=0, n_jobs=-1)),
    ]:
        lm = algo_m.fit_predict(X_moons) if hasattr(algo_m,'fit_predict') else algo_m.fit(X_moons).predict(X_moons)
        lb = algo_b.fit_predict(X_blobs) if hasattr(algo_b,'fit_predict') else algo_b.fit(X_blobs).predict(X_blobs)
        ari_m = adjusted_rand_score(y_moons, lm)
        ari_b = adjusted_rand_score(y_blobs, lb)
        print(f"  {name:<25} {ari_m:<12.4f} {ari_b:.4f}")


if __name__ == "__main__":
    kmeans_scratch()
    kmeans_sklearn()
    choose_k()
    dbscan_demo()
    hierarchical_clustering()
    gmm_demo()
    clustering_comparison()
