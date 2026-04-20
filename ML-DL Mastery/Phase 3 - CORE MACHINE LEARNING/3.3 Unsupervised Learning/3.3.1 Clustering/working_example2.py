"""
Working Example 2: Clustering — K-Means, DBSCAN, Hierarchical, Cal Housing
===========================================================================
K-Means with elbow/silhouette, DBSCAN density clustering, Agglomerative.

Run:  python working_example2.py
"""
from pathlib import Path
try:
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.datasets import fetch_california_housing
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
    from sklearn.metrics import silhouette_score
    from sklearn.decomposition import PCA
except ImportError:
    raise SystemExit("pip install numpy matplotlib scikit-learn")

OUTPUT = Path(__file__).parent / "output"
OUTPUT.mkdir(exist_ok=True)

def demo_kmeans():
    print("=== K-Means Clustering (Cal Housing) ===")
    h = fetch_california_housing()
    X = StandardScaler().fit_transform(h.data)

    inertias, sil_scores = [], []
    for k in range(2, 11):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X, labels, sample_size=2000, random_state=42))
        print(f"  k={k}: inertia={km.inertia_:.1f}  silhouette={sil_scores[-1]:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(range(2, 11), inertias, "o-"); axes[0].set_xlabel("k"); axes[0].set_ylabel("Inertia")
    axes[0].set_title("Elbow curve")
    axes[1].plot(range(2, 11), sil_scores, "s-"); axes[1].set_xlabel("k"); axes[1].set_ylabel("Silhouette")
    axes[1].set_title("Silhouette score")
    fig.savefig(OUTPUT / "kmeans_elbow.png", dpi=120, bbox_inches="tight")
    plt.close(fig); print("  Saved: kmeans_elbow.png")

def demo_dbscan():
    print("\n=== DBSCAN ===")
    h = fetch_california_housing()
    X = StandardScaler().fit_transform(h.data[:3000])
    for eps in [0.5, 1.0, 1.5]:
        db = DBSCAN(eps=eps, min_samples=10)
        labels = db.fit_predict(X)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise    = (labels == -1).sum()
        print(f"  eps={eps}: clusters={n_clusters}  noise_pts={n_noise}")

def demo_agglomerative():
    print("\n=== Agglomerative Clustering ===")
    h = fetch_california_housing()
    X = StandardScaler().fit_transform(h.data[:2000])
    for linkage in ["ward", "average", "complete"]:
        agg = AgglomerativeClustering(n_clusters=4, linkage=linkage)
        labels = agg.fit_predict(X)
        sil = silhouette_score(X, labels, sample_size=1000, random_state=42)
        print(f"  linkage={linkage:8s}: silhouette={sil:.4f}")

def demo_gaussian_mixture():
    """Gaussian Mixture Model as a soft clustering method."""
    print("\n=== Gaussian Mixture Model ===")
    from sklearn.mixture import GaussianMixture
    h = fetch_california_housing()
    X = StandardScaler().fit_transform(h.data)
    for n in [2, 4, 6, 8]:
        gmm = GaussianMixture(n_components=n, covariance_type="full", random_state=42, n_init=3)
        gmm.fit(X)
        bic = gmm.bic(X)
        sil = silhouette_score(X, gmm.predict(X), sample_size=2000, random_state=42)
        print(f"  components={n}: BIC={bic:.1f}  silhouette={sil:.4f}")


def demo_minibatch_kmeans():
    """MiniBatchKMeans — faster approximation for large datasets."""
    print("\n=== MiniBatchKMeans (speed comparison) ===")
    import time
    from sklearn.cluster import MiniBatchKMeans
    h = fetch_california_housing()
    X = StandardScaler().fit_transform(h.data)   # full 20 640 rows
    for k in [4, 8]:
        t0 = time.perf_counter()
        km = KMeans(n_clusters=k, random_state=42, n_init=5)
        km.fit(X); t_km = time.perf_counter() - t0

        t0 = time.perf_counter()
        mbkm = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=5, batch_size=1024)
        mbkm.fit(X); t_mb = time.perf_counter() - t0

        inertia_diff = abs(km.inertia_ - mbkm.inertia_) / km.inertia_ * 100
        print(f"  k={k}: KMeans {t_km:.2f}s  MiniBatch {t_mb:.2f}s  "
              f"inertia diff={inertia_diff:.2f}%")


def demo_cluster_pca_plot():
    """Visualise K-Means clusters in 2-D PCA space."""
    print("\n=== Cluster Visualisation (PCA 2D) ===")
    h = fetch_california_housing()
    X = StandardScaler().fit_transform(h.data)
    km = KMeans(n_clusters=4, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    X2d = PCA(n_components=2, random_state=42).fit_transform(X)
    fig, ax = plt.subplots(figsize=(7, 5))
    for c in range(4):
        mask = labels == c
        ax.scatter(X2d[mask, 0], X2d[mask, 1], s=5, alpha=0.4, label=f"Cluster {c}")
    ax.set_title("K-Means (k=4) — PCA 2D projection")
    ax.legend(markerscale=4)
    fig.savefig(OUTPUT / "kmeans_pca.png", dpi=120, bbox_inches="tight")
    plt.close(fig); print("  Saved: kmeans_pca.png")


if __name__ == "__main__":
    demo_kmeans()
    demo_dbscan()
    demo_agglomerative()
    demo_gaussian_mixture()
    demo_minibatch_kmeans()
    demo_cluster_pca_plot()
