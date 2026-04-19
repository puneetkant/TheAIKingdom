"""
Working Example 2: Clustering Metrics — Silhouette, Davies-Bouldin, Calinski-Harabasz
=======================================================================================
Internal metrics (no ground truth) applied to KMeans on Cal Housing.

Run:  python working_example2.py
"""
from pathlib import Path
try:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.datasets import fetch_california_housing
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                                  calinski_harabasz_score, adjusted_rand_score,
                                  normalized_mutual_info_score)
except ImportError:
    raise SystemExit("pip install numpy matplotlib scikit-learn")

OUTPUT = Path(__file__).parent / "output"
OUTPUT.mkdir(exist_ok=True)

def demo_internal_metrics_housing():
    print("=== Internal Clustering Metrics (no labels) ===")
    h = fetch_california_housing()
    X = StandardScaler().fit_transform(h.data)

    print(f"  {'k':>4}  {'Silhouette':>12}  {'Davies-Bouldin':>15}  {'Calinski':>12}")
    for k in range(2, 11):
        km = KMeans(k, random_state=42, n_init=5)
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels, sample_size=2000, random_state=42)
        dbi = davies_bouldin_score(X, labels)
        chi = calinski_harabasz_score(X, labels)
        print(f"  {k:>4}  {sil:>12.4f}  {dbi:>15.4f}  {chi:>12.1f}")

def demo_external_metrics_housing():
    """External metrics when ground truth labels exist."""
    print("\n=== External Metrics (simulated ground truth) ===")
    h = fetch_california_housing()
    X = StandardScaler().fit_transform(h.data)
    # Simulate 4 true "geo clusters" via quantile of lat/lon
    import pandas as pd
    df = pd.DataFrame(h.data, columns=h.feature_names)
    lat_cut = df["Latitude"].median()
    lon_cut = df["Longitude"].median()
    true_labels = ((df["Latitude"] > lat_cut).astype(int) * 2 +
                   (df["Longitude"] > lon_cut).astype(int))

    km = KMeans(4, random_state=42, n_init=10)
    pred_labels = km.fit_predict(X)
    ari = adjusted_rand_score(true_labels, pred_labels)
    nmi = normalized_mutual_info_score(true_labels, pred_labels)
    print(f"  KMeans(k=4) vs geo-quantile labels:")
    print(f"    ARI={ari:.4f}  NMI={nmi:.4f}")


def demo_internal_metrics():
    """KMeans on make_blobs, compute silhouette/CH/DB for k=2..7."""
    print("=== Internal Metrics on Make-Blobs ===")
    from sklearn.datasets import make_blobs
    X, _ = make_blobs(n_samples=500, centers=4, random_state=42)
    print("  {:>4}  {:>12}  {:>12}  {:>10}".format("k", "Silhouette", "Calinski-H", "Davies-B"))
    for k in range(2, 8):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels)
        chi = calinski_harabasz_score(X, labels)
        dbi = davies_bouldin_score(X, labels)
        print(f"  {k:>4}  {sil:>12.4f}  {chi:>12.1f}  {dbi:>10.4f}")


def demo_external_metrics():
    """AgglomerativeClustering on iris vs true labels: ARI, NMI, homogeneity."""
    print("\n=== External Metrics: AgglomerativeClustering on Iris ===")
    from sklearn.datasets import load_iris
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import homogeneity_score
    iris = load_iris()
    X = StandardScaler().fit_transform(iris.data)
    true_labels = iris.target
    for linkage in ["ward", "complete", "average"]:
        agg = AgglomerativeClustering(n_clusters=3, linkage=linkage)
        pred = agg.fit_predict(X)
        ari = adjusted_rand_score(true_labels, pred)
        nmi = normalized_mutual_info_score(true_labels, pred)
        hom = homogeneity_score(true_labels, pred)
        print(f"  linkage={linkage:<10} ARI={ari:.4f}  NMI={nmi:.4f}  Hom={hom:.4f}")


def demo_elbow_method():
    """Inertia curve for KMeans k=1..10 on blobs, save clustering_elbow.png."""
    print("\n=== Elbow Method (KMeans Inertia) ===")
    from sklearn.datasets import make_blobs
    X, _ = make_blobs(n_samples=500, centers=4, random_state=42)
    ks = list(range(1, 11))
    inertias = []
    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X)
        inertias.append(km.inertia_)
        print(f"  k={k:>2}: inertia={km.inertia_:.2f}")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ks, inertias, "o-")
    ax.set_xlabel("k")
    ax.set_ylabel("Inertia")
    ax.set_title("KMeans Elbow Curve")
    fig.savefig(OUTPUT / "clustering_elbow.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: clustering_elbow.png")


def demo_silhouette_plot():
    """Per-sample silhouette bar chart for best k, save clustering_silhouette.png."""
    print("\n=== Silhouette Plot for Best k ===")
    from sklearn.datasets import make_blobs
    from sklearn.metrics import silhouette_samples
    X, _ = make_blobs(n_samples=300, centers=4, random_state=42)
    best_k, best_score = 2, -1.0
    for k in range(2, 8):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        score = silhouette_score(X, labels)
        if score > best_score:
            best_score, best_k = score, k
    print(f"  Best k={best_k} (silhouette={best_score:.4f})")
    km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    sil_vals = silhouette_samples(X, labels)
    y_lower = 0
    fig, ax = plt.subplots(figsize=(7, 5))
    for c in range(best_k):
        c_vals = np.sort(sil_vals[labels == c])
        y_upper = y_lower + len(c_vals)
        ax.barh(range(y_lower, y_upper), c_vals, height=1.0)
        y_lower = y_upper + 2
    ax.axvline(x=best_score, color="red", linestyle="--",
               label=f"Mean={best_score:.3f}")
    ax.set_xlabel("Silhouette coefficient")
    ax.set_ylabel("Sample (by cluster)")
    ax.set_title(f"Silhouette Plot (k={best_k})")
    ax.legend()
    fig.savefig(OUTPUT / "clustering_silhouette.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: clustering_silhouette.png")


if __name__ == "__main__":
    demo_internal_metrics_housing()
    demo_external_metrics_housing()
    demo_internal_metrics()
    demo_external_metrics()
    demo_elbow_method()
    demo_silhouette_plot()
