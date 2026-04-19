"""
Working Example: Clustering Metrics
Covers external metrics (ARI, NMI, FMI), internal metrics (silhouette,
Davies-Bouldin, Calinski-Harabasz), inertia/elbow, and metric comparison.
"""
import numpy as np
from sklearn.datasets import make_blobs, make_moons
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                              fowlkes_mallows_score, silhouette_score,
                              silhouette_samples, davies_bouldin_score,
                              calinski_harabasz_score, homogeneity_score,
                              completeness_score, v_measure_score)
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_cluster_metrics")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -- 1. External metrics (require ground truth) -------------------------------
def external_metrics():
    print("=== External Clustering Metrics (require ground truth) ===")
    X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.8, random_state=0)
    y_pred    = KMeans(n_clusters=4, n_init=10, random_state=0).fit_predict(X)
    y_wrong   = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(X)

    print(f"\n  {'Metric':<38} {'Correct k=4':>14} {'Wrong k=2':>12}")
    for name, fn in [
        ("ARI (Adjusted Rand Index)",          adjusted_rand_score),
        ("NMI (Normalised Mutual Info)",       normalized_mutual_info_score),
        ("FMI (Fowlkes-Mallows Index)",        fowlkes_mallows_score),
        ("Homogeneity",                        homogeneity_score),
        ("Completeness",                       completeness_score),
        ("V-measure (HM of H & C)",            v_measure_score),
    ]:
        c4 = fn(y_true, y_pred)
        c2 = fn(y_true, y_wrong)
        print(f"  {name:<38} {c4:>14.4f} {c2:>12.4f}")

    print(f"\n  Note:")
    print(f"    ARI in [-1,1]: 1=perfect, 0=random, negative=worse than random")
    print(f"    NMI in [0,1]:  1=perfect agreement, 0=independent")
    print(f"    ARI is adjusted for chance (unbiased); NMI is not adjusted")


# -- 2. Internal metrics (no ground truth needed) -----------------------------
def internal_metrics():
    print("\n=== Internal Clustering Metrics (no ground truth needed) ===")
    X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.8, random_state=0)

    print(f"\n  {'k':<5} {'Silhouette':>14} {'Davies-Bouldin':>18} {'Calinski-H':>14} {'Inertia':>12}")
    for k in range(2, 9):
        km    = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
        labels = km.labels_
        sil = silhouette_score(X, labels)
        db  = davies_bouldin_score(X, labels)
        ch  = calinski_harabasz_score(X, labels)
        ine = km.inertia_
        arrow = " <- best" if k == 4 else ""
        print(f"  {k:<5} {sil:>14.4f} {db:>18.4f} {ch:>14.2f} {ine:>12.2f}{arrow}")

    print(f"\n  Silhouette: closer to 1 is better (^)")
    print(f"  Davies-Bouldin: closer to 0 is better (v)")
    print(f"  Calinski-Harabasz: higher is better (^)")
    print(f"  Inertia: elbow in inertia vs k curve")


# -- 3. Silhouette analysis ---------------------------------------------------
def silhouette_analysis():
    print("\n=== Silhouette Analysis ===")
    print("  s(i) = (b-a) / max(a,b)")
    print("  a = mean dist to same-cluster points (cohesion)")
    print("  b = mean dist to nearest other cluster (separation)")
    print("  s ~= 1: well clustered   s ~= 0: borderline   s < 0: misclassified")

    X, _ = make_blobs(n_samples=200, centers=3, cluster_std=0.6, random_state=1)
    km   = KMeans(n_clusters=3, n_init=10, random_state=0).fit(X)
    labels = km.labels_
    sil_vals = silhouette_samples(X, labels)

    print(f"\n  Global silhouette score: {silhouette_score(X, labels):.4f}")
    for c in range(3):
        mask = labels == c
        cluster_sil = sil_vals[mask]
        print(f"  Cluster {c}: n={mask.sum()}  mean_sil={cluster_sil.mean():.4f}"
              f"  min_sil={cluster_sil.min():.4f}")

    # Silhouette plot
    fig, ax = plt.subplots(figsize=(8, 5))
    y_lower = 10
    colors = plt.cm.Set1(np.linspace(0, 1, 3))
    for c in range(3):
        vals   = np.sort(sil_vals[labels == c])
        y_upper = y_lower + len(vals)
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, vals,
                         facecolor=colors[c], alpha=0.7, label=f"Cluster {c}")
        y_lower = y_upper + 10
    ax.axvline(silhouette_score(X, labels), color='r', lw=2, linestyle='--',
               label=f"Mean sil={silhouette_score(X,labels):.3f}")
    ax.set(xlabel="Silhouette coefficient", ylabel="Cluster", title="Silhouette Analysis")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "silhouette_plot.png")
    plt.savefig(path, dpi=90); plt.close()
    print(f"\n  Silhouette plot saved: {path}")


# -- 4. Choosing k: elbow and silhouette --------------------------------------
def choosing_k():
    print("\n=== Choosing k: Elbow + Silhouette ===")
    X, _ = make_blobs(n_samples=400, centers=5, cluster_std=0.9, random_state=2)
    ks = range(2, 12)
    inertias  = []
    sil_scores = []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X, km.labels_))

    # Find elbow using second derivative
    second_diff = np.diff(np.diff(inertias))
    elbow_k     = list(ks)[second_diff.argmax() + 2]
    sil_k       = list(ks)[np.argmax(sil_scores)]
    print(f"  Elbow method suggests:     k = {elbow_k}")
    print(f"  Silhouette suggests:       k = {sil_k}")
    print(f"  True number of clusters:   k = 5")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(list(ks), inertias, 'bo-'); ax1.axvline(elbow_k, color='r', lw=2, linestyle='--')
    ax1.set(xlabel="k", ylabel="Inertia", title="Elbow Curve"); ax1.grid(True, alpha=0.3)
    ax2.plot(list(ks), sil_scores, 'go-'); ax2.axvline(sil_k, color='r', lw=2, linestyle='--')
    ax2.set(xlabel="k", ylabel="Silhouette score", title="Silhouette vs k"); ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "choosing_k.png")
    plt.savefig(path, dpi=90); plt.close()
    print(f"\n  Elbow/silhouette plots saved: {path}")


# -- 5. Algorithm comparison with metrics -------------------------------------
def algorithm_metric_comparison():
    print("\n=== Algorithm × Metric Comparison ===")
    X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.9, random_state=3)

    algorithms = [
        ("KMeans (k=4)",   KMeans(n_clusters=4, n_init=10, random_state=0)),
        ("KMeans (k=2)",   KMeans(n_clusters=2, n_init=10, random_state=0)),
        ("Agglo (k=4)",    AgglomerativeClustering(n_clusters=4)),
        ("Agglo (k=4,sl)", AgglomerativeClustering(n_clusters=4, linkage="single")),
        ("DBSCAN",         DBSCAN(eps=1.2, min_samples=5)),
    ]
    print(f"  {'Algorithm':<22} {'ARI':>8} {'NMI':>8} {'Sil':>8} {'DB':>8}")
    for name, alg in algorithms:
        labels = alg.fit_predict(X)
        if len(set(labels)) < 2:
            print(f"  {name:<22} {'—':>8} {'—':>8} {'—':>8} {'—':>8}  (single cluster)")
            continue
        ari = adjusted_rand_score(y_true, labels)
        nmi = normalized_mutual_info_score(y_true, labels)
        sil = silhouette_score(X, labels)
        db  = davies_bouldin_score(X, labels)
        print(f"  {name:<22} {ari:>8.4f} {nmi:>8.4f} {sil:>8.4f} {db:>8.4f}")


if __name__ == "__main__":
    external_metrics()
    internal_metrics()
    silhouette_analysis()
    choosing_k()
    algorithm_metric_comparison()
