"""
Working Example 2: Clustering Metrics — Silhouette, Davies-Bouldin, Calinski-Harabasz
=======================================================================================
Internal metrics (no ground truth) applied to KMeans on Cal Housing.

Run:  python working_example2.py
"""
from pathlib import Path
try:
    import numpy as np
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

def demo_internal_metrics():
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

def demo_external_metrics():
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

if __name__ == "__main__":
    demo_internal_metrics()
    demo_external_metrics()
