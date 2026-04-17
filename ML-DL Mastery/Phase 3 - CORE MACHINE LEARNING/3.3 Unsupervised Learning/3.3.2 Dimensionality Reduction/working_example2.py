"""
Working Example 2: Dimensionality Reduction — PCA, t-SNE, UMAP, Cal Housing
=============================================================================
PCA scree plot, explained variance, t-SNE 2D visualisation, reconstruction error.

Run:  python working_example2.py
"""
from pathlib import Path
try:
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.datasets import fetch_california_housing
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA, TruncatedSVD
    from sklearn.manifold import TSNE
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_squared_error
except ImportError:
    raise SystemExit("pip install numpy matplotlib scikit-learn")

OUTPUT = Path(__file__).parent / "output"
OUTPUT.mkdir(exist_ok=True)

def demo_pca():
    print("=== PCA (Cal Housing) ===")
    h = fetch_california_housing()
    X = StandardScaler().fit_transform(h.data)
    pca = PCA().fit(X)
    evr = pca.explained_variance_ratio_
    cumvar = evr.cumsum()
    for i, (ev, cv) in enumerate(zip(evr, cumvar), 1):
        print(f"  PC{i}: var={ev:.4f}  cumulative={cv:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(range(1, len(evr)+1), evr); axes[0].set_xlabel("PC"); axes[0].set_ylabel("Explained variance")
    axes[0].set_title("Scree plot")
    axes[1].plot(range(1, len(cumvar)+1), cumvar, "o-")
    axes[1].axhline(0.95, color="r", ls="--", label="95%")
    axes[1].set_xlabel("n_components"); axes[1].set_ylabel("Cumulative variance"); axes[1].legend()
    axes[1].set_title("Cumulative explained variance")
    fig.savefig(OUTPUT / "pca_scree.png", dpi=120, bbox_inches="tight")
    plt.close(fig); print("  Saved: pca_scree.png")

    # n_components to reach 95%
    n95 = int(np.argmax(cumvar >= 0.95)) + 1
    print(f"\n  Components for 95% variance: {n95}/{len(evr)}")

def demo_reconstruction():
    print("\n=== PCA Reconstruction Error ===")
    h = fetch_california_housing()
    scaler = StandardScaler()
    X = scaler.fit_transform(h.data)
    for k in [1, 2, 3, 5, 8]:
        pca = PCA(n_components=k)
        Xr  = pca.inverse_transform(pca.fit_transform(X))
        err = np.mean((X - Xr)**2)
        print(f"  k={k}: reconstruction MSE={err:.4f}  retained var={pca.explained_variance_ratio_.sum():.4f}")

def demo_pca_regression():
    print("\n=== PCA + Ridge Regression ===")
    h = fetch_california_housing()
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    X, y = h.data, h.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    for k in [2, 4, 6, 8]:
        pipe = make_pipeline(StandardScaler(), PCA(n_components=k), Ridge(1.0))
        pipe.fit(X_train, y_train)
        rmse = mean_squared_error(y_test, pipe.predict(X_test))**0.5
        print(f"  PCA({k}) + Ridge: RMSE={rmse:.4f}")

if __name__ == "__main__":
    demo_pca()
    demo_reconstruction()
    demo_pca_regression()
