"""
Working Example: Dimensionality Reduction
Covers PCA (from scratch and sklearn), LDA, t-SNE, UMAP (if available),
autoencoders concept, and reconstruction error analysis.
"""
import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD, KernelPCA, FastICA
from sklearn.manifold import TSNE, MDS
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.datasets import load_digits, load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_dimred")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── 1. PCA from scratch ───────────────────────────────────────────────────────
def pca_scratch():
    print("=== PCA From Scratch ===")
    iris = load_iris()
    X, y = iris.data, iris.target
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    # Covariance matrix
    C = (X_s.T @ X_s) / (len(X_s) - 1)

    # Eigendecomposition
    vals, vecs = np.linalg.eigh(C)
    idx = vals.argsort()[::-1]
    vals, vecs = vals[idx], vecs[:, idx]

    # Explained variance ratio
    evr = vals / vals.sum()
    print(f"  Explained variance ratio: {evr.round(4)}")
    print(f"  Cumulative:               {evr.cumsum().round(4)}")

    # Project to 2D
    X_pca = X_s @ vecs[:, :2]
    print(f"\n  Original shape: {X_s.shape} → PCA(2): {X_pca.shape}")

    # Compare with sklearn
    pca = PCA(n_components=2)
    X_sk = pca.fit_transform(X_s)
    # Flip signs if needed (PCA components are sign-ambiguous)
    if np.corrcoef(X_pca[:,0], X_sk[:,0])[0,1] < 0:
        X_sk[:,0] *= -1
    diff = np.abs(np.abs(X_pca) - np.abs(X_sk)).max()
    print(f"  Max |scratch - sklearn|: {diff:.8f}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, X_plot, title in [(axes[0], X_pca, "PCA (from scratch)"),
                               (axes[1], X_sk, "PCA (sklearn)")]:
        colors = ["blue","red","green"]
        for c, name, col in zip([0,1,2], iris.target_names, colors):
            ax.scatter(X_plot[y==c,0], X_plot[y==c,1], label=name, color=col,
                       alpha=0.7, s=30, edgecolors='k', lw=0.4)
        ax.set(xlabel="PC1", ylabel="PC2", title=title)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "pca_iris.png")
    plt.savefig(path, dpi=90); plt.close()
    print(f"  PCA plot saved: {path}")


# ── 2. PCA: reconstruction and choosing n_components ─────────────────────────
def pca_reconstruction():
    print("\n=== PCA Reconstruction Error ===")
    digits = load_digits()
    X, y   = digits.data, digits.target
    X_s    = StandardScaler().fit_transform(X)

    print(f"  {'n_components':<15} {'Expl. var %':<14} {'Recon. MSE':<14} {'Downstream acc'}")
    for n in [2, 5, 10, 20, 40, 64]:
        pca  = PCA(n_components=n)
        X_r  = pca.fit_transform(X_s)
        X_re = pca.inverse_transform(X_r)
        mse  = np.mean((X_s - X_re)**2)
        evr  = pca.explained_variance_ratio_.sum() * 100
        # Downstream classification
        X_tr, X_te, y_tr, y_te = train_test_split(X_r, y, test_size=0.3, random_state=0)
        acc = LogisticRegression(max_iter=500).fit(X_tr, y_tr).score(X_te, y_te)
        print(f"  {n:<15} {evr:<14.1f} {mse:<14.6f} {acc:.4f}")


# ── 3. Kernel PCA ─────────────────────────────────────────────────────────────
def kernel_pca():
    print("\n=== Kernel PCA ===")
    from sklearn.datasets import make_moons
    X, y = make_moons(n_samples=300, noise=0.05, random_state=0)
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    print(f"  {'Kernel':<15} {'params':<20} {'LR acc (2 PCs)'}")
    for kernel, params in [
        ("linear",  {}),
        ("rbf",     {"gamma": 0.5}),
        ("rbf",     {"gamma": 2.0}),
        ("poly",    {"degree": 3, "gamma": 1}),
        ("sigmoid", {"gamma": 0.5}),
    ]:
        kpca = KernelPCA(n_components=2, kernel=kernel, **params)
        X_k  = kpca.fit_transform(X_s)
        X_tr, X_te, y_tr, y_te = train_test_split(X_k, y, test_size=0.3, random_state=0)
        acc = LogisticRegression().fit(X_tr, y_tr).score(X_te, y_te)
        print(f"  {kernel:<15} {str(params):<20} {acc:.4f}")


# ── 4. t-SNE for visualisation ───────────────────────────────────────────────
def tsne_demo():
    print("\n=== t-SNE ===")
    print("  Minimises KL divergence between high-dim and low-dim joint distributions")
    print("  Best for 2D/3D visualisation; NOT for downstream ML (non-deterministic)")

    digits = load_digits()
    X_s    = StandardScaler().fit_transform(digits.data)
    # Reduce to 50 dims with PCA first (speeds up t-SNE)
    X_pca  = PCA(n_components=50).fit_transform(X_s)

    print(f"\n  Running t-SNE on Digits dataset ({X_pca.shape})...")
    for perplexity in [5, 30]:
        tsne  = TSNE(n_components=2, perplexity=perplexity, random_state=0, n_iter=500)
        X_ts  = tsne.fit_transform(X_pca[:500])   # subset for speed
        print(f"  perplexity={perplexity}: shape={X_ts.shape}  "
              f"KL divergence={tsne.kl_divergence_:.4f}")

    # Plot with perplexity=30
    tsne  = TSNE(n_components=2, perplexity=30, random_state=0, n_iter=1000)
    X_ts  = tsne.fit_transform(X_pca[:500])
    y_sub = digits.target[:500]

    fig, ax = plt.subplots(figsize=(8, 7))
    scatter = ax.scatter(X_ts[:,0], X_ts[:,1], c=y_sub, cmap="tab10", s=15, alpha=0.8)
    plt.colorbar(scatter, ax=ax, label="Digit class")
    ax.set(title="t-SNE of Digits (n=500, perplexity=30)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "tsne_digits.png")
    plt.savefig(path, dpi=90); plt.close()
    print(f"  t-SNE plot saved: {path}")


# ── 5. ICA (Independent Component Analysis) ──────────────────────────────────
def ica_demo():
    print("\n=== Independent Component Analysis (ICA) ===")
    print("  PCA: uncorrelated components (max variance)")
    print("  ICA: statistically independent components (non-Gaussian sources)")

    rng = np.random.default_rng(0)
    n   = 1000
    # Two independent sources: sine and sawtooth-like
    t   = np.linspace(0, 4, n)
    s1  = np.sin(2*np.pi*5*t)
    s2  = ((t*7) % 1) * 2 - 1   # sawtooth

    S = np.column_stack([s1, s2])
    # Mix
    A = np.array([[1.0, 0.8], [0.3, 1.0]])
    X = S @ A.T + rng.normal(0, 0.05, (n,2))

    ica = FastICA(n_components=2, random_state=0)
    S_hat = ica.fit_transform(X)

    corr_s1 = max(abs(np.corrcoef(s1, S_hat[:,0])[0,1]),
                  abs(np.corrcoef(s1, S_hat[:,1])[0,1]))
    corr_s2 = max(abs(np.corrcoef(s2, S_hat[:,0])[0,1]),
                  abs(np.corrcoef(s2, S_hat[:,1])[0,1]))
    print(f"  Correlation with s1: {corr_s1:.4f}")
    print(f"  Correlation with s2: {corr_s2:.4f}")
    print(f"  (High → ICA recovered the original sources)")


if __name__ == "__main__":
    pca_scratch()
    pca_reconstruction()
    kernel_pca()
    tsne_demo()
    ica_demo()
