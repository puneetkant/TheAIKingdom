"""
Working Example 2: Eigenvalues & Eigenvectors — PCA and spectral analysis
=========================================================================
Computes eigendecomposition for PCA on Cal Housing data, power iteration,
and demonstrates diagonalisation and Markov chain convergence.

Run:  python working_example2.py
"""
import csv, urllib.request
from pathlib import Path
try:
    import numpy as np
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit("pip install numpy matplotlib")

DATA   = Path(__file__).parent / "data"
OUTPUT = Path(__file__).parent / "output"
DATA.mkdir(exist_ok=True); OUTPUT.mkdir(exist_ok=True)

def download():
    dest = DATA / "cal_housing.csv"
    if not dest.exists():
        try: urllib.request.urlretrieve(
            "https://huggingface.co/datasets/scikit-learn/california-housing/resolve/main/cal_housing.csv", dest)
        except Exception:
            import random; random.seed(42)
            rows=["MedInc,HouseAge,AveRooms,AveBedrms,Population,AveOccup"]
            for _ in range(200):
                rows.append(",".join([str(round(random.uniform(1,10),3)),str(random.randint(1,52)),
                    str(round(random.uniform(3,8),3)),str(round(random.uniform(0.8,2),3)),
                    str(random.randint(100,5000)),str(round(random.uniform(2,5),3))]))
            dest.write_text("\n".join(rows))
    with open(dest) as f: rows=list(csv.DictReader(f))
    feat=["MedInc","HouseAge","AveRooms","AveBedrms","Population","AveOccup"]
    return np.array([[float(r[c]) for c in feat] for r in rows[:300]])

def demo_eigen_basics():
    print("=== Eigendecomposition ===")
    A = np.array([[4., 2.], [1., 3.]])
    eigenvals, eigenvecs = np.linalg.eig(A)
    print(f"  Eigenvalues:  {eigenvals}")
    print(f"  Eigenvectors:\n{eigenvecs}")
    # Verify: Av = λv
    for i in range(len(eigenvals)):
        v = eigenvecs[:, i]
        print(f"  Av - λv check: {(A @ v - eigenvals[i] * v).round(10)}")

def demo_pca(X: np.ndarray):
    print("\n=== PCA via Eigendecomposition of Covariance Matrix ===")
    X_c = X - X.mean(axis=0)
    cov = X_c.T @ X_c / (len(X) - 1)
    eigenvals, eigenvecs = np.linalg.eigh(cov)  # symmetric — use eigh
    # Sort descending
    idx = np.argsort(eigenvals)[::-1]
    eigenvals, eigenvecs = eigenvals[idx], eigenvecs[:, idx]
    
    total_var = eigenvals.sum()
    explained = eigenvals / total_var
    cumulative = np.cumsum(explained)
    print(f"  Explained variance ratio: {explained.round(4)}")
    print(f"  Cumulative:               {cumulative.round(4)}")
    print(f"  First 2 PCs explain: {cumulative[1]:.2%} of variance")

    # Project to 2D
    W = eigenvecs[:, :2]       # top-2 eigenvectors
    X_pca = X_c @ W            # (n, 2)
    
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.3, s=8, c="steelblue")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_title("Cal Housing: PCA projection (top 2 components)")
    fig.savefig(OUTPUT / "pca_projection.png", dpi=120, bbox_inches="tight")
    plt.close(fig); print(f"  Saved: pca_projection.png")

def power_iteration(A: np.ndarray, n_iter: int = 50):
    """Find dominant eigenvector via power iteration."""
    v = np.random.randn(A.shape[0])
    for _ in range(n_iter):
        v = A @ v
        v /= np.linalg.norm(v)
    eigenval = float(v @ A @ v)
    return eigenval, v

def demo_power_iteration():
    print("\n=== Power Iteration ===")
    A = np.array([[3., 1.], [1., 2.]])
    val, vec = power_iteration(A)
    true_vals = np.linalg.eigvals(A)
    print(f"  Power iteration λ₁ = {val:.6f}")
    print(f"  True eigenvalues:   {sorted(true_vals, reverse=True)}")

def demo_markov():
    print("\n=== Markov Chain (Stationary distribution = dominant eigenvector) ===")
    # Transition matrix (rows sum to 1)
    P = np.array([[0.7, 0.2, 0.1],
                  [0.3, 0.5, 0.2],
                  [0.1, 0.3, 0.6]])
    # Stationary: π P = π → eigenvector of P^T with eigenvalue 1
    vals, vecs = np.linalg.eig(P.T)
    # Find eigenvalue closest to 1
    idx = np.argmin(np.abs(vals - 1.0))
    pi = vecs[:, idx].real
    pi /= pi.sum()
    print(f"  Stationary distribution π = {pi.round(4)}")
    print(f"  Check P^100 rows converge: {np.linalg.matrix_power(P, 100)[0].round(4)}")

if __name__ == "__main__":
    X = download()
    demo_eigen_basics()
    demo_pca(X)
    demo_power_iteration()
    demo_markov()
