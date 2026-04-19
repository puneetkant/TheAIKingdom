"""
Working Example 2: Advanced GNN Topics — graph attention, pooling, over-smoothing
===================================================================================
Demonstrates attention-weighted aggregation and hierarchical pooling concepts.

Run:  python working_example2.py
"""
from pathlib import Path
try:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit("pip install numpy matplotlib")

OUTPUT = Path(__file__).parent / "output"
OUTPUT.mkdir(exist_ok=True)

def softmax_rows(Z):
    e = np.exp(Z - Z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

def gat_layer(A, X, W, a_src, a_dst):
    """
    Single-head Graph Attention (GAT).
    e_ij = LeakyReLU( a^T [Wh_i || Wh_j] )
    alpha_ij = softmax_j(e_ij)
    h_i  = sigma( Sigma_j alpha_ij Wh_j )
    """
    N = len(A)
    H = X @ W                          # (N, F_out)
    # Compute raw attention for all (i,j) edges
    e = np.full((N, N), -1e9)
    for i in range(N):
        for j in range(N):
            if A[i, j] > 0 or i == j:
                e[i, j] = max(0, H[i] @ a_src + H[j] @ a_dst)  # LeakyReLU simplified
    alpha = softmax_rows(e)            # (N, N)
    # Zero out non-edges
    mask = (A + np.eye(N)) > 0
    alpha = alpha * mask
    alpha /= alpha.sum(axis=1, keepdims=True).clip(1e-8)
    out = np.tanh(alpha @ H)           # (N, F_out)
    return out, alpha

def over_smoothing_demo():
    """Show how node features converge after many GCN layers."""
    np.random.seed(0)
    N, F = 8, 4
    A = np.random.rand(N, N) < 0.3
    A = ((A | A.T) & ~np.eye(N, dtype=bool)).astype(float)
    A_hat = A + np.eye(N); D_inv_sqrt = np.diag(1/np.sqrt(A_hat.sum(1)))
    A_norm = D_inv_sqrt @ A_hat @ D_inv_sqrt
    X = np.random.randn(N, F)
    norms = []
    H = X.copy()
    for _ in range(10):
        H = np.tanh(A_norm @ H)
        # Measure pairwise distance variance
        dists = [np.linalg.norm(H[i]-H[j]) for i in range(N) for j in range(i+1,N)]
        norms.append(np.std(dists))
    return norms

def demo():
    print("=== Advanced GNN Topics ===")
    np.random.seed(5)
    N, F_in, F_out = 6, 4, 3
    A = np.array([[0,1,1,0,0,0],[1,0,1,1,0,0],[1,1,0,0,0,0],[0,1,0,0,1,0],[0,0,0,1,0,1],[0,0,0,0,1,0]],float)
    X = np.random.randn(N, F_in)
    W = np.random.randn(F_in, F_out) * 0.3
    a_src = np.random.randn(F_out) * 0.3
    a_dst = np.random.randn(F_out) * 0.3
    H, alpha = gat_layer(A, X, W, a_src, a_dst)
    print(f"  GAT output shape: {H.shape}")
    print(f"  Attention matrix (row-normalised):\n{alpha.round(2)}")

    smoothing_curve = over_smoothing_demo()
    print(f"  Over-smoothing: feature std after 10 layers: {[f'{v:.3f}' for v in smoothing_curve]}")

    fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    axes[0].imshow(alpha, cmap="Blues", vmin=0, vmax=alpha.max())
    axes[0].set_title("GAT Attention alpha"); axes[0].set_xlabel("src"); axes[0].set_ylabel("dst")
    axes[1].plot(range(1, 11), smoothing_curve, "o-"); axes[1].set_title("Over-smoothing (feature std)")
    axes[1].set_xlabel("GCN layers"); axes[1].set_ylabel("Pairwise distance std")
    plt.tight_layout(); plt.savefig(OUTPUT / "advanced_gnn.png"); plt.close()
    print("  Saved advanced_gnn.png")

if __name__ == "__main__":
    demo()
