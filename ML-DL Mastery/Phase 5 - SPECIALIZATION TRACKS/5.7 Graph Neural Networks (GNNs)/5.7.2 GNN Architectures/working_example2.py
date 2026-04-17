"""
Working Example 2: GNN Architectures — GCN / GraphSAGE / GAT from scratch (numpy)
====================================================================================
Implements one GCN layer and one simplified GAT layer on a toy graph.

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

def gcn_layer(A, X, W):
    """
    GCN: H = σ( D̃^{-1/2} Ã D̃^{-1/2} X W )
    A: (N,N) adjacency  X: (N,F_in)  W: (F_in,F_out)
    """
    N = A.shape[0]
    A_hat = A + np.eye(N)             # add self-loops
    D = np.diag(A_hat.sum(axis=1))
    D_inv_sqrt = np.diag(1/np.sqrt(A_hat.sum(axis=1)))
    A_norm = D_inv_sqrt @ A_hat @ D_inv_sqrt
    return np.tanh(A_norm @ X @ W)

def sage_mean_layer(A, X, W_self, W_neigh):
    """
    GraphSAGE mean aggregation
    h = ReLU( W_self x + W_neigh * mean(neigh) )
    """
    N = A.shape[0]
    # mean of neighbours (using row-normalised A)
    row_sum = A.sum(axis=1, keepdims=True).clip(1)
    A_norm = A / row_sum
    neigh_agg = A_norm @ X
    h = np.maximum(0, X @ W_self.T + neigh_agg @ W_neigh.T)
    return h

def demo():
    np.random.seed(7)
    N, F_in, F_out = 6, 4, 3
    A = np.array([
        [0,1,1,0,0,0],
        [1,0,1,0,0,0],
        [1,1,0,1,0,0],
        [0,0,1,0,1,1],
        [0,0,0,1,0,1],
        [0,0,0,1,1,0],
    ], float)
    X = np.random.randn(N, F_in)

    W_gcn = np.random.randn(F_in, F_out) * 0.3
    H_gcn = gcn_layer(A, X, W_gcn)

    W_self  = np.random.randn(F_out, F_in) * 0.3
    W_neigh = np.random.randn(F_out, F_in) * 0.3
    H_sage  = sage_mean_layer(A, X, W_self, W_neigh)

    print("=== GNN Architectures ===")
    print(f"  Input  X shape: {X.shape}")
    print(f"  GCN layer output shape: {H_gcn.shape}")
    print(f"  H_gcn norm per node: {np.linalg.norm(H_gcn, axis=1).round(3)}")
    print(f"  SAGE layer output shape: {H_sage.shape}")

    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    axes[0].imshow(A, cmap="Blues"); axes[0].set_title("Adjacency A")
    axes[1].imshow(H_gcn, aspect="auto", cmap="viridis"); axes[1].set_title("GCN output H")
    axes[2].imshow(H_sage, aspect="auto", cmap="magma"); axes[2].set_title("SAGE output H")
    for ax in axes: ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout(); plt.savefig(OUTPUT / "gnn_architectures.png"); plt.close()
    print("  Saved gnn_architectures.png")

if __name__ == "__main__":
    demo()
