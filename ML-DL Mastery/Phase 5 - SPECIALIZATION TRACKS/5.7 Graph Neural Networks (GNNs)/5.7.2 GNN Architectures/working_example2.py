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
    GCN: H = sigma( D^{-1/2} Ã D^{-1/2} X W )
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

def demo_multi_layer_gcn():
    """Stack two GCN layers and observe feature smoothing."""
    print("\n=== Multi-Layer GCN ===")
    np.random.seed(42)
    N, F_in, F_h, F_out = 6, 4, 6, 3
    A = np.array([
        [0,1,1,0,0,0],[1,0,1,0,0,0],[1,1,0,1,0,0],
        [0,0,1,0,1,1],[0,0,0,1,0,1],[0,0,0,1,1,0],
    ], float)
    X = np.random.randn(N, F_in)
    W1 = np.random.randn(F_in, F_h) * 0.3
    W2 = np.random.randn(F_h, F_out) * 0.3
    H1 = gcn_layer(A, X, W1)
    H2 = gcn_layer(A, H1, W2)
    print(f"  Input variance:   {X.var():.4f}")
    print(f"  After layer 1:    {H1.var():.4f}")
    print(f"  After layer 2:    {H2.var():.4f}  (smoothing = variance decrease)")


def demo_feature_propagation():
    """Show how GCN spreads features across graph hops."""
    print("\n=== Feature Propagation Depth ===")
    np.random.seed(1)
    N, F = 8, 3
    # Path graph: 0-1-2-3-4-5-6-7
    A = np.zeros((N, N))
    for i in range(N-1): A[i, i+1] = A[i+1, i] = 1
    X = np.eye(N, F)   # one-hot node features
    print("  Node 0 feature influence after k hops:")
    H = X.copy()
    for k in range(1, 5):
        W = np.eye(F) * 0.9  # identity to track propagation
        H = gcn_layer(A, H, W)
        influence = H[:, 0]  # how much of node-0's feature has spread
        nonzero = (influence > 1e-6).sum()
        print(f"    k={k}: {nonzero} nodes have node-0 influence")


if __name__ == "__main__":
    demo()
    demo_multi_layer_gcn()
    demo_feature_propagation()
