"""
Working Example 2: GNN Applications — social network community detection, molecular props
===========================================================================================
Community detection via spectral embedding and node similarity scoring.

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

def build_community_graph(n_per=8, n_communities=3, p_in=0.6, p_out=0.05, seed=42):
    """Stochastic block model graph."""
    rng = np.random.default_rng(seed)
    N = n_per * n_communities
    A = np.zeros((N, N))
    labels = np.repeat(np.arange(n_communities), n_per)
    for i in range(N):
        for j in range(i+1, N):
            p = p_in if labels[i] == labels[j] else p_out
            if rng.random() < p:
                A[i, j] = A[j, i] = 1
    return A, labels

def spectral_embedding(A, k=2):
    """Laplacian eigenmaps for k-dim node embedding."""
    D = np.diag(A.sum(axis=1))
    L = D - A
    eigvals, eigvecs = np.linalg.eigh(L)
    # Skip trivial eigenvector (lambda=0), take next k
    return eigvecs[:, 1:k+1]

def gcn_layer(A, X, W):
    A_hat = A + np.eye(len(A))
    D_inv_sqrt = np.diag(1/np.sqrt(A_hat.sum(1)))
    return np.tanh((D_inv_sqrt @ A_hat @ D_inv_sqrt) @ X @ W)

def demo():
    print("=== GNN Applications ===")
    A, labels = build_community_graph(n_per=8, n_communities=3)
    N = len(A)
    print(f"  Graph: {N} nodes, {int(A.sum()//2)} edges, {len(np.unique(labels))} communities")

    Z = spectral_embedding(A, k=2)
    print(f"  Spectral embedding shape: {Z.shape}")

    # Simple k-means on embedding
    from numpy.linalg import norm
    K = 3
    np.random.seed(0)
    centroids = Z[np.random.choice(N, K, replace=False)]
    for _ in range(20):
        dists = np.array([[norm(Z[n] - centroids[k]) for k in range(K)] for n in range(N)])
        preds = dists.argmin(axis=1)
        centroids = np.array([Z[preds == k].mean(axis=0) if (preds==k).any() else centroids[k] for k in range(K)])
    # Accuracy (best permutation via majority label assignment)
    acc_best = 0
    from itertools import permutations
    for perm in permutations(range(K)):
        mapped = np.array([perm[p] for p in preds])
        acc_best = max(acc_best, (mapped == labels).mean())
    print(f"  Community detection accuracy (spectral+kmeans): {acc_best:.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    for k in range(K):
        mask = labels == k
        axes[0].scatter(Z[mask, 0], Z[mask, 1], label=f"C{k}", color=colors[k], s=40)
    axes[0].set_title("Spectral Embedding (true labels)"); axes[0].legend()
    axes[1].imshow(A, cmap="Blues"); axes[1].set_title("Adjacency Matrix (Block Structure)")
    plt.tight_layout(); plt.savefig(OUTPUT / "gnn_applications.png"); plt.close()
    print("  Saved gnn_applications.png")

def demo_link_prediction():
    """Link prediction via dot product of spectral embeddings."""
    print("\n=== Link Prediction ===")
    A, labels = build_community_graph(n_per=8, n_communities=3)
    Z = spectral_embedding(A, k=4)
    # Score all non-edges by dot product similarity
    N = len(A)
    scores = Z @ Z.T
    existing_edges = set(zip(*np.where(A > 0)))
    preds = []
    for i in range(N):
        for j in range(i+1, N):
            s = scores[i, j]
            label = 1 if (i, j) in existing_edges else 0
            preds.append((s, label))
    preds.sort(key=lambda x: -x[0])
    # AUC via rank correlation
    top50 = preds[:50]
    prec = np.mean([lbl for _, lbl in top50])
    print(f"  Precision@50 (link prediction): {prec:.3f}")
    print(f"  Total edges: {len(existing_edges)//2}  Non-edges: {N*(N-1)//2 - len(existing_edges)//2}")


def demo_graph_classification():
    """Classify graphs by community structure using GCN mean-pooling."""
    print("\n=== Graph Classification ===")
    np.random.seed(42)
    results = []
    for label, p_in, p_out in [(0, 0.6, 0.05), (1, 0.2, 0.2)]:
        A, _ = build_community_graph(n_per=4, n_communities=2, p_in=p_in, p_out=p_out)
        N, F = len(A), 3
        X = np.random.randn(N, F)
        W = np.random.randn(F, 2) * 0.3
        H = gcn_layer(A, X, W)
        graph_repr = H.mean(axis=0)
        results.append((label, graph_repr))
    # Check embeddings differ
    d = np.linalg.norm(results[0][1] - results[1][1])
    print(f"  Embedding distance between class-0 and class-1 graph: {d:.4f}")
    print(f"  (Larger distance = more discriminative GCN representation)")


if __name__ == "__main__":
    demo()
    demo_link_prediction()
    demo_graph_classification()
