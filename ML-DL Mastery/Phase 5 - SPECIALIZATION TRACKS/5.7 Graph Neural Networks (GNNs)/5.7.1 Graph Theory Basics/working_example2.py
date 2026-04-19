"""
Working Example 2: Graph Theory Basics — adjacency, BFS/DFS, degree, clustering
=================================================================================
Builds a synthetic graph and computes standard graph properties.

Run:  python working_example2.py
"""
from pathlib import Path
from collections import deque
try:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit("pip install numpy matplotlib")

OUTPUT = Path(__file__).parent / "output"
OUTPUT.mkdir(exist_ok=True)

def make_graph(n=10, p=0.35, seed=0):
    """Erdős–Rényi random graph as adjacency matrix."""
    rng = np.random.default_rng(seed)
    A = (rng.random((n, n)) < p).astype(float)
    A = np.tril(A, -1); A = A + A.T  # undirected, no self-loops
    return A

def bfs(A, start=0):
    visited = [False] * len(A); order = []; queue = deque([start])
    visited[start] = True
    while queue:
        v = queue.popleft(); order.append(v)
        for u in np.where(A[v] > 0)[0]:
            if not visited[u]:
                visited[u] = True; queue.append(u)
    return order

def clustering_coeff(A, v):
    """Local clustering coefficient for node v."""
    nbrs = np.where(A[v] > 0)[0]; k = len(nbrs)
    if k < 2: return 0.0
    edges = sum(A[i, j] for i in nbrs for j in nbrs if j > i)
    return 2 * edges / (k * (k-1))

def demo():
    print("=== Graph Theory Basics ===")
    A = make_graph(n=10, p=0.35)
    print(f"  Nodes: {len(A)}  Edges: {int(A.sum()//2)}")
    degree = A.sum(axis=1)
    print(f"  Degree: {degree.astype(int).tolist()}")
    print(f"  Avg degree: {degree.mean():.2f}")

    bfs_order = bfs(A, start=0)
    print(f"  BFS from 0: {bfs_order}")

    cc = [clustering_coeff(A, v) for v in range(len(A))]
    print(f"  Avg clustering coeff: {np.mean(cc):.3f}")

    # Laplacian
    D = np.diag(degree); L = D - A
    eigvals = np.linalg.eigvalsh(L)
    print(f"  Algebraic connectivity (lambda2): {eigvals[1]:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    axes[0].imshow(A, cmap="Blues"); axes[0].set_title("Adjacency Matrix")
    axes[1].bar(range(len(A)), degree); axes[1].set_title("Degree Distribution")
    plt.tight_layout(); plt.savefig(OUTPUT / "graph_theory.png"); plt.close()
    print("  Saved graph_theory.png")

if __name__ == "__main__":
    demo()
