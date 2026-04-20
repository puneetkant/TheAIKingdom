"""
Working Example 2: GNN Tools — PyG / DGL patterns simulated with numpy
========================================================================
Demonstrates data object format, message passing pseudocode, and mini-batch.

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

class GraphData:
    """Minimal graph data container (mirrors PyG Data object)."""
    def __init__(self, x, edge_index, y=None):
        self.x = np.array(x, float)           # (N, F) node features
        self.edge_index = np.array(edge_index, int)  # (2, E) COO format
        self.y = y                             # node/graph labels
        self.N = len(x)

    def adjacency(self):
        A = np.zeros((self.N, self.N))
        A[self.edge_index[0], self.edge_index[1]] = 1
        return A

class MessagePassingLayer:
    """Minimal message passing layer."""
    def __init__(self, W):
        self.W = W

    def forward(self, data):
        A = data.adjacency() + np.eye(data.N)
        D_inv = np.diag(1/A.sum(1))
        return np.tanh((D_inv @ A) @ data.x @ self.W)

def mini_batch_demo(graphs, layer):
    """Process a list of graphs and return graph-level predictions."""
    outputs = []
    for g in graphs:
        H = layer.forward(g)
        outputs.append(H.mean(axis=0))   # mean readout
    return np.array(outputs)

def demo():
    np.random.seed(0)
    # Build 4 small graphs
    graphs = []
    for i in range(4):
        N = np.random.randint(4, 8)
        A = (np.random.rand(N, N) < 0.4).astype(int)
        A = np.tril(A, -1); A = A + A.T
        edges = np.array(np.where(A > 0))
        x = np.random.randn(N, 3)
        y = i % 2
        graphs.append(GraphData(x, edges, y))

    F_in, F_out = 3, 4
    W = np.random.randn(F_in, F_out) * 0.3
    layer = MessagePassingLayer(W)

    print("=== GNN Tools Demo ===")
    for i, g in enumerate(graphs):
        print(f"  Graph {i}: {g.N} nodes  edges={g.edge_index.shape[1]}  label={g.y}")

    batch_out = mini_batch_demo(graphs, layer)
    print(f"\n  Mini-batch output shape: {batch_out.shape}")
    print(f"  Readout vectors:\n{batch_out.round(3)}")

    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    for i, g in enumerate(graphs):
        axes[i].imshow(g.adjacency(), cmap="Blues"); axes[i].set_title(f"Graph {i} (y={g.y})")
        axes[i].set_xticks([]); axes[i].set_yticks([])
    plt.tight_layout(); plt.savefig(OUTPUT / "gnn_tools.png"); plt.close()
    print("  Saved gnn_tools.png")

def demo_edge_features():
    """Extend message passing to incorporate edge weights."""
    print("\n=== Edge-Feature Message Passing ===")
    np.random.seed(5)
    N, F_in, F_out = 5, 3, 4
    A = np.array([
        [0,1,1,0,0],[1,0,0,1,0],[1,0,0,1,1],[0,1,1,0,1],[0,0,1,1,0]
    ], float)
    edge_weights = A * np.random.uniform(0.5, 2.0, A.shape)  # weighted adjacency
    edge_weights = (edge_weights + edge_weights.T) / 2
    X = np.random.randn(N, F_in)
    W = np.random.randn(F_in, F_out) * 0.3

    # Standard GCN
    H_standard = MessagePassingLayer(W).forward(GraphData(X, np.array(np.where(A))))

    # Weighted GCN: use edge_weights instead of binary A
    A_hat = edge_weights + np.eye(N)
    D_inv = np.diag(1 / A_hat.sum(1))
    H_weighted = np.tanh((D_inv @ A_hat) @ X @ W)

    print(f"  Standard GCN output norm: {np.linalg.norm(H_standard):.4f}")
    print(f"  Weighted GCN output norm: {np.linalg.norm(H_weighted):.4f}")


def demo_graph_batching():
    """Show how multiple graphs can be batched via block-diagonal adjacency."""
    print("\n=== Graph Batching ===")
    np.random.seed(8)
    graph_sizes = [4, 5, 3]
    F_in, F_out = 3, 2
    W = np.random.randn(F_in, F_out) * 0.3
    layer = MessagePassingLayer(W)

    # Batched processing
    batch_repr = []
    total_nodes = 0
    for sz in graph_sizes:
        A = (np.random.rand(sz, sz) < 0.4).astype(float)
        A = np.tril(A, -1); A = A + A.T
        edges = np.array(np.where(A > 0))
        g = GraphData(np.random.randn(sz, F_in), edges)
        H = layer.forward(g)
        batch_repr.append(H.mean(axis=0))   # mean readout
        total_nodes += sz

    batch_out = np.array(batch_repr)
    print(f"  Graphs batched: {len(graph_sizes)}  Total nodes: {total_nodes}")
    print(f"  Batch output shape: {batch_out.shape}")
    print(f"  Readout norms: {np.linalg.norm(batch_out, axis=1).round(3)}")


if __name__ == "__main__":
    demo()
    demo_edge_features()
    demo_graph_batching()
