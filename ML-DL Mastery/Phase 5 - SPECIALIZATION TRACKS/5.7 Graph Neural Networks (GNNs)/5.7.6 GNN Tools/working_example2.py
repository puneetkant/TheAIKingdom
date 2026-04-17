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

if __name__ == "__main__":
    demo()
