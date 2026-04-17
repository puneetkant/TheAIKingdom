"""
Working Example 2: GNN Tasks — node classification, link prediction, graph classification
===========================================================================================
Demonstrates all three main GNN tasks on synthetic graphs.

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

def gcn_layer(A, X, W, act=np.tanh):
    A_hat = A + np.eye(len(A))
    D_inv_sqrt = np.diag(1/np.sqrt(A_hat.sum(axis=1)))
    A_norm = D_inv_sqrt @ A_hat @ D_inv_sqrt
    return act(A_norm @ X @ W)

def softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

def demo_node_classification():
    """Two-layer GCN: nodes assigned labels 0/1 based on community."""
    np.random.seed(1)
    N = 8; F_in, F_h, F_out = 4, 3, 2
    A = np.array([
        [0,1,1,0,0,0,0,0],[1,0,1,0,0,0,0,0],[1,1,0,1,0,0,0,0],[0,0,1,0,0,0,0,0],
        [0,0,0,0,0,1,1,0],[0,0,0,0,1,0,1,0],[0,0,0,0,1,1,0,1],[0,0,0,0,0,0,1,0],
    ], float)
    X = np.random.randn(N, F_in)
    W1 = np.random.randn(F_in, F_h) * 0.3
    W2 = np.random.randn(F_h, F_out) * 0.3
    H = gcn_layer(A, gcn_layer(A, X, W1), W2)
    probs = softmax(H)
    preds = probs.argmax(axis=1)
    labels = np.array([0,0,0,0,1,1,1,1])
    acc = (preds == labels).mean()
    print(f"  [Node classification] Random init acc: {acc:.2f}")
    return preds, labels

def demo_link_prediction():
    """Inner product decoder on embeddings."""
    np.random.seed(2)
    N = 6; F = 4
    A = np.array([[0,1,1,0,0,0],[1,0,0,0,0,0],[1,0,0,1,0,0],[0,0,1,0,1,0],[0,0,0,1,0,1],[0,0,0,0,1,0]], float)
    Z = np.random.randn(N, F) * 0.3   # embeddings (normally GCN output)
    scores = Z @ Z.T                   # inner product decoder
    pred_edges = [(i, j) for i in range(N) for j in range(i+1, N)
                  if scores[i,j] > scores.mean()]
    true_edges = {(i, j) for i in range(N) for j in range(i+1, N) if A[i,j]}
    tp = len({e for e in pred_edges if e in true_edges})
    prec = tp / (len(pred_edges)+1e-8)
    print(f"  [Link prediction] Precision: {prec:.2f}  Predicted edges: {len(pred_edges)}")

def demo_graph_classification():
    """Sum readout then linear classification on random graphs."""
    np.random.seed(3)
    n_graphs, N, F = 20, 6, 4
    labels = np.random.randint(0, 2, n_graphs)
    correct = 0
    for y in labels:
        A = (np.random.rand(N, N) < 0.3).astype(float)
        A = np.tril(A, -1); A = A + A.T
        X = np.random.randn(N, F)
        W = np.random.randn(F, 2) * 0.3
        H = gcn_layer(A, X, W)
        g = H.sum(axis=0)              # sum readout
        pred = (g[0] > g[1]).astype(int)
        correct += (pred == y)
    print(f"  [Graph classification] Acc: {correct/n_graphs:.2f}")

def main():
    print("=== GNN Tasks Demo ===")
    preds, labels = demo_node_classification()
    demo_link_prediction()
    demo_graph_classification()

    fig, ax = plt.subplots(figsize=(4, 2.5))
    ax.bar(["Match","Mismatch"], [(preds==labels).sum(), (preds!=labels).sum()])
    ax.set_title("Node Prediction Match"); plt.tight_layout()
    plt.savefig(OUTPUT / "gnn_tasks.png"); plt.close()
    print("  Saved gnn_tasks.png")

if __name__ == "__main__":
    main()
