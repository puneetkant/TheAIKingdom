"""
Working Example: GNN Tasks
Covers node classification, link prediction, graph classification,
and graph generation — with numpy implementations.
"""
import numpy as np
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_gnn_tasks")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def relu(x): return np.maximum(0, x)
def sigmoid(x): return 1 / (1 + np.exp(-np.clip(x, -30, 30)))
def softmax(z):
    e = np.exp(z - z.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


# ── Shared graph fixture ──────────────────────────────────────────────────────
def make_graph():
    N = 8; F_in = 4
    rng = np.random.default_rng(0)
    X   = rng.standard_normal((N, F_in))
    edges = [(0,1),(0,2),(1,2),(1,3),(2,4),(3,4),(4,5),(5,6),(5,7),(6,7)]
    A = np.zeros((N, N))
    for u, v in edges:
        A[u,v] = A[v,u] = 1.0
    return N, F_in, X, A, edges


def gcn_layer(X, A, W):
    """One GCN layer with self-loops."""
    A_hat  = A + np.eye(len(A))
    D_inv  = np.diag(1.0 / (A_hat.sum(axis=1)**0.5 + 1e-9))
    A_norm = D_inv @ A_hat @ D_inv
    return relu(A_norm @ X @ W)


# ── 1. Node classification ────────────────────────────────────────────────────
def node_classification():
    print("=== Node Classification ===")
    print("  Predict class label for each node")
    print("  Semi-supervised: use both labelled + unlabelled nodes")
    print()

    N, F_in, X, A, _ = make_graph()
    rng  = np.random.default_rng(1)
    W1   = rng.normal(0, 0.1, (F_in, 8))
    W2   = rng.normal(0, 0.1, (8, 3))   # 3 classes

    # 2-layer GCN
    H1   = gcn_layer(X, A, W1)
    A_hat = A + np.eye(N)
    D_inv = np.diag(1.0 / (A_hat.sum(axis=1)**0.5 + 1e-9))
    A_norm = D_inv @ A_hat @ D_inv
    logits = A_norm @ H1 @ W2
    probs  = softmax(logits)
    preds  = probs.argmax(axis=1)

    # Labels for first 4 nodes; rest unlabelled
    labels = np.array([0, 1, 0, 2, -1, -1, -1, -1])
    mask   = labels >= 0
    acc    = (preds[mask] == labels[mask]).mean()

    print(f"  2-layer GCN: F_in={F_in} → 8 → 3 classes")
    print(f"  Predicted classes: {preds}")
    print(f"  True labels:       {labels}")
    print(f"  Train accuracy (4 labelled nodes): {acc:.2%}")
    print()
    print("  Key benchmark datasets:")
    datasets = [
        ("Cora",        "2708 nodes, 5429 edges, 7 classes (papers)"),
        ("CiteSeer",    "3327 nodes, 4732 edges, 6 classes"),
        ("PubMed",      "19717 nodes, 44338 edges, 3 classes"),
        ("ogbn-arxiv",  "169k nodes, 1.2M edges, 40 classes"),
        ("ogbn-products","2.4M nodes, 61.9M edges, 47 classes"),
    ]
    for d, desc in datasets:
        print(f"  {d:<16} {desc}")


# ── 2. Link prediction ────────────────────────────────────────────────────────
def link_prediction():
    print("\n=== Link Prediction ===")
    print("  Predict whether an edge exists between two nodes")
    print("  Applications: social network friend recommendation, KG completion")
    print()

    N, F_in, X, A, edges = make_graph()
    rng = np.random.default_rng(2)
    W1  = rng.normal(0, 0.1, (F_in, 6))

    H = gcn_layer(X, A, W1)

    # Compute edge scores (dot product of node embeddings)
    def edge_score(u, v): return sigmoid(H[u] @ H[v])

    # Positive examples: existing edges
    # Negative examples: non-existing edges (negative sampling)
    neg_edges = []
    all_edges_set = set(edges) | {(v,u) for u,v in edges}
    rng2 = np.random.default_rng(99)
    while len(neg_edges) < len(edges):
        u, v = rng2.integers(N), rng2.integers(N)
        if u != v and (u,v) not in all_edges_set:
            neg_edges.append((u,v))

    pos_scores = [edge_score(u, v) for u, v in edges]
    neg_scores = [edge_score(u, v) for u, v in neg_edges]

    print(f"  Positive edges: {len(edges)}  scores: {np.round(pos_scores,3)}")
    print(f"  Negative edges: {len(neg_edges)} scores: {np.round(neg_scores,3)}")

    # BCE loss
    eps = 1e-9
    loss_pos = -np.log(np.array(pos_scores) + eps).mean()
    loss_neg = -np.log(1 - np.array(neg_scores) + eps).mean()
    loss     = (loss_pos + loss_neg) / 2
    print(f"  BCE loss: {loss:.4f}")
    print()
    print("  Scoring functions:")
    print("    Dot product:   h_u^T h_v")
    print("    Bilinear:      h_u^T R h_v  (TransE/DistMult/RotatE for KGs)")
    print("    MLP:           MLP([h_u || h_v])")
    print("    Hadamard:      h_u ⊙ h_v → MLP")


# ── 3. Graph classification ───────────────────────────────────────────────────
def graph_classification():
    print("\n=== Graph Classification ===")
    print("  Predict a single label for an entire graph")
    print("  Requires READOUT (pooling) over all node embeddings")
    print()
    print("  Readout functions:")
    readouts = [
        ("Global mean",   "ĥ = (1/N) Σ h_v"),
        ("Global sum",    "ĥ = Σ h_v  (used in GIN; more expressive)"),
        ("Global max",    "ĥ = max_v h_v  (element-wise)"),
        ("Hierarchical",  "DiffPool, MinCutPool: learn to cluster nodes"),
        ("Set2Vec",       "LSTM-based attention over node set"),
        ("Sort pooling",  "Sort nodes by last GNN layer; take top-k"),
    ]
    for r, d in readouts:
        print(f"  {r:<16} {d}")
    print()

    # Simulate classification of 5 graphs
    rng = np.random.default_rng(3)
    n_graphs = 5; F_in = 4; F_out = 6; n_classes = 2

    W1 = rng.normal(0, 0.1, (F_in, F_out))
    W2 = rng.normal(0, 0.1, (F_out, n_classes))

    graph_preds = []
    for g in range(n_graphs):
        Ng  = rng.integers(4, 8)
        X   = rng.standard_normal((Ng, F_in))
        A   = (rng.uniform(0, 1, (Ng, Ng)) > 0.6).astype(float)
        np.fill_diagonal(A, 0); A = np.maximum(A, A.T)

        H   = gcn_layer(X, A, W1)
        h_g = H.mean(axis=0)     # global mean pooling
        log = h_g @ W2
        pred = log.argmax()
        graph_preds.append(pred)

    print(f"  Predicted classes for {n_graphs} graphs: {graph_preds}")
    print()
    print("  Benchmark datasets:")
    benches = [
        ("MUTAG",       "188 graphs, 2 classes (mutagenic compounds)"),
        ("PROTEINS",    "1113 graphs, 2 classes (protein function)"),
        ("IMDB-B",      "1000 graphs, 2 classes (movie collaboration)"),
        ("ogbg-molhiv", "41k graphs, 2 classes (HIV activity prediction)"),
        ("ogbg-molpcba","440k graphs, 128 classes (bioactivity)"),
    ]
    for d, desc in benches:
        print(f"  {d:<16} {desc}")


if __name__ == "__main__":
    node_classification()
    link_prediction()
    graph_classification()
