"""
Working Example: Advanced GNN Topics
Covers over-smoothing, scalability, heterogeneous GNNs,
temporal GNNs, and graph transformers.
"""
import numpy as np
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_advanced_gnn")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def relu(x): return np.maximum(0, x)


# ── 1. Over-smoothing ─────────────────────────────────────────────────────────
def over_smoothing():
    print("=== Over-Smoothing in GNNs ===")
    print()
    print("  Problem: after many layers, node representations converge to")
    print("           the same vector → lose discriminative power")
    print()
    print("  Diagnosis: measure feature diversity = ||H^l||_F / N")
    print()

    # Simulate power iteration of A_norm
    N = 6
    edges = [(0,1),(0,2),(1,2),(1,3),(2,4),(3,4)]
    A = np.zeros((N, N))
    for u, v in edges:
        A[u,v] = A[v,u] = 1.0
    A_hat  = A + np.eye(N)
    D_inv  = np.diag(1.0 / (A_hat.sum(axis=1)**0.5 + 1e-9))
    A_norm = D_inv @ A_hat @ D_inv

    rng = np.random.default_rng(0)
    H = rng.standard_normal((N, 4))

    print(f"  Layer  Frobenius norm  Std(features)  Max pairwise dist")
    for l in range(10):
        fn   = np.linalg.norm(H, 'fro')
        std  = H.std()
        dmax = max(np.linalg.norm(H[i]-H[j]) for i in range(N) for j in range(i+1, N))
        print(f"  {l:>5}  {fn:>13.4f}  {std:>13.4f}  {dmax:>17.4f}")
        H    = A_norm @ H   # propagate without weight

    print()
    print("  Solutions to over-smoothing:")
    solutions = [
        ("DropEdge",        "Randomly drop edges during training (regularisation)"),
        ("PairNorm",        "Normalise to prevent all nodes collapsing"),
        ("Jumping Knowledge","JK-Net: aggregate representations from all layers"),
        ("APPNP",           "Personalised PageRank propagation; separate feature + prop"),
        ("GraphSAINT",      "Graph sub-sampling; avoid deep propagation"),
        ("Residual / GCNII","Residual connections to initial features"),
    ]
    for s, d in solutions:
        print(f"  {s:<18} {d}")


# ── 2. Scalability ────────────────────────────────────────────────────────────
def scalability():
    print("\n=== GNN Scalability ===")
    print()
    print("  Challenge: full-batch GCN requires entire graph in memory")
    print("  Solution: mini-batch training with neighbourhood sampling")
    print()
    print("  Scalability methods:")
    methods = [
        ("GraphSAGE",   "Sample fixed-size neighbourhoods; inductive"),
        ("PinSage",     "Pinterest; random walks to define neighbours"),
        ("Cluster-GCN", "Partition graph; train on subgraphs"),
        ("GraphSAINT",  "Node/edge/random-walk graph sampling"),
        ("LADIES",      "Layer-dependent importance sampling"),
        ("GAS",         "Historical embeddings to avoid neighbour explosion"),
        ("PyG (mini)", "Mini-batch via NeighborLoader in PyG 2.0"),
    ]
    print(f"  {'Method':<14} {'Description'}")
    for m, d in methods:
        print(f"  {m:<14} {d}")
    print()
    print("  Complexity comparison (per layer):")
    print("    Full-batch GCN:  O(|E| + N·d·F)  — limited to ~10M nodes")
    print("    Mini-batch:      O(k^L · F)  per sample, k=fan-out, L=layers")


# ── 3. Heterogeneous GNNs ─────────────────────────────────────────────────────
def heterogeneous_gnns():
    print("\n=== Heterogeneous GNNs ===")
    print()
    print("  Heterogeneous graph: multiple node and edge types")
    print("  Example (academic graph):")
    print("    Node types:  Paper, Author, Venue")
    print("    Edge types:  (Paper, cites, Paper), (Author, writes, Paper),")
    print("                 (Paper, published_at, Venue)")
    print()
    print("  Approaches:")
    approaches = [
        ("HAN",       "Heterogeneous Attention Network; meta-path-based"),
        ("HGT",       "Heterogeneous Graph Transformer; type-specific W, Q, K, V"),
        ("R-GCN",     "Relational GCN; separate W per relation type"),
        ("SeHGNN",    "Simple Heterogeneous GNN; scalable; feature propagation only"),
        ("GraphSAGE", "Multiple node types with type-specific aggregators"),
    ]
    for a, d in approaches:
        print(f"  {a:<12} {d}")
    print()
    print("  R-GCN update:")
    print("    h_v^{l+1} = σ(Σ_{r∈R} (1/c_{v,r}) Σ_{u∈N_r(v)} W_r^l h_u^l + W_0^l h_v^l)")

    # Toy R-GCN
    rng = np.random.default_rng(0)
    N = 4; F = 3; n_rels = 2
    X = rng.standard_normal((N, F))
    # Adjacency per relation
    A_r = [np.array([[0,1,1,0],[1,0,0,0],[1,0,0,1],[0,0,1,0]], dtype=float),
           np.array([[0,0,0,1],[0,0,1,0],[0,1,0,0],[1,0,0,0]], dtype=float)]
    W_r = [rng.normal(0,0.1,(F,F)) for _ in range(n_rels)]
    W_0 = rng.normal(0,0.1,(F,F))

    H_out = X @ W_0
    for r in range(n_rels):
        deg = A_r[r].sum(axis=1, keepdims=True) + 1e-9
        H_out += (A_r[r] / deg) @ X @ W_r[r]
    H_out = relu(H_out)
    print(f"\n  R-GCN (2 relations): {X.shape} → {H_out.shape}")


# ── 4. Temporal GNNs ─────────────────────────────────────────────────────────
def temporal_gnns():
    print("\n=== Temporal GNNs ===")
    print()
    print("  Two paradigms:")
    print()
    print("  Discrete-time (snapshot-based):")
    print("    Series of static graph snapshots G_1, G_2, ..., G_T")
    print("    Apply GCN to each snapshot; LSTM/Transformer across time")
    print("    Models: GCRN, EvolveGCN, ST-GCN (skeleton action)")
    print()
    print("  Continuous-time (event-based):")
    print("    Each edge (u, v, t) is a timestamped event")
    print("    Maintain node memory updated at each interaction")
    print("    Models: JODIE, TGN (Temporal Graph Networks), CAWN")
    print()
    print("  TGN (Rossi 2020):")
    print("    Node memory m_v:  persistent state updated by events")
    print("    Message function:  msg = CONCAT(m_v, m_u, Δt, e_{uv})")
    print("    Memory update:     m_v ← GRU(m_v, agg_msgs)")
    print("    Embedding:         h_v = MLP(m_v, agg_neighbours)")
    print()
    print("  Applications:")
    print("    Traffic forecasting, social network evolution, financial fraud")


if __name__ == "__main__":
    over_smoothing()
    scalability()
    heterogeneous_gnns()
    temporal_gnns()
