"""
Working Example: GNN Architectures
Covers GCN, GraphSAGE, GAT, GIN, message passing framework,
and common aggregation schemes — all in numpy.
"""
import numpy as np
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_gnn_arch")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def relu(x): return np.maximum(0, x)

def softmax(z):
    e = np.exp(z - z.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


# -- Shared graph fixture ------------------------------------------------------
def make_graph():
    N = 6; F_in = 4
    rng = np.random.default_rng(0)
    X   = rng.standard_normal((N, F_in))
    edges = [(0,1),(0,2),(1,2),(1,3),(2,4),(3,4),(4,5)]
    A = np.zeros((N, N))
    for u, v in edges:
        A[u,v] = A[v,u] = 1.0
    return N, F_in, X, A, edges


# -- 1. Message Passing Neural Network (MPNN) framework ------------------------
def mpnn_framework():
    print("=== Message Passing Neural Network Framework ===")
    print("  Gilmer et al. (2017)")
    print()
    print("  General update rule for node v at layer k:")
    print("    m_v^{k+1} = AGG_{u in N(v)} MSG(h_v^k, h_u^k, e_{uv})")
    print("    h_v^{k+1} = UPDATE(h_v^k, m_v^{k+1})")
    print()
    print("  MSG  = edge-wise message function")
    print("  AGG  = aggregation (sum, mean, max, attention)")
    print("  UPDATE = combination function (MLP, GRU, ...)")
    print()
    print("  Different choices -> different GNN architectures:")
    gnn_table = [
        ("GCN",       "mean (normalised sum)", "W·h",   "Kipf & Welling 2017"),
        ("GraphSAGE", "mean/max/LSTM",          "concat", "Hamilton 2017"),
        ("GAT",       "attention-weighted sum",  "W·h",   "Veličković 2018"),
        ("GIN",       "sum + MLP",              "MLP",   "Xu 2019"),
        ("MPNN",      "general function",        "GRU",   "Gilmer 2017"),
    ]
    print(f"  {'Model':<12} {'Aggregation':<24} {'Update':<10} {'Ref'}")
    print(f"  {'-'*12} {'-'*24} {'-'*10} {'-'*20}")
    for m, agg, upd, ref in gnn_table:
        print(f"  {m:<12} {agg:<24} {upd:<10} {ref}")


# -- 2. GCN (Graph Convolutional Network) -------------------------------------
def gcn_demo():
    print("\n=== GCN (Kipf & Welling, 2017) ===")
    print("  H^{l+1} = sigma(Ã H^l W^l)")
    print("  Ã = D^{-1/2}(A+I)D^{-1/2}  (normalised adjacency with self-loops)")
    print()

    N, F_in, X, A, _ = make_graph()
    F_out = 8
    rng   = np.random.default_rng(1)
    W     = rng.normal(0, 0.1, (F_in, F_out))

    A_hat  = A + np.eye(N)                         # add self-loops
    D_hat  = np.diag(A_hat.sum(axis=1))
    D_inv  = np.diag(1.0 / (np.sqrt(D_hat.diagonal()) + 1e-9))
    A_norm = D_inv @ A_hat @ D_inv                  # Ã

    H1 = relu(A_norm @ X @ W)
    print(f"  Input: X={X.shape}  W={W.shape}")
    print(f"  Output after 1 GCN layer: {H1.shape}")
    print(f"  Node 0 features: {H1[0].round(4)}")

    # 2-layer GCN
    W2 = rng.normal(0, 0.1, (F_out, 3))
    H2 = softmax(A_norm @ H1 @ W2)
    print(f"  After 2nd layer (3 classes): {H2.shape}")
    print(f"  Predicted class per node: {H2.argmax(axis=1)}")


# -- 3. GraphSAGE -------------------------------------------------------------
def graphsage_demo():
    print("\n=== GraphSAGE (Hamilton et al., 2017) ===")
    print("  Inductive: sample neighbours then aggregate")
    print("  h_v^l = sigma(W · CONCAT(h_v^{l-1}, AGG_{u in N_S(v)} h_u^{l-1}))")
    print()

    N, F_in, X, A, _ = make_graph()
    rng = np.random.default_rng(2)
    F_out = 6
    W     = rng.normal(0, 0.1, (2*F_in, F_out))

    def mean_agg(X, A):
        H_out = np.zeros((N, F_out))
        for v in range(N):
            nbrs = np.where(A[v])[0]
            agg  = X[nbrs].mean(axis=0) if len(nbrs) else np.zeros(F_in)
            combined = np.concatenate([X[v], agg])
            H_out[v] = relu(combined @ W)
        return H_out

    H = mean_agg(X, A)
    print(f"  GraphSAGE (mean) output: {H.shape}")
    print(f"  Node 0: {H[0].round(4)}")

    print()
    print("  Aggregation variants:")
    print("    MEAN:    average of neighbour features")
    print("    MAX:     element-wise max of neighbour features")
    print("    LSTM:    sequence over randomly-ordered neighbours")
    print("    GCN:     same as mean but without concatenating self")
    print()
    print("  Mini-batch sampling:")
    print("    K-hop neighbourhood explosion: 10^K nodes at K hops")
    print("    GraphSAGE samples S neighbours per hop (S_1=25, S_2=10)")


# -- 4. GAT (Graph Attention Network) -----------------------------------------
def gat_demo():
    print("\n=== GAT (Veličković et al., 2018) ===")
    print("  Attention weights between connected nodes:")
    print("    e_{vw} = LeakyReLU(a^T [W h_v || W h_w])")
    print("    alpha_{vw} = softmax_w(e_{vw})")
    print("    h_v' = sigma(Sigma_{w in N(v)} alpha_{vw} · W h_w)")
    print()

    N, F_in, X, A, _ = make_graph()
    rng    = np.random.default_rng(3)
    F_out  = 4; n_heads = 2

    def leaky_relu(x, alpha=0.2): return np.where(x >= 0, x, alpha*x)

    W = rng.normal(0, 0.1, (F_in, F_out))
    a = rng.normal(0, 0.1, 2*F_out)

    XW = X @ W  # (N, F_out)

    # Compute attention for single head
    attn_logits = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if A[i,j] > 0 or i == j:
                concat = np.concatenate([XW[i], XW[j]])
                attn_logits[i,j] = leaky_relu(a @ concat)

    mask = (A + np.eye(N)) == 0
    attn_logits[mask] = -1e9
    alpha = softmax(attn_logits)

    H_out = relu(alpha @ XW)
    print(f"  GAT output (1 head): {H_out.shape}")
    print(f"  Attention weights for node 0: {alpha[0].round(4)}")
    print()
    print(f"  Multi-head ({n_heads} heads) -> concatenate or average outputs")
    print(f"  GATv2: more expressive attention;  GraphTransformer: full attention")


# -- 5. GIN (Graph Isomorphism Network) ---------------------------------------
def gin_demo():
    print("\n=== GIN (Xu et al., 2019) ===")
    print("  Maximally expressive: as powerful as 1-WL isomorphism test")
    print("  h_v^{k} = MLP^k ((1+epsilon)·h_v^{k-1} + Sigma_{uinN(v)} h_u^{k-1})")
    print()

    N, F_in, X, A, _ = make_graph()
    rng = np.random.default_rng(4)
    eps = 0.0   # learnable; 0 is common initialisation

    W1 = rng.normal(0, 0.1, (F_in, 8))
    W2 = rng.normal(0, 0.1, (8, 4))

    H = X.copy()
    for v in range(N):
        nbrs = np.where(A[v])[0]
        agg  = H[nbrs].sum(axis=0) if len(nbrs) else np.zeros(F_in)
        h    = (1+eps)*H[v] + agg
        h    = relu(h @ W1)
        h    = relu(h @ W2)

    print(f"  GIN (1 layer, 2-MLP): input {F_in}->8->4")
    print(f"  Key insight: SUM aggregation > MEAN/MAX for graph isomorphism")
    print(f"  GIN≡1-WL test; can't distinguish some non-isomorphic graphs")
    print()
    print("  Beyond 1-WL: k-GNN, NGNN, OSAN, DE-GNN (higher-order)")


if __name__ == "__main__":
    mpnn_framework()
    gcn_demo()
    graphsage_demo()
    gat_demo()
    gin_demo()
