"""
Working Example: Deep Learning Recommenders
Covers neural CF, wide & deep networks, two-tower models,
sequential recommenders, and large-scale retrieval.
"""
import numpy as np
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_dl_rec")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def softmax(z):
    e = np.exp(z - z.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


# ── 1. Embedding-based neural CF ─────────────────────────────────────────────
def neural_cf():
    print("=== Neural Collaborative Filtering (NCF) ===")
    print("  He et al. (2017) — replaces dot product with MLP")
    print()
    print("  Architecture:")
    print("    User ID → embedding e_u  (size K)")
    print("    Item ID → embedding e_i  (size K)")
    print("    [e_u || e_i] → MLP layers → output score")
    print()
    print("  GMF (Generalized MF): score = w · (e_u ⊙ e_i)")
    print("  NeuMF: concatenate GMF + MLP pathways → final score")
    print()

    # Tiny NCF demo
    rng = np.random.default_rng(0)
    n_users, n_items, K = 10, 15, 4
    U_emb = rng.normal(0, 0.1, (n_users, K))
    I_emb = rng.normal(0, 0.1, (n_items, K))
    W1    = rng.normal(0, 0.1, (2*K, 8))
    W2    = rng.normal(0, 0.1, (8, 1))

    def predict(u, i):
        x = np.concatenate([U_emb[u], I_emb[i]])
        h = np.maximum(0, x @ W1)    # ReLU
        return 1 / (1 + np.exp(-(h @ W2).squeeze()))   # sigmoid

    # Rating matrix (binary)
    R_true = (rng.uniform(0, 1, (n_users, n_items)) > 0.7).astype(float)
    # BCE loss over observed
    R_hat  = np.array([[predict(u, i) for i in range(n_items)] for u in range(n_users)])
    bce    = -(R_true * np.log(R_hat + 1e-9) + (1-R_true) * np.log(1-R_hat + 1e-9)).mean()
    print(f"  Toy NCF: {n_users} users × {n_items} items  K={K}")
    print(f"  Binary predictions range: [{R_hat.min():.3f}, {R_hat.max():.3f}]")
    print(f"  BCE loss: {bce:.4f}")


# ── 2. Wide & Deep ────────────────────────────────────────────────────────────
def wide_and_deep():
    print("\n=== Wide & Deep (Google, 2016) ===")
    print()
    print("  Wide component (memorisation):")
    print("    Logistic regression on raw features + cross-product features")
    print("    Memorises frequent patterns (e.g. user×item pairs seen at training)")
    print()
    print("  Deep component (generalisation):")
    print("    Embedding → FC layers → logit")
    print("    Generalises to unseen feature combinations")
    print()
    print("  Combined: output = σ(w_wide^T [x, φ(x)] + w_deep^T a^{(lf)} + bias)")
    print()
    print("  Variants:")
    variants = [
        ("Wide & Deep",      "Google Play; app recommendations"),
        ("DeepFM",           "Replace wide part with FM; no manual cross-feature"),
        ("xDeepFM",          "Explicit + implicit feature interactions via CIN"),
        ("DCN (v2)",         "Cross network; explicit polynomial feature interactions"),
        ("AutoInt",          "Multi-head attention for feature interaction"),
        ("DLRM",             "Meta; production-scale; sparse+dense"),
    ]
    for m, d in variants:
        print(f"  {m:<16} {d}")


# ── 3. Two-tower retrieval model ─────────────────────────────────────────────
def two_tower():
    print("\n=== Two-Tower Retrieval Model ===")
    print()
    print("  Query tower:  user features → user embedding q ∈ R^D")
    print("  Item tower:   item features → item embedding d ∈ R^D")
    print("  Score:        sim(q, d) = q^T d  (or cosine similarity)")
    print()
    print("  Training objective: in-batch negatives softmax")
    print("    L = -log [ exp(q_i^T d_i / τ) / Σ_j exp(q_i^T d_j / τ) ]")
    print("    τ = temperature (typical: 0.05-0.1)")
    print()

    rng = np.random.default_rng(0)
    N = 8; D = 4; tau = 0.1

    q = rng.normal(0, 1, (N, D)); q /= np.linalg.norm(q, axis=1, keepdims=True)
    d = q + rng.normal(0, 0.5, (N, D))  # item embeddings close to query
    d /= np.linalg.norm(d, axis=1, keepdims=True)

    logits = q @ d.T / tau  # (N, N)
    labels = np.eye(N)
    loss   = -(np.log(softmax(logits) + 1e-9) * labels).sum() / N
    print(f"  In-batch softmax loss (N={N}, D={D}, τ={tau}): {loss:.4f}")
    print()
    print("  Deployment:")
    print("    Index all item embeddings in ANN (Approximate Nearest Neighbour) index")
    print("    At query time: compute query embedding → search ANN index")
    print("    ANN options: FAISS, ScaNN, Annoy, HNSW")
    print()
    print("  ANN comparison:")
    anns = [
        ("FAISS",   "Meta; GPU; IVF+PQ; billion-scale"),
        ("ScaNN",   "Google; streaming distance computation"),
        ("HNSW",    "Hierarchical graphs; high recall; moderate speed"),
        ("Annoy",   "Spotify; forest of random trees; good for static indices"),
    ]
    for a, d_ in anns:
        print(f"  {a:<8} {d_}")


# ── 4. Sequential recommenders ────────────────────────────────────────────────
def sequential_recommenders():
    print("\n=== Sequential Recommenders ===")
    print("  Model user history as a sequence → predict next item")
    print()
    models = [
        ("GRU4Rec",    2016, "GRU on item sequences; session-based"),
        ("SASRec",     2018, "Self-attention; causal masking; strong baseline"),
        ("BERT4Rec",   2019, "Masked item prediction; bidirectional attention"),
        ("S3-Rec",     2020, "Self-supervised pre-training for sequential rec"),
        ("RecFormer",  2022, "Language + sequential rec; item text tokens"),
        ("HSTU",       2024, "Meta; hierarchical sequential transduction"),
    ]
    print(f"  {'Model':<14} {'Year'} {'Notes'}")
    print(f"  {'─'*14} {'─'*4} {'─'*45}")
    for m, y, d in models:
        print(f"  {m:<14} {y}  {d}")
    print()

    # SASRec-like self-attention on item sequence
    rng  = np.random.default_rng(0)
    L    = 6; D = 4; n_items = 20
    item_seq = rng.integers(n_items, size=L)
    I_emb    = rng.normal(0, 0.1, (n_items, D))
    x        = I_emb[item_seq]   # (L, D)

    # Self-attention
    Wq, Wk, Wv = [rng.normal(0, 0.1, (D, D)) for _ in range(3)]
    Q, K, V = x @ Wq, x @ Wk, x @ Wv
    att_raw = Q @ K.T / np.sqrt(D)

    # Causal mask
    mask = np.triu(np.full((L, L), -np.inf), k=1)
    att  = softmax(att_raw + mask)
    out  = att @ V   # (L, D)
    # Last position → score all items
    scores = out[-1] @ I_emb.T   # (n_items,)

    print(f"  SASRec self-attention: seq_len={L}, D={D}")
    print(f"  Top-3 next item predictions: {scores.argsort()[::-1][:3]}")


if __name__ == "__main__":
    neural_cf()
    wide_and_deep()
    two_tower()
    sequential_recommenders()
