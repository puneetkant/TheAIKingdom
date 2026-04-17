"""
Working Example: Hybrid Recommender Approaches
Covers combining content-based and collaborative filtering,
knowledge graphs, contextual bandits, and session-based methods.
"""
import numpy as np
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_hybrid_rec")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── 1. Hybrid combination strategies ─────────────────────────────────────────
def hybrid_strategies():
    print("=== Hybrid Recommender Approaches ===")
    print()
    print("  Why hybrid?")
    print("    Content-based: item cold start OK; user cold start bad")
    print("    Collaborative:  overcomes content limitations; sparsity problem")
    print("    Hybrid:         best of both worlds")
    print()
    print("  Combination strategies:")
    strategies = [
        ("Weighted",      "s = α·s_CF + (1-α)·s_CB  (α tuned on validation)"),
        ("Switching",     "Use CB for cold users; switch to CF after enough data"),
        ("Mixed",         "Display both lists side by side"),
        ("Feature combo", "Item features as input to CF model"),
        ("Cascade",       "CF generates candidates; CB re-ranks"),
        ("Feature augment","CB generates user profile as feature for CF"),
        ("Unified model", "Single model jointly learns CF + CB signals"),
    ]
    for s, d in strategies:
        print(f"  {s:<16} {d}")

    print()
    print("  Modern production systems typically use cascade (funnel):")
    print("    Retrieval (ANN) → Filtering → Ranking → Re-ranking → Presentation")


# ── 2. LightFM hybrid model ───────────────────────────────────────────────────
def lightfm_hybrid():
    print("\n=== LightFM: Hybrid Embedding Model ===")
    print("  Kula (2015)")
    print()
    print("  User and item representations:")
    print("    User embedding e_u = Σ_{f ∈ user_features} e_f")
    print("    Item embedding e_i = Σ_{f ∈ item_features} e_f")
    print()
    print("  Score: ŷ_{ui} = e_u^T e_i + b_u + b_i")
    print("  Loss:  BPR or WARP (Weighted Approximate Rank Pairwise)")
    print()
    print("  WARP loss: rank(i+, u) / n_items  — focuses hard negatives")
    print()

    # Toy simulation
    rng = np.random.default_rng(0)
    K = 8
    # User has 3 features; item has 4 features
    user_feat_emb = rng.normal(0, 0.1, (3, K))
    item_feat_emb = rng.normal(0, 0.1, (4, K))

    def user_emb(feat_ids): return user_feat_emb[feat_ids].mean(axis=0)
    def item_emb(feat_ids): return item_feat_emb[feat_ids].mean(axis=0)

    eu = user_emb([0, 2])      # user has features 0 and 2
    ei_pos = item_emb([1, 3])  # positive item: features 1 and 3
    ei_neg = item_emb([0, 2])  # negative item

    s_pos = eu @ ei_pos
    s_neg = eu @ ei_neg
    print(f"  User embedding: {eu.round(3)}")
    print(f"  Score (positive item): {s_pos:.4f}")
    print(f"  Score (negative item): {s_neg:.4f}")
    print(f"  BPR objective contribution: {np.log(1/(1+np.exp(s_neg-s_pos))):.4f}")


# ── 3. Knowledge graph recommendations ────────────────────────────────────────
def knowledge_graph_rec():
    print("\n=== Knowledge Graph-Enhanced Recommendations ===")
    print()
    print("  Knowledge Graph (KG): entities + relations")
    print("  e.g. (Inception, directedBy, Nolan),")
    print("       (Inception, genre, Sci-Fi),")
    print("       (Nolan, bornIn, UK)")
    print()
    print("  Why KG for RecSys?")
    print("    Rich item semantics beyond user-item interactions")
    print("    Multi-hop reasoning: user liked Nolan film → Nolan → other films")
    print("    Explainability: 'because you liked X (same director as Y)'")
    print()
    print("  Approaches:")
    approaches = [
        ("TransE",          "Relation r: h + r ≈ t; entity embedding; h,r,t ∈ R^K"),
        ("KGCN",            "KG-aware graph conv; aggregate neighbour entity feats"),
        ("KGAT",            "Attention-based KG propagation for RecSys"),
        ("RippleNet",       "User preference propagation over KG ripples"),
        ("KGCL",            "KG + collaborative filtering via contrastive learning"),
    ]
    for m, d in approaches:
        print(f"  {m:<14} {d}")

    print()
    print("  Simple TransE score:")
    rng = np.random.default_rng(0)
    K = 4
    h = rng.normal(0,1,K); h/=np.linalg.norm(h)
    r = rng.normal(0,1,K); r/=np.linalg.norm(r)
    t = rng.normal(0,1,K); t/=np.linalg.norm(t)
    t_pred = h + r
    dist = np.linalg.norm(t_pred - t)
    print(f"  TransE ||h + r - t|| = {dist:.4f}  (low → valid triple)")


# ── 4. Contextual bandits ─────────────────────────────────────────────────────
def contextual_bandits():
    print("\n=== Contextual Bandits for Recommendations ===")
    print()
    print("  Frame recommendation as online decision problem:")
    print("    Context c_t: user features, time, platform, ...")
    print("    Action a_t: recommended item (one of K items)")
    print("    Reward r_t: click / purchase / rating")
    print()
    print("  Goal: maximise Σ_t r_t = minimise regret")
    print()
    print("  Exploration-exploitation trade-off:")
    print("    Exploit: recommend item with best estimated reward")
    print("    Explore: try new items to reduce uncertainty")
    print()

    # UCB contextual bandit simulation (LinUCB)
    rng    = np.random.default_rng(0)
    K = 5; d = 3; alpha = 1.5; T = 200
    # True reward = context · theta_k + noise
    thetas = rng.normal(0, 1, (K, d))

    A = [np.eye(d)   for _ in range(K)]
    b = [np.zeros(d) for _ in range(K)]
    rewards = []

    for t in range(T):
        ctx     = rng.normal(0, 1, d)
        ctx_n   = ctx / (np.linalg.norm(ctx) + 1e-9)
        ucb_vals = []
        for k in range(K):
            A_inv    = np.linalg.inv(A[k])
            theta_k  = A_inv @ b[k]
            UCB_k    = theta_k @ ctx_n + alpha * np.sqrt(ctx_n @ A_inv @ ctx_n)
            ucb_vals.append(UCB_k)
        chosen = np.argmax(ucb_vals)
        r      = float(thetas[chosen] @ ctx_n + rng.normal(0, 0.3))
        A[chosen] += np.outer(ctx_n, ctx_n)
        b[chosen] += r * ctx_n
        rewards.append(r)

    print(f"  LinUCB simulation: K={K} items, d={d} context dims, T={T} steps")
    print(f"  Mean reward:  first 50 steps: {np.mean(rewards[:50]):.3f}")
    print(f"  Mean reward:  last  50 steps: {np.mean(rewards[-50:]):.3f}")
    print(f"  Cumulative reward: {sum(rewards):.2f}")


if __name__ == "__main__":
    hybrid_strategies()
    lightfm_hybrid()
    knowledge_graph_rec()
    contextual_bandits()
