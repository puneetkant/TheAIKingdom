"""
Working Example: Ranking and Recommendation Metrics
Covers Precision@K, Recall@K, MAP, NDCG, MRR, Hit Rate, and collaborative
filtering evaluation.
"""
import numpy as np


# ── 1. Introduction to ranking metrics ───────────────────────────────────────
def intro():
    print("=== Ranking and Recommendation Metrics ===")
    print("  Evaluate ordered lists, not just set membership")
    print("  Key questions:")
    print("    - Are relevant items at the top of the list?")
    print("    - How many relevant items are retrieved?")
    print("    - How precisely are the top-K items relevant?")


# ── 2. Precision@K, Recall@K, F1@K ──────────────────────────────────────────
def precision_recall_at_k():
    print("\n=== Precision@K, Recall@K, F1@K ===")
    # Recommended items (order matters for P@K)
    recommended = [1, 5, 3, 7, 2, 8, 4, 6, 9, 10]
    relevant     = {1, 3, 5, 9, 11, 12}   # ground truth

    print(f"  Recommended: {recommended}")
    print(f"  Relevant:    {sorted(relevant)}")
    print()
    print(f"  {'K':<5} {'Precision@K':<18} {'Recall@K':<18} {'F1@K'}")
    for k in [1, 2, 3, 5, 7, 10]:
        top_k = set(recommended[:k])
        hits  = len(top_k & relevant)
        prec  = hits / k
        rec   = hits / len(relevant)
        f1    = 2*prec*rec/(prec+rec+1e-9)
        print(f"  {k:<5} {prec:<18.4f} {rec:<18.4f} {f1:.4f}")


# ── 3. Average Precision (AP) and MAP ────────────────────────────────────────
def map_score():
    print("\n=== MAP (Mean Average Precision) ===")
    print("  AP = Σ (P@k × rel(k)) / |relevant|")
    print("  MAP = mean AP over all users/queries")

    def avg_precision(recommended, relevant_set):
        hits, ap = 0, 0.0
        for k, item in enumerate(recommended, 1):
            if item in relevant_set:
                hits += 1
                ap   += hits / k
        return ap / max(len(relevant_set), 1)

    queries = [
        {"rec": [1, 5, 3, 7, 2],  "rel": {1, 3, 5}},        # query 0
        {"rec": [4, 2, 1, 8, 3],  "rel": {2, 3}},            # query 1
        {"rec": [10, 11, 1, 2, 5],"rel": {5, 6, 7}},         # query 2
    ]
    aps = []
    for i, q in enumerate(queries):
        ap = avg_precision(q["rec"], q["rel"])
        aps.append(ap)
        print(f"  Query {i}: rec={q['rec']}  rel={sorted(q['rel'])}  AP={ap:.4f}")

    print(f"\n  MAP = {np.mean(aps):.4f}")


# ── 4. NDCG (Normalised Discounted Cumulative Gain) ──────────────────────────
def ndcg():
    print("\n=== NDCG (Normalised DCG) ===")
    print("  Rewards relevant items at higher ranks (position-aware)")
    print("  DCG = Σ (2^rel_i - 1) / log2(i+2)")
    print("  NDCG = DCG / IDCG  (IDCG = DCG of perfect ranking)")

    def dcg(gains, k=None):
        if k: gains = gains[:k]
        return sum((2**g - 1) / np.log2(i+2) for i, g in enumerate(gains))

    def ndcg_score(gains, k=None):
        ideal = sorted(gains, reverse=True)
        idcg  = dcg(ideal, k)
        return dcg(gains, k) / (idcg + 1e-9)

    # Binary relevance
    print("\n  Binary relevance (0/1):")
    recommended = [1, 5, 3, 7, 2, 8, 4, 6, 9, 10]
    relevant    = {1, 3, 5, 9}
    gains       = [1 if item in relevant else 0 for item in recommended]
    print(f"  Ranking: {gains}")
    for k in [3, 5, 10]:
        score = ndcg_score(gains, k)
        print(f"  NDCG@{k} = {score:.4f}")

    # Graded relevance
    print("\n  Graded relevance (0-4):")
    graded_gains = [3, 2, 3, 0, 1, 2, 3, 2, 0, 1]  # relevance scores of ranked items
    ideal_order  = sorted(graded_gains, reverse=True)
    print(f"  Actual: {graded_gains}")
    print(f"  Ideal:  {ideal_order}")
    for k in [3, 5, 10]:
        score = ndcg_score(graded_gains, k)
        print(f"  NDCG@{k} = {score:.4f}")


# ── 5. MRR (Mean Reciprocal Rank) ────────────────────────────────────────────
def mrr_score():
    print("\n=== MRR (Mean Reciprocal Rank) ===")
    print("  RR = 1 / rank_of_first_relevant_item")
    print("  MRR = mean RR over all queries")
    print("  Useful when user only cares about the FIRST relevant result")

    queries = [
        ([5, 3, 1, 8, 2], {1, 7}),   # first hit at rank 3 → RR = 1/3
        ([2, 4, 6, 8, 10], {2}),      # first hit at rank 1 → RR = 1
        ([9, 8, 7, 6, 5],  {3}),      # no hit in top-5  → RR = 0
    ]
    rrs = []
    for i, (rec, rel) in enumerate(queries):
        rr = 0.0
        for rank, item in enumerate(rec, 1):
            if item in rel:
                rr = 1.0 / rank
                break
        rrs.append(rr)
        print(f"  Query {i}: rec={rec}  rel={sorted(rel)}  RR={rr:.4f}")

    print(f"\n  MRR = {np.mean(rrs):.4f}")


# ── 6. Hit Rate (HR@K) and Coverage ──────────────────────────────────────────
def hit_rate():
    print("\n=== Hit Rate@K and Coverage ===")
    print("  HR@K = fraction of users for whom at least one relevant item")
    print("         appears in top-K recommendations")

    rng    = np.random.default_rng(0)
    n_user = 100
    n_item = 50
    K      = 10

    hits = 0
    all_recommended = set()
    for _ in range(n_user):
        rec = list(rng.choice(n_item, K, replace=False))
        rel = set(rng.choice(n_item, 3, replace=False))
        if set(rec) & rel:
            hits += 1
        all_recommended.update(rec)

    hr       = hits / n_user
    coverage = len(all_recommended) / n_item
    print(f"  HR@{K}    = {hr:.4f}  ({hits}/{n_user} users had ≥1 hit)")
    print(f"  Catalog coverage = {coverage:.4f}  ({len(all_recommended)}/{n_item} items recommended)")


# ── 7. Collaborative filtering evaluation ─────────────────────────────────────
def collaborative_filtering_eval():
    print("\n=== Collaborative Filtering Evaluation ===")
    print("  Offline: hold-out last interaction per user, measure ranking metrics")
    print("  Online:  A/B test with click-through rate (CTR), conversion rate")
    print()
    print("  Common pitfalls:")
    print("    - Popularity bias: always recommending popular items inflates metrics")
    print("    - Cold start: new users/items have no interaction history")
    print("    - Temporal leakage: using future data to predict past interactions")
    print()

    # Simulate rating prediction (MAE, RMSE on predicted vs actual ratings)
    rng     = np.random.default_rng(1)
    n       = 500
    actual  = rng.uniform(1, 5, n)            # true ratings [1,5]
    predicted = actual + rng.normal(0, 0.8, n)  # noisy predictions
    predicted = predicted.clip(1, 5)

    mae  = np.abs(actual - predicted).mean()
    rmse = np.sqrt(((actual - predicted)**2).mean())
    print(f"  Rating prediction (explicit feedback):")
    print(f"    MAE  = {mae:.4f}")
    print(f"    RMSE = {rmse:.4f}")
    print()
    print("  Implicit feedback: treat as ranking; evaluate with P@K, MAP, NDCG")
    print("  Explicit feedback: evaluate with MAE/RMSE on predicted ratings")


if __name__ == "__main__":
    intro()
    precision_recall_at_k()
    map_score()
    ndcg()
    mrr_score()
    hit_rate()
    collaborative_filtering_eval()
