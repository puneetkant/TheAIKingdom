"""
Working Example: Recommender Systems Evaluation
Covers offline and online evaluation, ranking metrics,
A/B testing, and production evaluation considerations.
"""
import numpy as np
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_rec_eval")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -- 1. Offline ranking metrics ------------------------------------------------
def ranking_metrics():
    print("=== Recommender System Evaluation Metrics ===")
    print()
    # Simulated: user interacted with item IDs {2, 5, 8}
    relevant = {2, 5, 8}
    ranked_list = [3, 5, 1, 2, 9, 8, 4, 7]  # recommendation order

    def precision_at_k(recs, rel, k):
        return sum(1 for r in recs[:k] if r in rel) / k

    def recall_at_k(recs, rel, k):
        return sum(1 for r in recs[:k] if r in rel) / len(rel)

    def ap_at_k(recs, rel, k):
        hits = 0; prec_sum = 0
        for i, r in enumerate(recs[:k], 1):
            if r in rel:
                hits += 1; prec_sum += hits / i
        return prec_sum / min(len(rel), k)

    def ndcg_at_k(recs, rel, k):
        dcg  = sum(1/np.log2(i+2) for i,r in enumerate(recs[:k]) if r in rel)
        idcg = sum(1/np.log2(i+2) for i in range(min(len(rel), k)))
        return dcg / (idcg + 1e-9)

    def rr(recs, rel):
        for i, r in enumerate(recs, 1):
            if r in rel: return 1/i
        return 0

    def hit_rate_at_k(recs, rel, k):
        return int(any(r in rel for r in recs[:k]))

    print(f"  Relevant items: {sorted(relevant)}")
    print(f"  Ranked list:    {ranked_list}")
    print()

    for k in [3, 5, 8]:
        p = precision_at_k(ranked_list, relevant, k)
        r = recall_at_k(ranked_list, relevant, k)
        ap = ap_at_k(ranked_list, relevant, k)
        nd = ndcg_at_k(ranked_list, relevant, k)
        hr = hit_rate_at_k(ranked_list, relevant, k)
        print(f"  @k={k}: P={p:.3f}  R={r:.3f}  AP={ap:.3f}  NDCG={nd:.3f}  HitRate={hr}")

    mrr = rr(ranked_list, relevant)
    print(f"  MRR = {mrr:.3f}  (reciprocal rank of first hit)")


# -- 2. Multi-user evaluation --------------------------------------------------
def multi_user_eval():
    print("\n=== Multi-User Evaluation ===")
    print("  Average metrics over all users (macro-average)")
    rng = np.random.default_rng(0)
    n_users = 100; n_items = 200; k = 10

    all_ndcg = []
    for _ in range(n_users):
        n_rel  = rng.integers(1, 10)
        rel    = set(rng.choice(n_items, n_rel, replace=False))
        recs   = list(rng.choice(n_items, k, replace=False))
        hits   = [1 if r in rel else 0 for r in recs]
        dcg    = sum(h/np.log2(i+2) for i, h in enumerate(hits))
        idcg   = sum(1/np.log2(i+2) for i in range(min(len(rel), k)))
        ndcg   = dcg / (idcg + 1e-9)
        all_ndcg.append(ndcg)

    all_ndcg = np.array(all_ndcg)
    print(f"  {n_users} users, k={k}: "
          f"mean NDCG={all_ndcg.mean():.4f}  "
          f"std={all_ndcg.std():.4f}  "
          f"min={all_ndcg.min():.4f}  "
          f"max={all_ndcg.max():.4f}")

    # Plot distribution
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(all_ndcg, bins=20, edgecolor='black')
    ax.set_xlabel("NDCG@10"); ax.set_ylabel("Users"); ax.set_title("Per-User NDCG Distribution")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "ndcg_dist.png")
    plt.savefig(path, dpi=90); plt.close()
    print(f"  Plot: {path}")


# -- 3. Online evaluation and A/B testing -------------------------------------
def online_evaluation():
    print("\n=== Online Evaluation (A/B Testing) ===")
    print()
    print("  Problem: offline metrics != online business metrics")
    print("    NDCG up 2% offline != CTR up 2% online (correlation ~= 0.4-0.7)")
    print()
    print("  A/B test design:")
    print("    Control: existing model (A)")
    print("    Treatment: new model (B)")
    print("    User-level random assignment; track for 2-4 weeks")
    print()

    # Simulate A/B test
    rng = np.random.default_rng(42)
    n_a = n_b = 5000
    # CTR (click-through rate) simulation
    p_a = 0.20; p_b = 0.22   # true CTR
    clicks_a = rng.binomial(1, p_a, n_a)
    clicks_b = rng.binomial(1, p_b, n_b)

    ctr_a = clicks_a.mean(); ctr_b = clicks_b.mean()
    lift = (ctr_b - ctr_a) / ctr_a

    # Two-proportion z-test
    p_pool = (clicks_a.sum() + clicks_b.sum()) / (n_a + n_b)
    se     = np.sqrt(p_pool * (1-p_pool) * (1/n_a + 1/n_b))
    z      = (ctr_b - ctr_a) / (se + 1e-9)
    p_val  = 2 * (1 - (0.5 * (1 + np.sign(z) * (1 - np.exp(-0.7 * z**2)))))

    print(f"  n_A={n_a}  n_B={n_b}")
    print(f"  CTR_A={ctr_a:.4f}  CTR_B={ctr_b:.4f}  Lift={lift:+.2%}")
    print(f"  z={z:.3f}  p~={abs(p_val):.4f}  {'SIGNIFICANT (p<0.05)' if abs(p_val) < 0.05 else 'NOT significant'}")
    print()
    print("  Online metrics to track:")
    online_metrics = [
        ("CTR",         "Click-through rate"),
        ("CVR",         "Conversion rate (purchase/signup)"),
        ("Revenue",     "Revenue per user"),
        ("Diversity",   "User-perceived variety"),
        ("Session len", "Time spent / pages per session"),
        ("Churn",       "User retention rate"),
    ]
    for m, d in online_metrics:
        print(f"  {m:<12} {d}")


# -- 4. Evaluation pitfalls ----------------------------------------------------
def evaluation_pitfalls():
    print("\n=== Evaluation Pitfalls ===")
    pitfalls = [
        ("Popularity bias in eval",  "Popular items dominate held-out sets; inflate metrics"),
        ("Temporal leakage",         "Train contains future data; must use time-based split"),
        ("User leakage",             "Same user interaction in train + test"),
        ("Mismatched metrics",       "Optimise NDCG but care about revenue"),
        ("Cold-start neglect",       "Evaluate only warm users; cold start harder"),
        ("Small test set",           "High variance in metrics; use confidence intervals"),
        ("Offline-online disconnect", "Offline gains don't translate; use interleaving"),
    ]
    print(f"  {'Pitfall':<28} {'Notes'}")
    print(f"  {'-'*28} {'-'*45}")
    for p, d in pitfalls:
        print(f"  {p:<28} {d}")


if __name__ == "__main__":
    ranking_metrics()
    multi_user_eval()
    online_evaluation()
    evaluation_pitfalls()
