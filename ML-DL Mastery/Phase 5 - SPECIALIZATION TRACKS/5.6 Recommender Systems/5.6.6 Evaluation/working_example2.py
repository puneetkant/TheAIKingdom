"""
Working Example 2: RecSys Evaluation — Precision@k, Recall@k, NDCG, MRR, coverage
====================================================================================
Computes ranking metrics for a simulated recommender system.

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

def precision_at_k(recommended, relevant, k):
    recs = recommended[:k]
    return len(set(recs) & set(relevant)) / k

def recall_at_k(recommended, relevant, k):
    recs = recommended[:k]
    return len(set(recs) & set(relevant)) / (len(relevant) + 1e-8)

def ndcg_at_k(recommended, relevant, k):
    recs = recommended[:k]
    dcg = sum(1/np.log2(i+2) for i, r in enumerate(recs) if r in relevant)
    idcg = sum(1/np.log2(i+2) for i in range(min(len(relevant), k)))
    return dcg / (idcg + 1e-8)

def mrr(recommended, relevant):
    for i, r in enumerate(recommended):
        if r in relevant:
            return 1 / (i+1)
    return 0

def demo():
    np.random.seed(42)
    N_USERS, N_ITEMS = 50, 100
    # Simulate ground truth: each user has 5 relevant items
    relevant_all = {u: set(np.random.choice(N_ITEMS, 5, replace=False)) for u in range(N_USERS)}
    # Simulate two recommenders
    def random_rec(u): return np.random.permutation(N_ITEMS).tolist()
    def smart_rec(u):
        # 60% chance to include each relevant item near top
        rel = list(relevant_all[u])
        non_rel = [i for i in range(N_ITEMS) if i not in relevant_all[u]]
        np.random.shuffle(non_rel)
        rec = []
        for i in range(10):
            if np.random.rand() < 0.6 and rel:
                rec.append(rel.pop(0))
            else:
                rec.append(non_rel.pop(0))
        rec += non_rel
        return rec

    print("=== RecSys Evaluation Metrics ===")
    print(f"\n  {'Metric':15s} {'Random':>10s} {'Smart':>10s}")
    print("  " + "-"*38)
    for metric_name, metric_fn, k_val in [
        ("P@5",  lambda r, rel: precision_at_k(r, rel, 5), 5),
        ("R@10", lambda r, rel: recall_at_k(r, rel, 10),  10),
        ("NDCG@10", lambda r, rel: ndcg_at_k(r, rel, 10), 10),
        ("MRR",  lambda r, rel: mrr(r, rel), None),
    ]:
        rand_vals = [metric_fn(random_rec(u), relevant_all[u]) for u in range(N_USERS)]
        smart_vals = [metric_fn(smart_rec(u), relevant_all[u]) for u in range(N_USERS)]
        print(f"  {metric_name:15s} {np.mean(rand_vals):10.4f} {np.mean(smart_vals):10.4f}")

    ks = [1, 3, 5, 10, 20]
    p_rand = [np.mean([precision_at_k(random_rec(u), relevant_all[u], k) for u in range(N_USERS)]) for k in ks]
    p_smart = [np.mean([precision_at_k(smart_rec(u), relevant_all[u], k) for u in range(N_USERS)]) for k in ks]
    plt.figure(figsize=(6, 3))
    plt.plot(ks, p_rand, "o--", label="Random"); plt.plot(ks, p_smart, "s-", label="Smart")
    plt.xlabel("k"); plt.ylabel("Precision@k"); plt.legend(); plt.title("Precision@k Curve")
    plt.tight_layout(); plt.savefig(OUTPUT / "recsys_evaluation.png"); plt.close()
    print("\n  Saved recsys_evaluation.png")

def demo_catalog_coverage():
    """Measure catalog coverage and long-tail exposure across users."""
    print("\n=== Catalog Coverage ===")
    np.random.seed(42)
    N_USERS, N_ITEMS, K = 50, 100, 10
    relevant_all = {u: set(np.random.choice(N_ITEMS, 5, replace=False)) for u in range(N_USERS)}

    # Popularity-biased recommender (only recommends top-20 items)
    popular_items = list(range(20))
    def pop_rec(u): return popular_items + list(np.random.choice(
        [i for i in range(N_ITEMS) if i >= 20], N_ITEMS - 20, replace=False))

    # Diverse recommender
    def diverse_rec(u): return np.random.permutation(N_ITEMS).tolist()

    for name, rec_fn in [("Popular", pop_rec), ("Diverse", diverse_rec)]:
        all_recs = set()
        for u in range(N_USERS):
            all_recs.update(rec_fn(u)[:K])
        coverage = len(all_recs) / N_ITEMS
        print(f"  {name:10s} catalog coverage: {coverage:.2%}")


def demo_metric_tradeoffs():
    """Show precision vs diversity tradeoff across lambda in MMR."""
    print("\n=== Precision-Diversity Tradeoff ===")
    rng = np.random.default_rng(9)
    N_ITEMS = 20; K = 5
    # Relevance scores and random similarity matrix
    scores_v = rng.uniform(0, 1, N_ITEMS)
    sim_mat = rng.uniform(0, 1, (N_ITEMS, N_ITEMS))
    sim_mat = (sim_mat + sim_mat.T) / 2; np.fill_diagonal(sim_mat, 1)

    def mmr_select(scores, sim, lam, k):
        selected = []; cands = list(range(len(scores)))
        while len(selected) < k and cands:
            if not selected:
                best = max(cands, key=lambda i: scores[i])
            else:
                best = max(cands, key=lambda i: lam*scores[i] - (1-lam)*max(sim[i,j] for j in selected))
            selected.append(best); cands.remove(best)
        return selected

    print(f"  {'Lambda':>8}  {'Avg Relevance':>15}  {'Avg Diversity':>14}")
    for lam in [1.0, 0.7, 0.5, 0.3, 0.0]:
        sel = mmr_select(scores_v, sim_mat, lam, K)
        rel = np.mean(scores_v[sel])
        pairs = [(i, j) for i in sel for j in sel if i < j]
        div = np.mean([1 - sim_mat[i, j] for i, j in pairs]) if pairs else 0
        print(f"  {lam:>8.1f}  {rel:>15.3f}  {div:>14.3f}")


if __name__ == "__main__":
    demo()
    demo_catalog_coverage()
    demo_metric_tradeoffs()
