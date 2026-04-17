"""
Working Example: Advanced Recommender Systems Topics
Covers fairness, diversity, debiasing, LLM-based recommenders,
and federated/privacy-preserving recommendations.
"""
import numpy as np
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_advanced_rec")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def softmax(z):
    e = np.exp(z - z.max()); return e / e.sum()


# ── 1. Diversity and serendipity ──────────────────────────────────────────────
def diversity_metrics():
    print("=== Diversity and Serendipity in Recommendations ===")
    print()
    print("  Relevance alone is not enough:")
    print("    Top-N purely by predicted score → redundant, filter-bubble")
    print()

    # Simulate item embeddings and predicted scores
    rng    = np.random.default_rng(0)
    n_items = 20; D = 8
    embeddings = rng.normal(0, 1, (n_items, D))
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    scores = rng.uniform(0.5, 1.0, n_items)

    # Greedy diversity via MMR (Maximal Marginal Relevance)
    def mmr(scores, emb, N=5, lam=0.5):
        selected = []
        remaining = list(range(len(scores)))
        for _ in range(N):
            best, best_score = None, -np.inf
            for i in remaining:
                rel = scores[i]
                red = max(emb[i] @ emb[s] for s in selected) if selected else 0.0
                mmr_score = lam * rel - (1-lam) * red
                if mmr_score > best_score:
                    best, best_score = i, mmr_score
            selected.append(best); remaining.remove(best)
        return selected

    top_5_greedy = np.argsort(scores)[::-1][:5]
    top_5_mmr    = mmr(scores, embeddings, N=5, lam=0.5)

    def intra_list_div(idx, emb):
        if len(idx) < 2: return 0
        pairs = [(i,j) for i in idx for j in idx if i<j]
        return np.mean([1 - emb[i] @ emb[j] for i,j in pairs])

    print(f"  Greedy top-5: {top_5_greedy}")
    print(f"  Intra-list diversity (greedy): {intra_list_div(top_5_greedy, embeddings):.4f}")
    print(f"  MMR top-5:    {top_5_mmr}")
    print(f"  Intra-list diversity (MMR):    {intra_list_div(top_5_mmr, embeddings):.4f}")

    print()
    print("  Diversity metrics:")
    metrics = [
        ("ILD (Intra-List Distance)",  "avg pairwise distance in recommended list"),
        ("Coverage",                    "fraction of items/categories represented"),
        ("Serendipity",                "unexpected + relevant items"),
        ("Novelty",                    "-log p(item)  (popularity-inverse)"),
        ("Surprise",                   "distance from user profile expectations"),
    ]
    for m, d in metrics:
        print(f"  {m:<30} {d}")


# ── 2. Popularity bias and debiasing ──────────────────────────────────────────
def debiasing():
    print("\n=== Bias and Debiasing in Recommenders ===")
    print()
    print("  Types of bias:")
    biases = [
        ("Popularity bias",    "Popular items dominate; long-tail items ignored"),
        ("Position bias",      "Users click higher-ranked items more (exposure bias)"),
        ("Selection bias",     "Data missing NOT at random (users only rate liked items)"),
        ("Conformity bias",    "Users rate items higher due to group influence"),
        ("Filter bubble",      "Reinforcement loop narrowing content exposure"),
    ]
    for b, d in biases:
        print(f"  {b:<22} {d}")
    print()
    print("  Debiasing methods:")
    methods = [
        ("IPS weighting",   "Inverse propensity score; reweight by 1/P(exposed)"),
        ("Causal inference","PO4Rec; counterfactual reasoning"),
        ("Popularity regularise","Penalise over-recommendation of popular items"),
        ("Unbiased BPR",    "UBPR — correct for position bias in implicit data"),
        ("BC-Loss",         "Bilateral branch network for balanced learning"),
    ]
    for m, d in methods:
        print(f"  {m:<22} {d}")

    # Simulate IPS reweighting
    print()
    print("  IPS simulation:")
    rng    = np.random.default_rng(0)
    items  = np.arange(10)
    pop    = np.array([100, 80, 60, 40, 20, 15, 10, 5, 3, 1], dtype=float)
    pop_p  = pop / pop.sum()   # exposure probability proportional to popularity
    clicks = (rng.uniform(size=10) < pop_p * 3).astype(float)
    # Biased recommendation (sort by click rate)
    biased_order = clicks.argsort()[::-1]
    # IPS-corrected score
    ips_score    = clicks / (pop_p + 1e-6)
    ips_order    = ips_score.argsort()[::-1]
    print(f"  Biased top-5 items:      {biased_order[:5]}")
    print(f"  IPS-corrected top-5:     {ips_order[:5]}")
    print(f"  Long-tail items in IPS:  {(ips_order[:5] >= 5).sum()}/5")


# ── 3. Fairness ───────────────────────────────────────────────────────────────
def fairness():
    print("\n=== Fairness in Recommender Systems ===")
    print()
    print("  Two-sided fairness:")
    print("    User-side: equitable quality of recommendations across user groups")
    print("    Provider-side: equitable exposure across item providers/creators")
    print()
    print("  Fairness metrics:")
    fm = [
        ("NDKL",           "Normalised Discounted KL-divergence from ideal exposure"),
        ("Exposure fairness","Producer exposure ≥ relevance threshold"),
        ("DP (Demo. Parity)","Equal positive rates across demographic groups"),
        ("EO (Equal Opp)", "Equal true positive rates across groups"),
        ("Calibration",    "User preference distribution matches served distribution"),
    ]
    for f, d in fm:
        print(f"  {f:<22} {d}")
    print()
    print("  Methods:")
    print("    Re-ranking: post-hoc fairness adjustment")
    print("    In-processing: fairness constraint during training")
    print("    Adversarial: discriminator removes protected group features")


# ── 4. LLM-based recommendations ─────────────────────────────────────────────
def llm_recommenders():
    print("\n=== LLM-Based Recommenders ===")
    print()
    print("  Approaches:")
    approaches = [
        ("Zero-shot",     "Prompt LLM directly to rank items; no training needed"),
        ("Few-shot",      "Provide examples of user history in prompt"),
        ("LLM as ranker", "Generate ranking from candidate set via chain-of-thought"),
        ("P5",            "Pre-train on recommendation tasks as text-to-text (T5)"),
        ("CTRL-Rec",      "Control code conditioning for diverse recs"),
        ("LLaRa",         "LLaMA fine-tuned with structured prompts for RecSys"),
        ("RecSys as LM",  "Tokenise items; model history as language"),
    ]
    for a, d in approaches:
        print(f"  {a:<18} {d}")
    print()
    print("  Example zero-shot prompt:")
    print('    "Based on movies the user rated highly: Inception (5★), Interstellar (4★),')
    print('     recommend 3 movies they might enjoy. Output: JSON list."')
    print()
    print("  Challenges:")
    print("    Hallucination: LLM may invent non-existent items")
    print("    Latency:       large models too slow for real-time retrieval")
    print("    Context limit: user history may exceed context window")
    print("    Solution:      use LLM for re-ranking only (small candidate set)")


if __name__ == "__main__":
    diversity_metrics()
    debiasing()
    fairness()
    llm_recommenders()
