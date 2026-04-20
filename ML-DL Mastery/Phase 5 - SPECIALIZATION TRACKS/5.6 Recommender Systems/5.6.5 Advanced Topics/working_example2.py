"""
Working Example 2: Advanced RecSys — diversity, serendipity, exploration vs exploitation
=========================================================================================
Demonstrates diversity re-ranking and epsilon-greedy exploration.

Run:  python working_example2.py
"""
from pathlib import Path
try:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    raise SystemExit("pip install numpy matplotlib scikit-learn")

OUTPUT = Path(__file__).parent / "output"
OUTPUT.mkdir(exist_ok=True)

ITEMS = {
    "Matrix": "sci-fi action hacker",
    "Terminator": "sci-fi action robot",
    "Interstellar": "sci-fi space drama",
    "Batman": "action superhero crime",
    "Inception": "thriller sci-fi dreams",
    "Titanic": "romance drama disaster",
    "Toy Story": "animation comedy family",
    "The Godfather": "crime drama mafia",
}
titles = list(ITEMS.keys()); docs = list(ITEMS.values())

def mmr_rerank(scores, sim_matrix, lam=0.5, k=5):
    """Maximal Marginal Relevance re-ranking for diversity."""
    selected = []
    candidates = list(range(len(scores)))
    while len(selected) < k and candidates:
        if not selected:
            best = max(candidates, key=lambda i: scores[i])
        else:
            def mmr_score(i):
                rel = scores[i]
                red = max(sim_matrix[i, j] for j in selected)
                return lam * rel - (1-lam) * red
            best = max(candidates, key=mmr_score)
        selected.append(best); candidates.remove(best)
    return selected

def demo():
    print("=== Advanced RecSys: Diversity Re-ranking (MMR) ===")
    tfidf = TfidfVectorizer()
    item_mat = tfidf.fit_transform(docs)
    sim = cosine_similarity(item_mat)
    # Simulate relevance scores (high for sci-fi)
    scores = np.array([1.0 if "sci-fi" in d else 0.4 for d in docs])

    top_k_naive = np.argsort(scores)[::-1][:5]
    top_k_mmr = mmr_rerank(scores, sim, lam=0.5, k=5)

    print("  Naive top-5:  ", [titles[i] for i in top_k_naive])
    print("  MMR top-5:    ", [titles[i] for i in top_k_mmr])

    # Diversity metric: avg pairwise distance in list
    def diversity(idxs):
        pairs = [(i, j) for i in idxs for j in idxs if i < j]
        return np.mean([1 - sim[i, j] for i, j in pairs]) if pairs else 0

    print(f"  Naive diversity: {diversity(top_k_naive):.3f}")
    print(f"  MMR diversity:   {diversity(top_k_mmr):.3f}")

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.bar(range(len(titles)), scores, label="Relevance", alpha=0.7)
    for i in top_k_naive:
        ax.scatter(i, scores[i]+0.02, marker="^", color="blue", s=80)
    for i in top_k_mmr:
        ax.scatter(i, scores[i]+0.06, marker="o", color="red", s=80)
    ax.set_xticks(range(len(titles))); ax.set_xticklabels(titles, rotation=30, ha="right", fontsize=8)
    ax.set_title("Naive (^) vs MMR (●) top-5")
    plt.tight_layout(); plt.savefig(OUTPUT / "recsys_diversity.png"); plt.close()
    print("  Saved recsys_diversity.png")

def demo_exploration_ucb():
    """UCB1 exploration for cold-start items."""
    print("\n=== Exploration: UCB for New Items ===")
    rng = np.random.default_rng(42)
    true_ctr = rng.uniform(0.05, 0.4, len(titles))  # click-through rates
    N = np.zeros(len(titles)); Q = np.zeros(len(titles))
    rewards_log = []
    for t in range(1, 201):
        ucb = Q + 2 * np.sqrt(np.log(t) / (N + 1e-9))
        i = ucb.argmax()
        reward = float(rng.random() < true_ctr[i])
        N[i] += 1; Q[i] += (reward - Q[i]) / N[i]
        rewards_log.append(reward)
    best_ctr = true_ctr.max()
    achieved  = np.mean(rewards_log[-50:])
    print(f"  Optimal CTR:   {best_ctr:.3f}")
    print(f"  UCB last-50 avg CTR: {achieved:.3f}")
    print(f"  Regret fraction: {(best_ctr - achieved) / best_ctr:.1%}")


def demo_temporal_dynamics():
    """Simulate concept drift: item popularity shifts over time."""
    print("\n=== Temporal Dynamics Simulation ===")
    rng = np.random.default_rng(7)
    n_items = len(titles); T = 100
    # Popularity shifts at T=50
    early_pop = rng.dirichlet(np.ones(n_items))
    late_pop  = rng.dirichlet(np.ones(n_items))

    early_top = np.argsort(early_pop)[::-1][:3]
    late_top  = np.argsort(late_pop)[::-1][:3]
    print(f"  Early top items: {[titles[i] for i in early_top]}")
    print(f"  Late  top items: {[titles[i] for i in late_top]}")
    overlap = len(set(early_top) & set(late_top))
    print(f"  Top-3 overlap: {overlap}/3  (drift = {1 - overlap/3:.1%})")


if __name__ == "__main__":
    demo()
    demo_exploration_ucb()
    demo_temporal_dynamics()
