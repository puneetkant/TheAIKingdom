"""
Working Example 2: Hybrid Recommender — combining collaborative + content signals
==================================================================================
Weighted ensemble of content similarity and collaborative factorisation scores.

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

MOVIES = [
    ("Matrix",       "sci-fi action hacker ai"),
    ("Terminator",   "sci-fi action robot ai"),
    ("Inception",    "thriller sci-fi dreams"),
    ("Interstellar", "sci-fi space drama"),
    ("Batman",       "action superhero crime"),
]
titles, docs = zip(*MOVIES)

def cf_scores(n_items, seed=0):
    """Simulate MF scores for user 0 on all items."""
    rng = np.random.default_rng(seed)
    return rng.dirichlet(np.ones(n_items))  # normalised fake CF scores

def demo():
    print("=== Hybrid Recommender: Content + CF Ensemble ===")
    tfidf = TfidfVectorizer()
    item_mat = tfidf.fit_transform(docs)
    content_sim = cosine_similarity(item_mat)

    query = 0  # Matrix
    content_scores = content_sim[query]
    cf = cf_scores(len(titles))

    for alpha in [0.0, 0.5, 1.0]:
        hybrid = (1-alpha) * content_scores + alpha * cf
        ranked = np.argsort(hybrid)[::-1]
        top3 = [titles[r] for r in ranked if r != query][:3]
        print(f"  alpha={alpha:.1f} (alpha*CF + (1-alpha)*Content): {top3}")

    fig, ax = plt.subplots(figsize=(7, 3))
    x = np.arange(len(titles))
    ax.bar(x-0.2, content_scores, 0.4, label="Content")
    ax.bar(x+0.2, cf, 0.4, label="CF")
    ax.set_xticks(x); ax.set_xticklabels(titles, rotation=15)
    ax.legend(); ax.set_title("Content vs CF Scores for 'Matrix'")
    plt.tight_layout(); plt.savefig(OUTPUT / "hybrid_recsys.png"); plt.close()
    print("  Saved hybrid_recsys.png")

def demo_switching_hybrid():
    """Switching hybrid: use CF when enough ratings, else content-based."""
    print("\n=== Switching Hybrid Recommender ===")
    n_ratings_cf_threshold = 3
    # Simulate user interaction counts per item
    rng = np.random.default_rng(42)
    interaction_counts = rng.integers(0, 6, len(titles))

    tfidf = TfidfVectorizer()
    item_mat = tfidf.fit_transform(docs)
    content_sim = cosine_similarity(item_mat)
    cf = np.array([0.15, 0.30, 0.25, 0.20, 0.10])  # fixed fake CF scores

    query = 0
    hybrid_scores = []
    strategies = []
    for idx in range(len(titles)):
        if interaction_counts[idx] >= n_ratings_cf_threshold:
            score = cf[idx]; strat = "CF"
        else:
            score = content_sim[query, idx]; strat = "Content"
        hybrid_scores.append(score)
        strategies.append(strat)

    ranked = np.argsort(hybrid_scores)[::-1]
    print(f"  Recommendations for '{titles[query]}':")
    for r in ranked:
        if r != query:
            print(f"    {titles[r]:15s}  score={hybrid_scores[r]:.4f}  "
                  f"strategy={strategies[r]}  ratings={interaction_counts[r]}")


def demo_cascade_hybrid():
    """Cascade hybrid: content-based pre-filters, CF re-ranks."""
    print("\n=== Cascade Hybrid Recommender ===")
    tfidf = TfidfVectorizer()
    item_mat = tfidf.fit_transform(docs)
    content_sim = cosine_similarity(item_mat)
    cf = np.array([0.15, 0.30, 0.25, 0.20, 0.10])
    query = 0

    # Stage 1: keep top-3 by content similarity
    content_scores = content_sim[query].copy()
    content_scores[query] = -1
    top3_idx = np.argsort(content_scores)[::-1][:3]
    print(f"  Stage 1 (Content top-3): {[titles[i] for i in top3_idx]}")

    # Stage 2: re-rank top-3 by CF scores
    reranked = sorted(top3_idx, key=lambda i: cf[i], reverse=True)
    print(f"  Stage 2 (CF re-rank):    {[titles[i] for i in reranked]}")


if __name__ == "__main__":
    demo()
    demo_switching_hybrid()
    demo_cascade_hybrid()
