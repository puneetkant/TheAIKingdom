"""
Working Example 2: Content-Based Filtering — TF-IDF item profiles + cosine similarity
======================================================================================
Builds item content vectors and recommends similar items.

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

# Synthetic movie catalogue
MOVIES = [
    ("The Matrix",       "action sci-fi technology hacker artificial intelligence"),
    ("Terminator 2",     "action sci-fi robot time travel artificial intelligence"),
    ("Inception",        "thriller sci-fi dreams mind heist"),
    ("Interstellar",     "sci-fi space time travel wormhole drama"),
    ("The Dark Knight",  "action crime drama superhero batman"),
    ("Iron Man",         "action superhero technology billionaire"),
    ("Avengers",         "action superhero team fantasy drama"),
    ("Blade Runner",     "sci-fi neo-noir artificial intelligence dystopia"),
    ("Ex Machina",       "sci-fi thriller artificial intelligence robot"),
    ("Her",              "romance sci-fi artificial intelligence drama"),
]
titles, docs = zip(*MOVIES)

def demo():
    print("=== Content-Based Filtering ===")
    tfidf = TfidfVectorizer()
    item_matrix = tfidf.fit_transform(docs)  # (n_items, n_terms)
    sim_matrix = cosine_similarity(item_matrix)  # (n_items, n_items)

    query_idx = 0  # "The Matrix"
    sims = sim_matrix[query_idx]
    ranked = np.argsort(sims)[::-1]
    print(f"\n  Recommendations for '{titles[query_idx]}':")
    for i in ranked[1:5]:
        print(f"    {titles[i]:25s} similarity: {sims[i]:.3f}")

    # User profile: weighted average of liked item vectors
    liked = [0, 1, 7]  # Matrix, Terminator, Blade Runner
    user_vec = item_matrix[liked].mean(axis=0)
    user_sims = cosine_similarity(user_vec, item_matrix).ravel()
    ranked_user = np.argsort(user_sims)[::-1]
    print(f"\n  User profile recommendations (liked: {[titles[i] for i in liked]}):")
    for i in ranked_user:
        if i not in liked:
            print(f"    {titles[i]:25s} score: {user_sims[i]:.3f}")
            if len([j for j in ranked_user[:list(ranked_user).index(i)+1] if j not in liked]) >= 3:
                break

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(sim_matrix, cmap="Blues")
    ax.set_xticks(range(len(titles))); ax.set_yticks(range(len(titles)))
    ax.set_xticklabels([t[:10] for t in titles], rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels([t[:10] for t in titles], fontsize=7)
    ax.set_title("Content Similarity Matrix")
    plt.tight_layout(); plt.savefig(OUTPUT / "content_based.png"); plt.close()
    print("\n  Saved content_based.png")

if __name__ == "__main__":
    demo()
