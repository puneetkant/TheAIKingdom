"""
Working Example: Content-Based Filtering
Covers item profiles, user profiles, TF-IDF, cosine similarity,
and content-based recommendation systems.
"""
import numpy as np
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_content_based")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -- 1. Concept overview -------------------------------------------------------
def overview():
    print("=== Content-Based Filtering ===")
    print()
    print("  Core idea: recommend items SIMILAR to items a user has liked")
    print("  Key components:")
    print("    Item profiles:  feature vectors describing each item")
    print("    User profiles:  aggregate of liked-item features")
    print("    Similarity:     cosine, Euclidean, Pearson, Jaccard, ...")
    print()
    print("  Advantages:")
    print("    No cold-start for items (features available from metadata)")
    print("    No data sparsity problem (no need for other users' data)")
    print("    Transparent: can explain WHY a recommendation was made")
    print()
    print("  Disadvantages:")
    print("    Feature engineering required for each domain")
    print("    Over-specialisation / filter bubble (no serendipity)")
    print("    User cold-start still a problem")


# -- 2. TF-IDF item profiles ---------------------------------------------------
def tfidf_profiles():
    print("\n=== TF-IDF Item Profiles ===")
    # Movie tag corpus
    movies = {
        "Inception":       ["action", "sci-fi", "thriller", "mind-bending", "complex"],
        "Interstellar":    ["sci-fi", "space", "drama", "complex", "time"],
        "The Dark Knight": ["action", "superhero", "thriller", "crime", "complex"],
        "Toy Story":       ["animation", "comedy", "family", "adventure"],
        "Finding Nemo":    ["animation", "family", "adventure", "ocean"],
        "Avatar":          ["sci-fi", "action", "adventure", "fantasy", "visual"],
    }
    movie_names = list(movies.keys())
    M = len(movie_names)

    # Build vocabulary
    vocab = sorted(set(tag for tags in movies.values() for tag in tags))
    V = len(vocab); word_idx = {w: i for i, w in enumerate(vocab)}
    print(f"  Movies: {M}  Vocabulary: {V}")

    # TF (term frequency)
    tf = np.zeros((M, V))
    for i, movie in enumerate(movie_names):
        tags = movies[movie]
        for tag in tags:
            tf[i, word_idx[tag]] += 1
        tf[i] /= len(tags)

    # IDF (inverse document frequency)
    df   = (tf > 0).sum(axis=0)
    idf  = np.log((M + 1) / (df + 1)) + 1
    tfidf = tf * idf

    # L2 normalise
    norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
    tfidf_norm = tfidf / (norms + 1e-10)

    print(f"\n  TF-IDF matrix: {tfidf_norm.shape}")
    print(f"  Top features for 'Inception': "
          f"{[vocab[i] for i in tfidf_norm[0].argsort()[::-1][:4]]}")

    # Cosine similarity
    sim = tfidf_norm @ tfidf_norm.T
    print()
    print(f"  Item-item cosine similarity:")
    print(f"  {'Movie':<20} " + "  ".join(f"{m[:8]:<8}" for m in movie_names))
    for i, mi in enumerate(movie_names):
        row = "  ".join(f"{sim[i,j]:.2f}    " for j in range(M))
        print(f"  {mi:<20} {row}")

    # Recommend for user who liked "Inception"
    liked_idx = 0
    scores    = sim[liked_idx].copy()
    scores[liked_idx] = -1  # exclude itself
    top3 = scores.argsort()[::-1][:3]
    print()
    print(f"  Recommendations for user who liked '{movie_names[liked_idx]}':")
    for rank, idx in enumerate(top3, 1):
        print(f"    {rank}. {movie_names[idx]:<20} (sim={sim[liked_idx, idx]:.3f})")


# -- 3. User profiles ----------------------------------------------------------
def user_profiles():
    print("\n=== User Profile Construction ===")
    print()
    print("  User profile = weighted average of item profiles for liked items")
    print()

    # Item feature matrix (movies as 6-D feature vector)
    items = {
        "Inception":       np.array([1, 1, 1, 0, 0, 0]),   # action,sci-fi,thriller,anim,family,comedy
        "Interstellar":    np.array([0, 1, 0, 0, 0, 0]),
        "The Dark Knight": np.array([1, 0, 1, 0, 0, 0]),
        "Toy Story":       np.array([0, 0, 0, 1, 1, 1]),
        "Finding Nemo":    np.array([0, 0, 0, 1, 1, 0]),
        "Avatar":          np.array([1, 1, 0, 0, 0, 0]),
    }
    features = ["action", "sci-fi", "thriller", "animation", "family", "comedy"]
    item_names = list(items.keys())
    I = np.array([items[m] for m in item_names], dtype=float)

    # Users with ratings (1-5 scale)
    users = {
        "Alice": {"Inception": 5, "Interstellar": 4, "Avatar": 3},
        "Bob":   {"Toy Story": 5, "Finding Nemo": 4},
    }

    for user, ratings in users.items():
        # Weighted user profile
        profile = np.zeros(len(features))
        total_w = 0
        for movie, rating in ratings.items():
            idx    = item_names.index(movie)
            weight = rating - 2.5   # centre around neutral
            if weight > 0:
                profile += weight * I[idx]
                total_w  += abs(weight)
        if total_w > 0:
            profile /= total_w

        # Score unseen items
        seen      = set(ratings.keys())
        unseen    = [(m, I[i] @ profile) for i, m in enumerate(item_names) if m not in seen]
        unseen    = sorted(unseen, key=lambda x: -x[1])

        print(f"  {user} profile: {profile.round(3)}")
        print(f"  Top recommendations:")
        for m, score in unseen[:3]:
            print(f"    {m:<22} score={score:.3f}")
        print()


# -- 4. Hybrid content features ------------------------------------------------
def hybrid_features():
    print("=== Feature Engineering for Content-Based ===")
    print()
    features_by_domain = {
        "Movies":     ["genre", "director", "cast", "tags", "synopsis embedding"],
        "Music":      ["genre", "tempo", "key", "energy", "danceability", "audio embedding"],
        "E-commerce": ["category", "brand", "price", "description embedding", "image embedding"],
        "News":       ["topic", "entities", "publication date", "text embedding"],
        "Jobs":       ["skills", "salary", "location", "company", "JD embedding"],
    }
    for domain, feats in features_by_domain.items():
        print(f"  {domain:<12}: {', '.join(feats)}")

    print()
    print("  Modern approach: pre-trained embedding models")
    print("    Sentence-BERT -> 768-D text embeddings")
    print("    CLIP -> 512-D image+text shared embeddings")
    print("    These replace manual TF-IDF and improve semantic matching")


if __name__ == "__main__":
    overview()
    tfidf_profiles()
    user_profiles()
    hybrid_features()
