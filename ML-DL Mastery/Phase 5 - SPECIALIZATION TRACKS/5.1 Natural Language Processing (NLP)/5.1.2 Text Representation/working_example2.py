"""
Working Example 2: Text Representation — Bag of Words, TF-IDF, N-grams
=======================================================================
Sklearn vectorizers on mini newsgroup corpus.

Run:  python working_example2.py
"""
from pathlib import Path
try:
    import numpy as np
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    from sklearn.datasets import fetch_20newsgroups
except ImportError:
    raise SystemExit("pip install scikit-learn numpy")

OUTPUT = Path(__file__).parent / "output"
OUTPUT.mkdir(exist_ok=True)

CORPUS = [
    "the cat sat on the mat",
    "the dog chased the cat",
    "machine learning is great for NLP",
    "natural language processing uses machine learning",
    "the cat and the dog are pets",
]

def demo():
    print("=== Text Representation ===")

    # Bag of Words
    cv = CountVectorizer(max_features=20)
    bow = cv.fit_transform(CORPUS).toarray()
    print("\n--- Bag of Words ---")
    print("Vocabulary:", list(cv.vocabulary_.keys())[:10])
    print("BoW shape:", bow.shape)

    # TF-IDF
    tfidf = TfidfVectorizer(max_features=20)
    X = tfidf.fit_transform(CORPUS).toarray()
    print("\n--- TF-IDF (doc 2) ---")
    idxs = X[2].argsort()[::-1][:5]
    feats = np.array(tfidf.get_feature_names_out())
    for i in idxs:
        print(f"  {feats[i]:15s}: {X[2,i]:.4f}")

    # Bigrams
    bg = CountVectorizer(ngram_range=(1,2), max_features=20)
    Xb = bg.fit_transform(CORPUS).toarray()
    print(f"\n--- Bigrams: {Xb.shape[1]} features ---")
    print("Sample:", list(bg.vocabulary_.keys())[:8])

    # Save feature matrix
    header = ",".join(tfidf.get_feature_names_out())
    np.savetxt(OUTPUT / "tfidf_matrix.csv", X, delimiter=",", header=header, comments="")
    print("\nSaved tfidf_matrix.csv")

if __name__ == "__main__":
    demo()
