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

def demo_bow_tfidf_compare():
    """Compare CountVectorizer vs TfidfVectorizer with LogisticRegression on 20newsgroups."""
    print("\n=== BoW vs TF-IDF Classification Comparison ===")
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    cats = ["sci.space", "rec.sport.hockey", "talk.politics.guns", "comp.graphics"]
    train = fetch_20newsgroups(subset="train", categories=cats,
                               remove=("headers", "footers", "quotes"))
    test  = fetch_20newsgroups(subset="test",  categories=cats,
                               remove=("headers", "footers", "quotes"))

    saved_vect = None
    saved_clf  = None

    for name, vect in [
        ("BoW (CountVectorizer)", CountVectorizer(max_features=10000)),
        ("TF-IDF", TfidfVectorizer(max_features=10000, sublinear_tf=True)),
    ]:
        X_tr = vect.fit_transform(train.data)
        X_te = vect.transform(test.data)
        clf = LogisticRegression(max_iter=500, C=1.0, solver="lbfgs")
        clf.fit(X_tr, train.target)
        acc = accuracy_score(test.target, clf.predict(X_te))
        print(f"  {name}: {acc:.4f}")
        if name == "TF-IDF":
            saved_vect = vect
            saved_clf  = clf

    feat_names = np.array(saved_vect.get_feature_names_out())
    print("\n  Top 5 TF-IDF features per class:")
    for ci, cat in enumerate(cats):
        top5 = np.argsort(saved_clf.coef_[ci])[::-1][:5]
        print(f"    {cat.split('.')[-1]:20s}: {list(feat_names[top5])}")


def demo_cooccurrence_pmi():
    """Build word co-occurrence matrix with PPMI; find top similar word pairs by cosine sim."""
    print("\n=== Word Co-occurrence + PPMI ===")

    sentences = [
        "the cat sat on the mat",
        "the dog chased the cat",
        "machine learning is great",
        "natural language processing is fun",
        "the cat and the dog play",
        "deep learning beats machine learning",
        "language models use neural networks",
        "the mat is on the floor",
    ]

    tokens_list = [s.lower().split() for s in sentences]
    vocab = sorted(set(w for s in tokens_list for w in s))
    w2i = {w: i for i, w in enumerate(vocab)}
    V = len(vocab)
    window = 2

    cooc = np.zeros((V, V), dtype=float)
    for toks in tokens_list:
        for ci, cword in enumerate(toks):
            for offset in range(1, window + 1):
                for sign in (-1, 1):
                    ni = ci + sign * offset
                    if 0 <= ni < len(toks):
                        cooc[w2i[cword], w2i[toks[ni]]] += 1.0

    total = cooc.sum() + 1e-10
    pw = cooc.sum(axis=1) / total
    pc = cooc.sum(axis=0) / total
    expected = np.outer(pw, pc) * total + 1e-10
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.where(cooc > 0, np.log2(cooc / expected), 0.0)
    ppmi = np.maximum(pmi, 0.0)
    np.fill_diagonal(ppmi, 0.0)

    norms = np.linalg.norm(ppmi, axis=1, keepdims=True) + 1e-10
    ppmi_n = ppmi / norms
    sim_mat = ppmi_n @ ppmi_n.T

    pairs = []
    for i in range(V):
        for j in range(i + 1, V):
            if sim_mat[i, j] > 0:
                pairs.append((sim_mat[i, j], vocab[i], vocab[j]))
    pairs.sort(reverse=True)

    print("  Top 3 similar word pairs (PPMI cosine sim):")
    for sim, w1, w2 in pairs[:3]:
        print(f"    '{w1}' <-> '{w2}': {sim:.4f}")


def demo_lsa():
    """TF-IDF + TruncatedSVD (LSA) on 20newsgroups: top terms per dim, word similarity."""
    print("\n=== Latent Semantic Analysis (LSA / TruncatedSVD) ===")
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import Normalizer

    cats = ["sci.space", "rec.sport.hockey", "talk.politics.guns", "comp.graphics"]
    data = fetch_20newsgroups(subset="train", categories=cats,
                              remove=("headers", "footers", "quotes"))

    vect = TfidfVectorizer(max_features=20000, sublinear_tf=True, min_df=2)
    X = vect.fit_transform(data.data)
    feat_names = np.array(vect.get_feature_names_out())

    svd = TruncatedSVD(n_components=100, random_state=42)
    X_lsa = svd.fit_transform(X)
    X_lsa = Normalizer(copy=False).fit_transform(X_lsa)

    print("  Top 5 terms for first 4 latent dimensions:")
    for dim in range(4):
        top5 = np.argsort(np.abs(svd.components_[dim]))[::-1][:5]
        print(f"    Dim {dim + 1}: {list(feat_names[top5])}")

    # Word vectors from right singular vectors (columns of V^T = components_)
    target = ["space", "nasa"]
    vecs = {}
    for word in target:
        idxs = np.where(feat_names == word)[0]
        if idxs.size > 0:
            vec = svd.components_[:, idxs[0]]
            vecs[word] = vec / (np.linalg.norm(vec) + 1e-10)

    if len(vecs) == 2:
        sim = float(vecs["space"] @ vecs["nasa"])
        print(f"\n  Cosine('space', 'nasa') in LSA space: {sim:.4f}")
    else:
        missing = [w for w in target if w not in vecs]
        print(f"  Words not found in vocab: {missing}")


if __name__ == "__main__":
    demo()
    demo_bow_tfidf_compare()
    demo_cooccurrence_pmi()
    demo_lsa()
