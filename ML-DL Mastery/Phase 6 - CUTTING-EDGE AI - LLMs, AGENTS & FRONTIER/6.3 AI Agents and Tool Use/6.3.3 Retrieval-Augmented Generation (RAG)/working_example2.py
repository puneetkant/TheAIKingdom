"""
Working Example 2: Retrieval-Augmented Generation (RAG)
TF-IDF document store with cosine similarity retrieval and top-k
context assembly.
Run: python working_example2.py
"""
from pathlib import Path
from collections import Counter
import math

try:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit("pip install numpy matplotlib")

OUTPUT = Path(__file__).parent / "output"
OUTPUT.mkdir(exist_ok=True)

CORPUS = [
    "Machine learning is a subset of artificial intelligence.",
    "Deep learning uses neural networks with many layers.",
    "Transformers are neural networks using attention mechanisms.",
    "Retrieval-augmented generation combines retrieval with generation.",
    "Large language models are trained on massive text corpora.",
    "Vector databases store embeddings for similarity search.",
    "Cosine similarity measures the angle between two vectors.",
    "TF-IDF weights terms by frequency and inverse document frequency.",
    "BERT is a bidirectional transformer pre-trained on masked LM.",
    "GPT generates text by predicting the next token autoregressively.",
]


def tokenize(text):
    return text.lower().split()


def build_tfidf(corpus):
    N = len(corpus)
    tok = [tokenize(d) for d in corpus]
    df = Counter()
    for doc in tok:
        df.update(set(doc))
    vocab = sorted(df.keys())
    v2i = {w: i for i, w in enumerate(vocab)}

    vecs = []
    for doc in tok:
        tf = Counter(doc)
        total = len(doc)
        vec = np.zeros(len(vocab))
        for w, cnt in tf.items():
            if w in v2i:
                tfidf = (cnt / total) * math.log((N + 1) / (df[w] + 1))
                vec[v2i[w]] = tfidf
        norm = np.linalg.norm(vec)
        vecs.append(vec / (norm + 1e-10))
    return np.array(vecs), vocab, v2i, df, N


def query_tfidf(q, vecs, v2i, df, N, top_k=3):
    qtok = tokenize(q)
    qtf = Counter(qtok)
    qvec = np.zeros(len(v2i))
    for w, cnt in qtf.items():
        if w in v2i:
            qvec[v2i[w]] = (cnt / len(qtok)) * math.log((N + 1) / (df.get(w, 0) + 1))
    qnorm = np.linalg.norm(qvec)
    qvec = qvec / (qnorm + 1e-10)
    scores = vecs @ qvec
    top_idx = np.argsort(scores)[::-1][:top_k]
    return top_idx, scores


def demo():
    print("=== RAG: TF-IDF Retrieval ===")
    vecs, vocab, v2i, df, N = build_tfidf(CORPUS)
    print(f"  Corpus: {N} docs, vocab: {len(vocab)} terms")

    queries = [
        "How do transformers work?",
        "What is retrieval augmented generation?",
        "neural networks deep learning",
    ]

    fig, axes = plt.subplots(len(queries), 1, figsize=(12, 10))
    for ax, q in zip(axes, queries):
        top_idx, scores = query_tfidf(q, vecs, v2i, df, N, top_k=5)
        print(f"\n  Query: {q!r}")
        for i in top_idx:
            print(f"    [{scores[i]:.3f}] {CORPUS[i][:60]}")
        # Plot scores
        ax.barh(range(N), scores, color="steelblue", alpha=0.7)
        for i in top_idx:
            ax.barh(i, scores[i], color="tomato")
        ax.set(yticks=range(N),
               yticklabels=[f"D{i}: {CORPUS[i][:40]}..." for i in range(N)],
               xlabel="Cosine Similarity", title=f"Query: {q!r}")
        ax.tick_params(axis="y", labelsize=7)

    plt.tight_layout()
    plt.savefig(OUTPUT / "rag_retrieval.png", dpi=100, bbox_inches="tight")
    plt.close()
    print("\n  Saved rag_retrieval.png")


if __name__ == "__main__":
    demo()
