"""
Working Example 2: Word Embeddings — Word2Vec (CBOW) from scratch + GloVe-like co-occurrence
==============================================================================================
Trains a minimal CBOW model and visualises word vectors.

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

CORPUS = [
    "the cat sat on the mat",
    "the dog chased the cat",
    "machine learning is great",
    "natural language processing",
    "the cat and dog are pets",
    "deep learning uses neural networks",
    "NLP is a branch of AI",
]

def build_vocab(sentences):
    words = [w for s in sentences for w in s.split()]
    vocab = sorted(set(words))
    w2i = {w: i for i, w in enumerate(vocab)}
    return vocab, w2i

def make_cbow_pairs(sentences, w2i, window=2):
    pairs = []
    for sent in sentences:
        words = [w2i[w] for w in sent.split() if w in w2i]
        for i, target in enumerate(words):
            ctx = words[max(0,i-window):i] + words[i+1:i+window+1]
            if ctx: pairs.append((ctx, target))
    return pairs

def train_cbow(vocab, w2i, pairs, dim=8, lr=0.05, epochs=200):
    V = len(vocab)
    W_in  = np.random.randn(V, dim) * 0.01
    W_out = np.random.randn(dim, V) * 0.01
    softmax = lambda x: np.exp(x - x.max()) / np.exp(x - x.max()).sum()
    for ep in range(epochs):
        total_loss = 0
        for ctx_ids, target in pairs:
            h = W_in[ctx_ids].mean(axis=0)
            logits = h @ W_out; probs = softmax(logits)
            loss = -np.log(probs[target] + 1e-9); total_loss += loss
            dout = probs.copy(); dout[target] -= 1
            W_out -= lr * np.outer(h, dout)
            W_in[ctx_ids] -= lr * (W_out @ dout) / len(ctx_ids)
        if (ep+1) % 50 == 0: print(f"  Epoch {ep+1}: loss={total_loss/len(pairs):.4f}")
    return W_in

def demo():
    print("=== CBOW Word Embeddings ===")
    vocab, w2i = build_vocab(CORPUS)
    pairs = make_cbow_pairs(CORPUS, w2i)
    print(f"  Vocab: {len(vocab)}  Training pairs: {len(pairs)}")
    embeddings = train_cbow(vocab, w2i, pairs, dim=8, epochs=200)

    # 2D PCA projection
    cov = np.cov(embeddings.T); vals, vecs = np.linalg.eigh(cov)
    proj = embeddings @ vecs[:, -2:]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(proj[:, 0], proj[:, 1], alpha=0.5)
    for i, word in enumerate(vocab):
        ax.annotate(word, proj[i], fontsize=8)
    ax.set_title("CBOW Embeddings (PCA 2D)")
    plt.tight_layout(); plt.savefig(OUTPUT / "word_embeddings.png"); plt.close()
    print("  Saved word_embeddings.png")

    # Nearest neighbours for "cat"
    target = "cat"
    if target in w2i:
        v = embeddings[w2i[target]]
        sims = {w: np.dot(v, embeddings[w2i[w]]) / (np.linalg.norm(v)*np.linalg.norm(embeddings[w2i[w]])+1e-9) for w in vocab if w != target}
        nn = sorted(sims, key=sims.get, reverse=True)[:5]
        print(f"  Nearest to '{target}':", nn)

def demo_analogy():
    """Word analogy: king - man + woman ≈ queen (via cosine similarity)."""
    print("\n=== Word Analogy Test ===")
    vocab, w2i = build_vocab(CORPUS)
    pairs = make_cbow_pairs(CORPUS, w2i)
    embeddings = train_cbow(vocab, w2i, pairs, dim=8, epochs=150)

    def cosine(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)

    # Find pairs of semantically similar words in our tiny corpus
    print("  Top cosine similarity pairs (excluding self):")
    found = 0
    for i in range(len(vocab)):
        for j in range(i+1, len(vocab)):
            sim = cosine(embeddings[i], embeddings[j])
            if sim > 0.7:
                print(f"    '{vocab[i]}' ~ '{vocab[j]}': {sim:.3f}")
                found += 1
                if found >= 5: break
        if found >= 5: break
    if found == 0:
        print("  (no high-sim pairs after short training)")


def demo_cooccurrence_glove():
    """Build GloVe-style co-occurrence matrix and factorize it."""
    print("\n=== GloVe-style Co-occurrence Matrix ===")
    vocab, w2i = build_vocab(CORPUS)
    V = len(vocab); window = 2
    X_coo = np.zeros((V, V))
    for sent in CORPUS:
        words = [w for w in sent.split() if w in w2i]
        ids = [w2i[w] for w in words]
        for i, wi in enumerate(ids):
            for d in range(1, window+1):
                if i + d < len(ids):
                    X_coo[wi, ids[i+d]] += 1/d
                    X_coo[ids[i+d], wi] += 1/d
    # SVD-based factorization
    U, S, Vt = np.linalg.svd(X_coo)
    dim = 4
    emb = U[:, :dim] * S[:dim]
    print(f"  Vocab: {V}  Co-occurrence matrix density: {(X_coo>0).mean():.2f}")
    print(f"  GloVe-like embeddings shape: {emb.shape}")
    # Most similar to 'the'
    if 'the' in w2i:
        v = emb[w2i['the']]
        sims = {w: np.dot(v, emb[w2i[w]]) for w in vocab if w != 'the'}
        print(f"  Words most similar to 'the': {sorted(sims, key=sims.get, reverse=True)[:3]}")


if __name__ == "__main__":
    demo()
    demo_analogy()
    demo_cooccurrence_glove()
