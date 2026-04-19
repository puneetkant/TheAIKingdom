"""
Working Example: Word Embeddings
Covers Word2Vec (skip-gram), GloVe concepts, fastText, and analogies.
Word2Vec implemented from scratch with numpy.
"""
import numpy as np
from collections import Counter, defaultdict
import os, math, re, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_embeddings")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -- toy corpus ----------------------------------------------------------------
CORPUS_RAW = """
the king rules the kingdom and the queen also rules
the man is a prince and the woman is a princess
the king and queen have power
the dog barks and the cat meows
the prince became a king after the old king died
words and language capture meaning
machine learning models learn word representations
the embedding captures semantic meaning
deep learning transforms natural language processing
""".strip()


def build_vocab(corpus: str, min_freq=1):
    tokens = corpus.lower().split()
    cnt    = Counter(tokens)
    vocab  = {w: i for i, (w, c) in enumerate(cnt.items()) if c >= min_freq}
    return tokens, vocab, {i: w for w, i in vocab.items()}


def sigmoid(x): return 1 / (1 + np.exp(-x.clip(-10, 10)))


# -- Word2Vec Skip-gram (negative sampling) ------------------------------------
class Word2Vec:
    """Simplified Skip-gram with negative sampling."""
    def __init__(self, vocab_size, dim=10, rng=None):
        rng = rng or np.random.default_rng(0)
        s   = 0.5 / dim**0.5
        self.W_in  = rng.uniform(-s, s, (vocab_size, dim))   # centre vectors
        self.W_out = rng.uniform(-s, s, (vocab_size, dim))   # context vectors
        self.vocab_size = vocab_size

    def train(self, token_ids, window=2, n_neg=5, lr=0.025, epochs=30, rng=None):
        rng = rng or np.random.default_rng(1)
        ids  = list(range(self.vocab_size))
        losses = []
        for ep in range(epochs):
            ep_loss = 0; n_updates = 0
            for t in range(len(token_ids)):
                ci = token_ids[t]
                start = max(0, t - window); end = min(len(token_ids), t + window + 1)
                ctx_ids = [token_ids[j] for j in range(start, end) if j != t]
                for oi in ctx_ids:
                    # Positive sample
                    v_c = self.W_in[ci]
                    v_o = self.W_out[oi]
                    pos_score = sigmoid(v_c @ v_o)
                    loss = -math.log(pos_score + 1e-9)
                    # Grad
                    grad_c  = (pos_score - 1) * v_o
                    grad_o  = (pos_score - 1) * v_c
                    # Negative samples
                    negs = rng.choice(ids, n_neg, replace=False)
                    for ni in negs:
                        v_n = self.W_out[ni]
                        neg_score = sigmoid(-(v_c @ v_n))
                        loss += -math.log(neg_score + 1e-9)
                        grad_c  += (1 - neg_score) * v_n
                        grad_neg = (1 - neg_score) * v_c
                        self.W_out[ni] -= lr * np.clip(grad_neg, -1, 1)
                    self.W_in[ci]  -= lr * np.clip(grad_c, -1, 1)
                    self.W_out[oi] -= lr * np.clip(grad_o, -1, 1)
                    ep_loss += loss; n_updates += 1
            losses.append(ep_loss / max(n_updates, 1))
        return losses

    def embedding(self, word_id):
        """Mean of input and output vectors."""
        return (self.W_in[word_id] + self.W_out[word_id]) / 2

    def most_similar(self, word_id, vocab_size, top_k=5):
        v = self.embedding(word_id)
        sims = []
        for i in range(vocab_size):
            u    = self.embedding(i)
            cos  = np.dot(v, u) / (np.linalg.norm(v) * np.linalg.norm(u) + 1e-9)
            sims.append((i, cos))
        return sorted(sims, key=lambda x: -x[1])[1:top_k+1]


def word2vec_demo():
    print("=== Word2Vec (Skip-gram) from Scratch ===")
    tokens, vocab, idx2w = build_vocab(CORPUS_RAW)
    token_ids = [vocab[t] for t in tokens]
    print(f"  Corpus: {len(tokens)} tokens, vocab: {len(vocab)}")

    rng   = np.random.default_rng(42)
    model = Word2Vec(len(vocab), dim=15, rng=rng)
    losses = model.train(token_ids, window=2, n_neg=4, lr=0.02, epochs=60, rng=rng)

    print(f"  Loss: {losses[0]:.4f} -> {losses[-1]:.4f}")
    print()

    # Most similar words
    for query in ["king", "man", "dog", "learning"]:
        if query not in vocab: continue
        qid   = vocab[query]
        sims  = model.most_similar(qid, len(vocab), top_k=4)
        words = [(idx2w[i], f"{s:.3f}") for i, s in sims]
        print(f"  Similar to '{query}': {words}")

    # Word analogy: king - man + woman ~= queen
    if all(w in vocab for w in ["king", "man", "woman", "queen"]):
        v_king  = model.embedding(vocab["king"])
        v_man   = model.embedding(vocab["man"])
        v_woman = model.embedding(vocab["woman"])
        target  = v_king - v_man + v_woman
        best_i, best_sim = -1, -1
        for i in range(len(vocab)):
            if i in [vocab["king"], vocab["man"], vocab["woman"]]: continue
            u   = model.embedding(i)
            cos = np.dot(target, u) / (np.linalg.norm(target) * np.linalg.norm(u) + 1e-9)
            if cos > best_sim:
                best_sim, best_i = cos, i
        print(f"\n  Analogy: king - man + woman = {idx2w[best_i]} (cos={best_sim:.3f})")

    # 2D PCA of embeddings
    E = np.stack([model.embedding(i) for i in range(len(vocab))])
    E -= E.mean(0); U, S, Vt = np.linalg.svd(E, full_matrices=False)
    E2d = U[:, :2] * S[:2]

    words_to_plot = [w for w in ["king", "queen", "man", "woman", "prince",
                                  "princess", "dog", "cat"] if w in vocab]
    fig, ax = plt.subplots(figsize=(7, 6))
    for w in words_to_plot:
        i = vocab[w]; x, y = E2d[i]
        ax.scatter(x, y, s=50)
        ax.annotate(w, (x, y), fontsize=9)
    ax.set(title="Word Embeddings (PCA 2D)", xlabel="PC1", ylabel="PC2")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "word_embeddings_pca.png")
    plt.savefig(path, dpi=90); plt.close()
    print(f"\n  2D embedding plot: {path}")


# -- GloVe theory --------------------------------------------------------------
def glove_theory():
    print("\n=== GloVe (Global Vectors for Word Representation) ===")
    print("  Key insight: exploit global word co-occurrence statistics")
    print()
    print("  Objective: minimise  Sigma f(X_ij) · (w_i · w_j + b_i + b_j - log X_ij)²")
    print("             i,j")
    print()
    print("  where X_ij  = co-occurrence count of words i and j")
    print("        w_i   = centre word vector")
    print("        w_j   = context word vector")
    print("        f(x)  = weighting function  (caps very common pairs)")
    print()
    print("  Weighting:  f(x) = (x/x_max)^alpha  if x < x_max,  else 1")
    print("                     alpha = 0.75, x_max = 100 (typical)")
    print()
    print("  Final embedding: w + w  (average of both vectors)")
    print()
    print("  Pre-trained GloVe: glove.6B (6B tokens, dim 50/100/200/300)")
    print("  Usage: np.load('glove.6B.100d.txt') -> dict {word: vector}")


# -- fastText ------------------------------------------------------------------
def fasttext_overview():
    print("\n=== fastText (Subword Embeddings) ===")
    print("  Problem with Word2Vec: OOV (out-of-vocabulary) words get no vector")
    print("  Solution: represent words as bag of character n-grams")
    print()
    word = "learning"
    ngrams = [f"<{word[i:i+3]}>" for i in range(len(word)-2)]
    ngrams = [f"<{word[i:i+n]}>" for n in range(3, 6+1) for i in range(len(word)-n+2)]
    print(f"  Character n-grams of '{word}' (n=3..6):")
    shown = sorted(set(ngrams))[:12]
    print(f"    {shown}")
    print()
    print("  Embedding(word) = mean of all subword n-gram embeddings")
    print("  OOV handling: compose from subword n-grams (no lookup needed)")
    print("  Better for morphologically rich languages (Finnish, Turkish)")
    print()
    print("  Pre-trained: fasttext.cc/en -> 2M words, 300-dim")

    # Simulate OOV robustness
    known   = "learn"
    unknown = "unlearning"
    print(f"\n  OOV demo: '{unknown}' shares subwords with '{known}'")
    def get_ngrams(w, lo=3, hi=5):
        w = f"<{w}>"; return {w[i:i+n] for n in range(lo, hi+1) for i in range(len(w)-n+1)}
    shared = get_ngrams(known) & get_ngrams(unknown)
    print(f"  Shared subwords: {sorted(shared)}")


# -- Embedding comparison -------------------------------------------------------
def embedding_comparison():
    print("\n=== Word Embedding Comparison ===")
    rows = [
        ("Method",        "Training",      "OOV", "Dim",       "Context"),
        ("Word2Vec CBoW", "Context->word",  "No",  "50-300",    "Fixed window"),
        ("Word2Vec SG",   "Word->context",  "No",  "50-300",    "Fixed window"),
        ("GloVe",         "Global co-occ", "No",  "50-300",    "Global matrix"),
        ("fastText",      "Subword SG",    "Yes", "300",       "Fixed window"),
        ("ELMo",          "Bi-LSTM LM",    "Yes", "1024",      "Full sentence"),
        ("BERT",          "Masked LM",     "Yes", "768-1024",  "Full document"),
        ("GPT",           "Autoregressive","Yes", "768-1600",  "Full context"),
    ]
    w = [14, 16, 5, 10, 14]
    print(f"  {'Method':<14} {'Training':<16} {'OOV':<5} {'Dim':<10} Context")
    print(f"  {'-'*14} {'-'*16} {'-'*5} {'-'*10} {'-'*14}")
    for r in rows[1:]:
        print(f"  {r[0]:<14} {r[1]:<16} {r[2]:<5} {r[3]:<10} {r[4]}")


if __name__ == "__main__":
    word2vec_demo()
    glove_theory()
    fasttext_overview()
    embedding_comparison()
