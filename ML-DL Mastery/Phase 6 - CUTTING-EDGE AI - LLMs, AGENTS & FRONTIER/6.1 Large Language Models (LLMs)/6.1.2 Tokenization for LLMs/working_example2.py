"""
Working Example 2: Tokenisation for LLMs — BPE from scratch, vocabulary, encoding stats
=========================================================================================
Builds a minimal BPE tokeniser on a small corpus.

Run:  python working_example2.py
"""
from pathlib import Path
from collections import Counter
try:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit("pip install numpy matplotlib")

OUTPUT = Path(__file__).parent / "output"
OUTPUT.mkdir(exist_ok=True)

def get_vocab(corpus):
    vocab = Counter()
    for word in corpus.split():
        vocab[" ".join(list(word)) + " </w>"] += 1
    return vocab

def get_pairs(vocab):
    pairs = Counter()
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
            pairs[(symbols[i], symbols[i+1])] += freq
    return pairs

def merge_vocab(pair, vocab):
    new_vocab = {}
    bigram = " ".join(pair)
    replacement = "".join(pair)
    for word, freq in vocab.items():
        new_word = word.replace(bigram, replacement)
        new_vocab[new_word] = freq
    return new_vocab

def bpe(corpus, n_merges=20):
    vocab = get_vocab(corpus)
    merges = []
    for _ in range(n_merges):
        pairs = get_pairs(vocab)
        if not pairs: break
        best = pairs.most_common(1)[0][0]
        vocab = merge_vocab(best, vocab)
        merges.append(best)
    return vocab, merges

def demo():
    corpus = ("the cat sat on the mat the cat is on the mat a cat sat "
              "lower lower lower lowest lowest the the the")
    vocab, merges = bpe(corpus, n_merges=15)
    print("=== Tokenisation for LLMs ===")
    print(f"  Corpus words: {len(corpus.split())}")
    print(f"  BPE merges applied: {len(merges)}")
    print(f"  Top 5 merges: {merges[:5]}")
    print(f"  Final vocabulary size: {len(vocab)}")

    # Token length distribution in original corpus
    words = corpus.split()
    lengths = [len(w) for w in words]
    avg = np.mean(lengths)
    print(f"  Avg word length: {avg:.2f}")

    plt.figure(figsize=(5, 3))
    plt.hist(lengths, bins=range(1, 12), edgecolor="white")
    plt.xlabel("Word length (chars)"); plt.ylabel("Count"); plt.title("Word Length Distribution")
    plt.tight_layout(); plt.savefig(OUTPUT / "tokenization.png"); plt.close()
    print("  Saved tokenization.png")

if __name__ == "__main__":
    demo()
