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

def demo_subword_fertility():
    """Fertility = avg tokens per word. Lower = more efficient tokenisation."""
    print("\n=== Subword Fertility ===")
    corpus = ("the cat sat on the mat the cat is on the mat "
              "lower lower lower lowest lowest the the the")
    words = corpus.split()
    # Char-level: each character is a token
    char_tokens = sum(len(w) for w in words)
    char_fertility = char_tokens / len(words)
    # Word-level: each word is a token (fertility = 1 by definition)
    word_fertility = 1.0
    # BPE-level: run BPE and count tokens from vocabulary entries
    vocab, merges = bpe(corpus, n_merges=15)
    bpe_token_count = sum(len(k.split()) * v for k, v in vocab.items())
    bpe_fertility = bpe_token_count / len(words)
    print(f"  Char-level  fertility: {char_fertility:.2f} tokens/word")
    print(f"  BPE         fertility: {bpe_fertility:.2f} tokens/word")
    print(f"  Word-level  fertility: {word_fertility:.2f} tokens/word")

    methods = ["Char", "BPE", "Word"]
    values  = [char_fertility, bpe_fertility, word_fertility]
    plt.figure(figsize=(5, 3))
    plt.bar(methods, values, color=["#e74c3c", "#3498db", "#2ecc71"], edgecolor="white")
    plt.ylabel("Fertility (tokens / word)")
    plt.title("Tokenisation Fertility Comparison")
    plt.tight_layout()
    plt.savefig(OUTPUT / "fertility.png", dpi=100); plt.close()
    print("  Saved fertility.png")


def demo_vocab_coverage():
    """What fraction of test words appear unsplit in the BPE vocab?"""
    print("\n=== Vocabulary Coverage ===")
    train_corpus = ("the cat sat on the mat lower lower lowest the the ")
    test_words   = ["the", "cat", "sat", "mat", "lower", "lowest",
                    "running", "jumped", "unknown", "rare"]
    vocab, _ = bpe(train_corpus, n_merges=10)
    # Flatten vocabulary tokens (remove </w> marker for comparison)
    known = {k.replace(" </w>", "").replace(" ", "") for k in vocab}
    covered = [w for w in test_words if w in known]
    oov     = [w for w in test_words if w not in known]
    coverage = len(covered) / len(test_words)
    print(f"  Test words:   {test_words}")
    print(f"  Covered ({len(covered)}):  {covered}")
    print(f"  OOV     ({len(oov)}):  {oov}")
    print(f"  Coverage: {coverage:.0%}")


if __name__ == "__main__":
    demo()
    demo_subword_fertility()
    demo_vocab_coverage()
