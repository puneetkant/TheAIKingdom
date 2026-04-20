"""
Working Example 2: Subword Tokenization — BPE from scratch
============================================================
Implements Byte-Pair Encoding merge algorithm on a toy corpus.

Run:  python working_example2.py
"""
from pathlib import Path
from collections import Counter, defaultdict
import re

OUTPUT = Path(__file__).parent / "output"
OUTPUT.mkdir(exist_ok=True)

CORPUS = ["low lower lowest", "newer new news", "wider wide", "highest high", "lower lowest"]

def get_vocab(corpus):
    vocab = Counter()
    for sentence in corpus:
        for word in sentence.split():
            chars = " ".join(list(word)) + " </w>"
            vocab[chars] += 1
    return vocab

def get_stats(vocab):
    pairs = defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
            pairs[(symbols[i], symbols[i+1])] += freq
    return pairs

def merge_vocab(pair, vocab):
    new_vocab = {}
    pattern = re.escape(" ".join(pair))
    merged = "".join(pair)
    for word in vocab:
        new_word = re.sub(pattern, merged, word)
        new_vocab[new_word] = vocab[word]
    return new_vocab

def bpe(corpus, n_merges=10):
    vocab = get_vocab(corpus)
    merges = []
    for i in range(n_merges):
        stats = get_stats(vocab)
        if not stats: break
        best = max(stats, key=stats.get)
        vocab = merge_vocab(best, vocab)
        merges.append(best)
        print(f"  Merge {i+1}: {best[0]} + {best[1]} -> {''.join(best)}")
    return vocab, merges

def demo():
    print("=== Byte-Pair Encoding (BPE) ===")
    print("Corpus:", CORPUS)
    vocab, merges = bpe(CORPUS, n_merges=12)
    print("\nFinal vocab tokens:")
    tokens = set()
    for word in vocab:
        tokens.update(word.split())
    for t in sorted(tokens):
        print(f"  {t}")
    (OUTPUT / "bpe_merges.txt").write_text("\n".join(f"{a} {b}" for a,b in merges))
    print(f"\nSaved {len(merges)} BPE merges to bpe_merges.txt")

def demo_tokenize_word(word, merges):
    """Apply learned BPE merges to tokenize an unseen word."""
    tokens = list(word) + ["</w>"]
    for merge_a, merge_b in merges:
        merged = merge_a + merge_b
        i = 0
        new_tokens = []
        while i < len(tokens):
            if i < len(tokens)-1 and tokens[i] == merge_a and tokens[i+1] == merge_b:
                new_tokens.append(merged); i += 2
            else:
                new_tokens.append(tokens[i]); i += 1
        tokens = new_tokens
    return tokens


def demo_apply_bpe():
    """Apply BPE merges to new words not in training corpus."""
    print("\n=== Applying BPE to New Words ===")
    _, merges = bpe(CORPUS, n_merges=12)
    test_words = ["lower", "newest", "widen", "unknown"]
    for w in test_words:
        toks = demo_tokenize_word(w, merges)
        print(f"  '{w}' -> {toks}")


def demo_sentencepiece_comparison():
    """Compare BPE characteristics with SentencePiece / WordPiece design choices."""
    print("\n=== Tokenization Algorithm Comparison ===")
    rows = [
        ("BPE",          "Frequency-based merge",       "GPT-2, Llama, Mistral"),
        ("WordPiece",    "Likelihood-maximising merge", "BERT, DistilBERT"),
        ("SentencePiece","Language-agnostic, unigram",  "T5, ALBERT, mBERT"),
        ("Unigram LM",   "Prune from large vocab",      "XLNet, Gemma"),
        ("Char-level",   "Every character is a token",  "ByT5"),
    ]
    print(f"  {'Algorithm':18s}  {'Strategy':35s}  {'Used in'}")
    print("  " + "-"*75)
    for r in rows:
        print(f"  {r[0]:18s}  {r[1]:35s}  {r[2]}")


if __name__ == "__main__":
    demo()
    demo_apply_bpe()
    demo_sentencepiece_comparison()
