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

if __name__ == "__main__":
    demo()
