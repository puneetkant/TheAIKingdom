"""
Working Example: Tokenisation for LLMs
Covers BPE, WordPiece, SentencePiece, tokenisation effects,
and byte-level tokenisation.
"""
import numpy as np
import os, re
from collections import Counter, defaultdict

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_tokenization")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── 1. Why tokenisation matters ───────────────────────────────────────────────
def tokenization_intro():
    print("=== Tokenisation for LLMs ===")
    print()
    print("  LLMs operate on token sequences, not raw text or characters.")
    print("  Tokeniser maps strings ↔ integer IDs (vocabulary).")
    print()
    print("  Tokeniser comparison:")
    info = [
        ("GPT-2/3/4",    "cl100k_base / p50k",  "BPE",         50_257,  "Byte-level BPE"),
        ("GPT-4o",       "o200k_base",           "BPE",        200_000,  "Extended multilingual"),
        ("LLaMA-3",      "tiktoken",             "BPE",        128_256,  "cl100k_base extended"),
        ("BERT/RoBERTa", "WordPiece",            "WordPiece",   30_522,  "CLS, SEP tokens"),
        ("T5",           "SentencePiece",        "Unigram",     32_100,  "Subword; sentencepiece"),
        ("Gemma",        "SentencePiece",        "BPE",        256_000,  "Byte-fallback"),
    ]
    print(f"  {'Model':<14} {'Tokeniser':<16} {'Algorithm':<12} {'Vocab':<10} {'Notes'}")
    for m, tok, alg, vocab, notes in info:
        print(f"  {m:<14} {tok:<16} {alg:<12} {vocab:<10,} {notes}")


# ── 2. Byte-Pair Encoding (BPE) ───────────────────────────────────────────────
def bpe_demo():
    print("\n=== Byte-Pair Encoding (BPE) ===")
    print()
    print("  Algorithm:")
    print("    1. Start with character-level vocabulary")
    print("    2. Count all adjacent pair frequencies")
    print("    3. Merge most frequent pair → new token")
    print("    4. Repeat for N merge steps")
    print()

    corpus = ["low", "lower", "newest", "widest", "low", "low"]
    word_freq = Counter(corpus)

    def get_vocab(word_freq):
        vocab = {}
        for word, freq in word_freq.items():
            chars = list(word) + ["</w>"]
            vocab[" ".join(chars)] = freq
        return vocab

    def get_pairs(vocab):
        pairs = defaultdict(int)
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols)-1):
                pairs[(symbols[i], symbols[i+1])] += freq
        return pairs

    def merge_vocab(pair, vocab_in):
        bigram = re.escape(" ".join(pair))
        vocab_out = {}
        for word in vocab_in:
            new_word = re.sub(r"(?<!\S)" + bigram + r"(?!\S)",
                              "".join(pair), word)
            vocab_out[new_word] = vocab_in[word]
        return vocab_out

    vocab = get_vocab(word_freq)
    print("  Initial vocab:", list(vocab.keys()))
    print()
    for i in range(6):
        pairs = get_pairs(vocab)
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        vocab = merge_vocab(best, vocab)
        print(f"  Merge {i+1}: {best!r} → {''.join(best)!r}  (freq={pairs[best]})")

    print()
    print("  Final vocab segments:", list(vocab.keys()))


# ── 3. WordPiece tokenisation ─────────────────────────────────────────────────
def wordpiece_demo():
    print("\n=== WordPiece Tokenisation ===")
    print()
    print("  Used by BERT.  Key difference from BPE:")
    print("    BPE:       merges most frequent pair")
    print("    WordPiece: merges pair maximising likelihood of corpus")
    print("               ≈ maximise log P(corpus) = Σ log P(token)")
    print()
    # Simple illustrative tokenisation
    vocab = {"[UNK]", "un", "##happy", "##ness", "happy", "ness", "unhappy", "good"}
    examples = [
        ("unhappy",  ["un", "##happy"]),
        ("happiness",["happy", "##ness"]),
        ("sadness",  ["[UNK]"]),
    ]
    print(f"  {'Word':<12} {'Tokenised'}")
    for word, tokens in examples:
        print(f"  {word:<12} {tokens}")
    print()
    print("  ## prefix marks continuation subwords (not word starts)")
    print("  [CLS] and [SEP] are special tokens added at sequence level")


# ── 4. Tokenisation effects ───────────────────────────────────────────────────
def tokenization_effects():
    print("\n=== Tokenisation Effects and Gotchas ===")
    print()
    print("  Common tokenisation anomalies (cl100k_base / GPT-4 tokeniser):")
    issues = [
        ("Arithmetic",     "'9.11 > 9.9'  — tokens split badly: '9', '.', '11'"),
        ("Code indentation","Python: 4 spaces vs 1 tab token; affects continuation"),
        ("Non-English",    "Non-latin scripts use far more tokens; 'efficiency gap'"),
        ("Numbers",        "Large numbers tokenised digit-by-digit → poor maths"),
        ("Repeated chars", "'aaaaaaa' → n separate tokens; no compression"),
        ("Leading spaces", "' hello' ≠ 'hello'; whitespace matters"),
    ]
    for issue, desc in issues:
        print(f"  {issue:<20} {desc}")
    print()

    # Simulate a simple character-to-token count analysis
    texts = [
        ("English (simple)", "The quick brown fox jumps over the lazy dog"),
        ("Code (Python)",    "def factorial(n): return 1 if n<=1 else n*factorial(n-1)"),
        ("Numbers",          "123456789012345678901234567890"),
        ("Japanese (rough)", "人工知能は未来を変える技術です。"),
    ]
    print(f"  {'Text type':<22} {'Chars':<8} {'~Tokens':<10} {'Chars/Token'}")
    ratios = {"English (simple)": 4.0, "Code (Python)": 3.5,
              "Numbers": 2.5, "Japanese (rough)": 1.5}
    for name, text in texts:
        chars = len(text)
        ratio = ratios[name]
        tokens = int(chars / ratio)
        print(f"  {name:<22} {chars:<8} {tokens:<10} ~{ratio:.1f}")
    print()
    print("  Rule of thumb: 1 token ≈ 4 chars English ≈ ¾ word")

    print()
    print("  Context window cost:")
    print("    GPT-4o: 1M tokens input = ~$5.00")
    print("    LLaMA-3-8B (self-hosted): ~$0.10/1M tokens")
    print("    → Tokenisation efficiency directly affects operational cost")


if __name__ == "__main__":
    tokenization_intro()
    bpe_demo()
    wordpiece_demo()
    tokenization_effects()
