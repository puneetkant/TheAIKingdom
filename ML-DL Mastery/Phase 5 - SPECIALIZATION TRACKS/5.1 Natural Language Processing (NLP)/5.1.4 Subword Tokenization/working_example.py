"""
Working Example: Subword Tokenization
Covers BPE (Byte-Pair Encoding), WordPiece, Unigram LM, and SentencePiece
concepts — BPE implemented from scratch.
"""
import re, collections
from collections import Counter, defaultdict
from typing import Dict, List, Tuple


# ── 1. Why Subword Tokenization? ──────────────────────────────────────────────
def motivation():
    print("=== Why Subword Tokenization? ===")
    issues = [
        ("Word-level", "Large vocab (50k+)", "OOV for rare/new words, no morphology"),
        ("Char-level", "Tiny vocab (256)",    "Long sequences, no word-level semantics"),
        ("Subword",    "Mid vocab (32-64k)",  "OOV-free, morphology, efficiency"),
    ]
    print(f"  {'Method':<12} {'Vocab':<22} {'Issues / Strengths'}")
    print(f"  {'─'*12} {'─'*22} {'─'*40}")
    for m, v, i in issues:
        print(f"  {m:<12} {v:<22} {i}")

    print()
    print("  Example: 'unbelievably' (OOV for word-level)")
    print("    BPE might encode: ['un', '##believ', '##ably']")
    print("    Or:               ['_un', '_believe', '_ably']")
    print()
    print("  Used by: GPT (BPE), BERT (WordPiece), T5/LLaMA (SentencePiece/BPE)")


# ── 2. BPE from scratch ───────────────────────────────────────────────────────
def get_vocab(corpus: List[str]) -> Dict[str, int]:
    """Convert corpus to character-level vocabulary with end-of-word marker."""
    vocab = Counter()
    for word in corpus:
        chars = " ".join(list(word)) + " </w>"
        vocab[chars] += 1
    return dict(vocab)


def get_pairs(vocab: Dict[str, int]) -> Counter:
    """Count all adjacent symbol pairs."""
    pairs = Counter()
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i+1])] += freq
    return pairs


def merge_vocab(pair: Tuple[str, str], vocab: Dict[str, int]) -> Dict[str, int]:
    """Merge the most frequent pair in the vocabulary."""
    bigram = re.escape(" ".join(pair))
    pattern = re.compile(r"(?<!\S)" + bigram + r"(?!\S)")
    new_vocab = {}
    for word in vocab:
        new_word = pattern.sub("".join(pair), word)
        new_vocab[new_word] = vocab[word]
    return new_vocab


def bpe_train(corpus: List[str], n_merges: int = 20):
    """Train BPE on a list of words."""
    vocab = get_vocab(corpus)
    merges = []

    print(f"  Initial vocab ({len(vocab)} entries):")
    for k, v in list(vocab.items())[:4]:
        print(f"    '{k}' (×{v})")

    for i in range(n_merges):
        pairs = get_pairs(vocab)
        if not pairs: break
        best  = max(pairs, key=pairs.get)
        vocab = merge_vocab(best, vocab)
        merges.append(best)
        if i < 5 or i == n_merges - 1:
            print(f"  Merge #{i+1:>2}: {best[0]} + {best[1]} → {''.join(best)} "
                  f"(freq={pairs[best]})")

    return vocab, merges


def bpe_encode(word: str, merges: List[Tuple[str, str]]) -> List[str]:
    """Apply learned BPE merges to a word."""
    chars = list(word) + ["</w>"]
    for pair in merges:
        i = 0
        new_chars = []
        while i < len(chars):
            if i < len(chars) - 1 and (chars[i], chars[i+1]) == pair:
                new_chars.append("".join(pair))
                i += 2
            else:
                new_chars.append(chars[i])
                i += 1
        chars = new_chars
    return chars


def bpe_demo():
    print("\n=== Byte-Pair Encoding (BPE) from Scratch ===")
    corpus = (
        ["low"] * 5 + ["lower"] * 2 + ["newest"] * 6 +
        ["wider"] * 3 + ["low"] * 3 + ["new"] * 4 +
        ["newer"] * 2 + ["widest"] * 1
    )
    print(f"  Training corpus: {len(corpus)} words, "
          f"{len(set(corpus))} unique")

    vocab, merges = bpe_train(corpus, n_merges=15)

    print(f"\n  Final vocab tokens:")
    tokens = set()
    for k in vocab:
        tokens.update(k.split())
    print(f"    {sorted(tokens)}")

    print(f"\n  BPE encoding examples:")
    for w in ["low", "newer", "widest", "lowest"]:
        enc = bpe_encode(w, merges)
        print(f"    '{w}' → {enc}")


# ── 3. WordPiece ──────────────────────────────────────────────────────────────
def wordpiece_overview():
    print("\n=== WordPiece (BERT-style) ===")
    print("  Similar to BPE but maximises likelihood of training data")
    print("  Score(pair) = freq(pair) / (freq(A) × freq(B))")
    print()
    print("  Key difference from BPE:")
    print("    BPE:       merge most *frequent* pair")
    print("    WordPiece: merge pair that maximises language model likelihood")
    print()
    print("  Continuation prefix: '##' marks token continues previous token")
    examples = [
        ("unaffable",   ["un", "##aff", "##able"]),
        ("tokenization",["token", "##ization"]),
        ("huggingface", ["hugging", "##face"]),
        ("supercalifragilisticexpialidocious", ["super", "##cal", "##if", "##ragi", "##listic", "##expiali", "##doc", "##ious"]),
    ]
    print()
    print(f"  {'Word':<35} Tokens")
    print(f"  {'─'*35} {'─'*40}")
    for word, toks in examples:
        print(f"  {word:<35} {toks}")


# ── 4. Unigram LM tokenization ────────────────────────────────────────────────
def unigram_lm_overview():
    print("\n=== Unigram Language Model Tokenization ===")
    print("  Used by: SentencePiece (T5, LLaMA, Gemma)")
    print()
    print("  Algorithm:")
    print("    1. Start with a large initial vocabulary (all substrings)")
    print("    2. Score each token with a unigram LM: P(text) = Π P(token_i)")
    print("    3. Remove tokens that reduce likelihood the least")
    print("    4. Repeat until target vocab size reached")
    print()
    print("  Encodes space as ▁ (special prefix character)")
    print("  'Hello world' → ['▁Hello', '▁world']  (no space token needed)")
    print()
    print("  Probabilistic: multiple segmentations possible")
    print("  Training uses EM algorithm to estimate token probabilities")


# ── 5. SentencePiece ──────────────────────────────────────────────────────────
def sentencepiece_overview():
    print("\n=== SentencePiece ===")
    print("  Language-agnostic tokenizer (no word boundary assumption)")
    print("  Treats input as raw characters (including spaces)")
    print("  Can use BPE or Unigram LM as underlying algorithm")
    print()
    print("  Used by: T5, mT5, LLaMA, Gemma, Mistral, BLOOM")
    print()
    features = [
        ("Lossless",      "Original text can always be recovered"),
        ("Language-free", "Works for Chinese, Japanese, Arabic, etc."),
        ("BOS/EOS",       "Adds <s> and </s> tokens by default"),
        ("Byte fallback", "Unknown bytes → byte tokens (no UNK for UTF-8)"),
        ("Subword reg.",  "Random segmentation at training for robustness"),
    ]
    for f, d in features:
        print(f"  {f:<15} {d}")


# ── 6. Tokenizer comparison ───────────────────────────────────────────────────
def tokenizer_comparison():
    print("\n=== Tokenizer Comparison ===")
    text = "The tokenizer converts text into numerical tokens."
    print(f"  Input: '{text}'")
    print()
    examples = {
        "BERT (WordPiece)": ["The", "token", "##izer", "converts", "text", "into",
                              "numerical", "token", "##s", "."],
        "GPT-2 (BPE)":      ["The", "Ġtoken", "izer", "Ġconverts", "Ġtext", "Ġinto",
                              "Ġnumerical", "Ġtokens", "."],
        "T5 (SentencePiece)": ["▁The", "▁token", "izer", "▁converts", "▁text", "▁into",
                                "▁numerical", "▁tokens", "."],
    }
    print(f"  {'Model':<25} Tokens")
    print(f"  {'─'*25} {'─'*55}")
    for model, toks in examples.items():
        print(f"  {model:<25} {toks}")
    print()
    print("  Vocab sizes:")
    print("    BERT-base:  30,522  (WordPiece, English)")
    print("    GPT-2:      50,257  (BPE)")
    print("    GPT-4:     ~100,000 (BPE)")
    print("    T5:         32,100  (SentencePiece)")
    print("    LLaMA-2:    32,000  (SentencePiece)")
    print("    Gemma:     256,000  (SentencePiece)")


if __name__ == "__main__":
    motivation()
    bpe_demo()
    wordpiece_overview()
    unigram_lm_overview()
    sentencepiece_overview()
    tokenizer_comparison()
