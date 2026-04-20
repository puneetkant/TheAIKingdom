"""
Working Example 2: Text Preprocessing — tokenization, cleaning, normalization
==============================================================================
Full text preprocessing pipeline on sample text corpus using stdlib only.

Run:  python working_example2.py
"""
import re, string
from collections import Counter
from pathlib import Path

OUTPUT = Path(__file__).parent / "output"
OUTPUT.mkdir(exist_ok=True)

CORPUS = [
    "The cat sat on the mat. The mat was dirty!",
    "Running quickly through the forest, the fox jumped over lazy dogs.",
    "NLP is amazing — it helps computers understand human language!!",
    "Text pre-processing: tokenisation, stemming, and lemmatisation.",
    "HELLO WORLD. hello world. Hello, World!",
]

def clean(text):
    text = text.lower()
    text = re.sub(r"[—–]", " ", text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize(text): return text.split()

def remove_stopwords(tokens, stopwords=None):
    sw = stopwords or {"the","a","an","is","on","was","it","and","of","to","through","over"}
    return [t for t in tokens if t not in sw]

def simple_stem(word):
    """Suffix-stripping stemmer (Porter-lite)."""
    for suffix in ("ing","ed","ness","tion","ly","er"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[:-len(suffix)]
    return word

def demo():
    print("=== Text Preprocessing Pipeline ===\n")
    all_tokens = []
    for doc in CORPUS:
        cleaned = clean(doc)
        tokens = tokenize(cleaned)
        filtered = remove_stopwords(tokens)
        stemmed = [simple_stem(t) for t in filtered]
        print(f"  Raw:     {doc[:60]}")
        print(f"  Tokens:  {filtered}")
        print(f"  Stemmed: {stemmed}\n")
        all_tokens.extend(stemmed)

    freq = Counter(all_tokens)
    print("Top 10 terms:", freq.most_common(10))
    (OUTPUT / "token_freq.txt").write_text("\n".join(f"{w}\t{c}" for w,c in freq.most_common(20)))
    print(f"Saved token_freq.txt ({len(freq)} unique terms)")

def demo_ngrams():
    """Build character and word n-grams from corpus."""
    print("\n=== N-gram Extraction ===")
    text = "the cat sat on the mat and the cat ate a rat"
    tokens = text.split()
    for n in [2, 3]:
        ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
        freq   = Counter(ngrams)
        print(f"  {n}-grams (top 3): {freq.most_common(3)}")
    # Character 3-grams for OOV handling
    word = "preprocessing"
    char_ngrams = [word[i:i+3] for i in range(len(word)-2)]
    print(f"  Char 3-grams of '{word}': {char_ngrams}")


def demo_vocab_stats():
    """Compute vocabulary statistics: TTR, hapax legomena."""
    print("\n=== Vocabulary Statistics ===")
    extended = CORPUS + [
        "Computers process natural language using algorithms and models.",
        "Machine translation helps humans communicate across languages.",
    ]
    all_tok = []
    for doc in extended:
        all_tok.extend(remove_stopwords(tokenize(clean(doc))))
    total = len(all_tok)
    unique = len(set(all_tok))
    ttr   = unique / total
    hapax = sum(1 for w, c in Counter(all_tok).items() if c == 1)
    print(f"  Total tokens: {total}  Unique: {unique}  TTR: {ttr:.3f}")
    print(f"  Hapax legomena (appear once): {hapax} ({100*hapax/unique:.1f}% of vocab)")


if __name__ == "__main__":
    demo()
    demo_ngrams()
    demo_vocab_stats()
