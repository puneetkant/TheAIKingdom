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

if __name__ == "__main__":
    demo()
