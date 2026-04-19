"""
Working Example: Text Preprocessing
Covers tokenization, lowercasing, stopword removal, stemming, lemmatization,
punctuation handling, and a full preprocessing pipeline.
"""
import re, string, math
from collections import Counter


# -- 1. Raw text pipeline steps ------------------------------------------------
SAMPLE = """Natural Language Processing (NLP) is a subfield of Artificial Intelligence.
It enables computers to understand, interpret, and generate human language.
The challenges include: ambiguity, context-dependence, and sarcasm!
Dr. Smith's research covers BERT, GPT-3, and similar Large Language Models (LLMs)."""


def lowercasing():
    print("=== Lowercasing ===")
    text = "Hello World! NLP is AMAZING."
    print(f"  Before: {text}")
    print(f"  After:  {text.lower()}")


def punctuation_handling():
    print("\n=== Punctuation Handling ===")
    text = "Hello, World! How's it going? I'm fine—thanks."
    # Remove all punctuation
    remove = text.translate(str.maketrans("", "", string.punctuation))
    # Replace with space
    replace = re.sub(r"[^\w\s]", " ", text)
    print(f"  Original: {text}")
    print(f"  Remove:   {remove}")
    print(f"  Replace:  {replace}")


def tokenization():
    print("\n=== Tokenization ===")
    text = "Dr. Smith said, 'NLP is fun!' She wasn't joking."

    # Word tokenization (simple split)
    word_tokens = text.split()
    print(f"  Split tokens ({len(word_tokens)}): {word_tokens[:8]}...")

    # Regex tokenizer: keep alphanumeric + apostrophe
    regex_tokens = re.findall(r"\b\w[\w']*\b", text)
    print(f"  Regex tokens ({len(regex_tokens)}): {regex_tokens}")

    # Sentence tokenization
    sentences = re.split(r"(?<=[.!?])\s+", SAMPLE)
    print(f"\n  Sentence tokenization: {len(sentences)} sentences")
    for i, s in enumerate(sentences):
        print(f"    {i+1}: {s[:70]}...")


# -- 2. Stopword removal -------------------------------------------------------
STOPWORDS = set("""
a an the is are was were be been being have has had do does did
will would could should may might shall can need dare ought used
i me my myself we our ours ourselves you your yours he him his
she her hers it its they them their of in at by for with on to
as if and but or not so nor yet from than after before while until
""".split())


def stopword_removal():
    print("\n=== Stopword Removal ===")
    text = "The quick brown fox jumps over the lazy dog and it is very fun to watch"
    tokens = text.lower().split()
    filtered = [t for t in tokens if t not in STOPWORDS]
    print(f"  Original ({len(tokens)} tokens): {tokens[:10]}...")
    print(f"  Filtered ({len(filtered)} tokens): {filtered}")
    removed = set(tokens) - set(filtered)
    print(f"  Removed: {sorted(removed)}")


# -- 3. Stemming (Porter algorithm simplified) ---------------------------------
def stem_word(word: str) -> str:
    """Simplified Porter-like stemming rules."""
    word = word.lower()
    # Step 1a
    for suf, rep in [("sses", "ss"), ("ies", "i"), ("ss", "ss"), ("s", "")]:
        if word.endswith(suf) and len(word) > len(suf) + 1:
            return word[: -len(suf)] + rep
    # Step 1b
    for suf in ["eed", "ed", "ing"]:
        if word.endswith(suf):
            stem = word[: -len(suf)]
            if len(stem) >= 2:
                return stem
    # Step 2
    for suf, rep in [("ational", "ate"), ("tional", "tion"), ("enci", "ence"),
                     ("anci", "ance"), ("iser", "ise"), ("iser", "ize")]:
        if word.endswith(suf) and len(word) > len(suf) + 2:
            return word[: -len(suf)] + rep
    return word


def stemming():
    print("\n=== Stemming ===")
    words = ["running", "runs", "ran", "easily", "fairly", "processing",
             "processed", "processes", "happiness", "happily", "studies"]
    print(f"  {'Word':<15} Stem")
    print(f"  {'-'*15} {'-'*12}")
    for w in words:
        print(f"  {w:<15} {stem_word(w)}")
    print("\n  Note: stemming is fast but may produce non-words ('happi', 'studi')")


# -- 4. Lemmatization (rule-based) ---------------------------------------------
LEMMA_MAP = {
    "running": "run", "runs": "run", "ran": "run",
    "flies": "fly", "flying": "fly", "flew": "fly",
    "better": "good", "best": "good", "worse": "bad",
    "is": "be", "are": "be", "was": "be", "were": "be",
    "has": "have", "had": "have",
    "studies": "study", "studied": "study", "studying": "study",
    "processed": "process", "processing": "process",
}


def lemmatization():
    print("\n=== Lemmatization ===")
    words = list(LEMMA_MAP.keys())
    print(f"  {'Word':<15} Lemma")
    print(f"  {'-'*15} {'-'*10}")
    for w in words:
        print(f"  {w:<15} {LEMMA_MAP.get(w, w)}")
    print("\n  Note: proper lemmatization requires a full morphological lexicon + POS tags")
    print("  (nltk.WordNetLemmatizer, spacy .lemma_)")

    print("\n  Stemming vs Lemmatization:")
    print(f"  {'Word':<12} {'Stem':<12} {'Lemma':<12}")
    print(f"  {'-'*12} {'-'*12} {'-'*12}")
    for w in ["running", "better", "studies", "flies"]:
        print(f"  {w:<12} {stem_word(w):<12} {LEMMA_MAP.get(w, w):<12}")


# -- 5. Text normalisation ----------------------------------------------------
def text_normalisation():
    print("\n=== Text Normalisation ===")
    rules = [
        ("URLs",         r"https?://\S+",             "[URL]"),
        ("Emails",       r"\S+@\S+\.\S+",             "[EMAIL]"),
        ("Numbers",      r"\b\d+(\.\d+)?\b",          "[NUM]"),
        ("HTML tags",    r"<[^>]+>",                   ""),
        ("Contractions", r"n't",                       " not"),
        ("Contractions", r"'re",                       " are"),
        ("Extra space",  r"\s+",                       " "),
    ]
    print(f"  {'Pattern':<14} {'Replacement':<12} Example")
    print(f"  {'-'*14} {'-'*12} {'-'*20}")
    examples = {
        "URLs":         "Visit https://openai.com now",
        "Emails":       "Contact me@example.com",
        "Numbers":      "There are 42 items at $3.99",
        "HTML tags":    "<b>Bold</b> and <em>italic</em>",
        "Contractions": "I won't and they're here",
    }
    for name, pattern, repl in rules[:-1]:
        if name in examples:
            out = re.sub(pattern, repl, examples[name])
            print(f"  {name:<14} {repr(repl):<12} '{examples[name]}' -> '{out}'")


# -- 6. Full pipeline ----------------------------------------------------------
def full_pipeline(text: str, stop=True, lower=True, stem=False) -> list:
    text = re.sub(r"https?://\S+", "[URL]", text)
    text = re.sub(r"\S+@\S+\.\S+", "[EMAIL]", text)
    if lower: text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = text.split()
    if stop: tokens = [t for t in tokens if t not in STOPWORDS]
    if stem: tokens = [stem_word(t) for t in tokens]
    return tokens


def pipeline_demo():
    print("\n=== Full Preprocessing Pipeline Demo ===")
    corpus = [
        "Natural Language Processing is a fascinating field!",
        "I'm learning BERT and GPT-4 for various NLP tasks.",
        "Visit https://example.com for more info. Email: info@example.com",
        "The quick brown fox jumps over the lazy dog.",
    ]
    print(f"  {'Config':<30} Tokens")
    print(f"  {'-'*30} {'-'*50}")
    for text in corpus:
        tokens = full_pipeline(text, stop=True, lower=True, stem=False)
        print(f"  IN:  {text[:55]}")
        print(f"  OUT: {tokens[:10]}")
        print()

    # Term frequency example
    print("  Term Frequency (top-10 across corpus):")
    all_tokens = []
    for text in corpus:
        all_tokens.extend(full_pipeline(text))
    tf = Counter(all_tokens)
    for word, cnt in tf.most_common(10):
        print(f"    {word:<15} {cnt}")


if __name__ == "__main__":
    lowercasing()
    punctuation_handling()
    tokenization()
    stopword_removal()
    stemming()
    lemmatization()
    text_normalisation()
    pipeline_demo()
