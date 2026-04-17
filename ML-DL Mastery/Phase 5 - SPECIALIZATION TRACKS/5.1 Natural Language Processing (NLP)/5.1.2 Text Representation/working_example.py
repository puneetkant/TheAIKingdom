"""
Working Example: Text Representation
Covers Bag of Words (BoW), TF-IDF, n-grams, co-occurrence matrices,
and PMI — all from scratch plus sklearn comparison.
"""
import math, re, string
from collections import Counter, defaultdict
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

CORPUS = [
    "the cat sat on the mat",
    "the cat sat on the hat",
    "the dog sat on the mat",
    "the dog chased the cat",
    "the mat is on the floor",
    "a cat chased a dog",
]


def tokenise(text):
    return re.sub(r"[^\w\s]", "", text.lower()).split()


# ── 1. Bag of Words ───────────────────────────────────────────────────────────
def bag_of_words():
    print("=== Bag of Words (BoW) ===")
    tokens_list = [tokenise(d) for d in CORPUS]
    vocab       = sorted(set(t for doc in tokens_list for t in doc))
    print(f"  Vocab size: {len(vocab)}")
    print(f"  Vocabulary: {vocab}")
    print()

    # Build BoW matrix
    bow = np.zeros((len(CORPUS), len(vocab)), dtype=int)
    v_idx = {w: i for i, w in enumerate(vocab)}
    for di, tokens in enumerate(tokens_list):
        for t in tokens:
            bow[di, v_idx[t]] += 1

    print(f"  {'Doc':<45} {'a':>2} {'cat':>3} {'chased':>6} {'dog':>3} {'floor':>5} ...")
    print(f"  {'─'*45} {'─'*2} {'─'*3} {'─'*6} {'─'*3} {'─'*5}")
    for di, doc in enumerate(CORPUS):
        row = bow[di, [v_idx[w] for w in ["a", "cat", "chased", "dog", "floor"]]]
        print(f"  {doc:<45} {row[0]:>2} {row[1]:>3} {row[2]:>6} {row[3]:>3} {row[4]:>5}")

    # Cosine similarity
    from sklearn.metrics.pairwise import cosine_similarity as cs
    sim = cs(bow)
    print(f"\n  Cosine similarity (doc 0 vs all):")
    for i, d in enumerate(CORPUS):
        print(f"    [{i}] {sim[0, i]:.4f}  {d}")


# ── 2. TF-IDF from scratch ────────────────────────────────────────────────────
def tfidf_scratch():
    print("\n=== TF-IDF (from scratch) ===")
    N = len(CORPUS)
    tokens_list = [tokenise(d) for d in CORPUS]
    vocab = sorted(set(t for doc in tokens_list for t in doc))
    v_idx = {w: i for i, w in enumerate(vocab)}

    # TF: term frequency (normalised)
    tf = np.zeros((N, len(vocab)))
    for di, tokens in enumerate(tokens_list):
        cnt  = Counter(tokens)
        dlen = len(tokens)
        for w, c in cnt.items():
            tf[di, v_idx[w]] = c / dlen

    # IDF: log((N+1)/(df+1)) + 1   (smooth IDF)
    df  = np.zeros(len(vocab))
    for tokens in tokens_list:
        for w in set(tokens):
            df[v_idx[w]] += 1
    idf = np.log((N + 1) / (df + 1)) + 1

    tfidf = tf * idf

    print(f"  TF-IDF matrix shape: {tfidf.shape}")
    print(f"  Top tokens by mean TF-IDF:")
    mean_tfidf = tfidf.mean(0)
    for wi in np.argsort(mean_tfidf)[::-1][:8]:
        print(f"    {vocab[wi]:<12} {mean_tfidf[wi]:.4f}")

    # Compare with sklearn
    sk_tfidf = TfidfVectorizer().fit_transform(CORPUS).toarray()
    print(f"\n  sklearn TF-IDF shape: {sk_tfidf.shape}")
    sim = cosine_similarity(tfidf)[0]
    print(f"  Scratch cosine (doc 0 vs all): {[f'{s:.3f}' for s in sim]}")


# ── 3. N-grams ────────────────────────────────────────────────────────────────
def ngrams_demo():
    print("\n=== N-grams ===")
    text = "the cat sat on the mat"

    def get_ngrams(tokens, n):
        return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

    tokens = text.split()
    for n in [1, 2, 3]:
        grams = get_ngrams(tokens, n)
        print(f"  {n}-grams: {grams}")

    print()
    print("  Character n-grams (n=3) for 'hello':")
    word = "hello"
    char_grams = [word[i:i+3] for i in range(len(word)-2)]
    print(f"    {char_grams}")
    print("  Use: out-of-vocabulary words, language identification, misspelling detection")

    # sklearn n-gram TF-IDF
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1).fit(CORPUS)
    print(f"\n  TF-IDF unigram+bigram features: {len(vec.vocabulary_)}")
    features = sorted(vec.vocabulary_.keys())[:10]
    print(f"  First 10 features: {features}")


# ── 4. Co-occurrence matrix ───────────────────────────────────────────────────
def cooccurrence_matrix():
    print("\n=== Co-occurrence Matrix ===")
    window = 2
    tokens_list = [tokenise(d) for d in CORPUS]
    all_tokens  = [t for doc in tokens_list for t in doc]
    vocab       = sorted(set(all_tokens))
    v_idx       = {w: i for i, w in enumerate(vocab)}
    n           = len(vocab)

    M = np.zeros((n, n), dtype=float)
    for tokens in tokens_list:
        for i, w in enumerate(tokens):
            start = max(0, i - window)
            end   = min(len(tokens), i + window + 1)
            for j in range(start, end):
                if i != j:
                    M[v_idx[w], v_idx[tokens[j]]] += 1

    print(f"  Vocab: {vocab}")
    print(f"  Co-occurrence matrix (window={window}):")
    # Print a mini subset (cat, dog, mat, sat)
    subset_words = ["cat", "dog", "mat", "sat"]
    subset_idx   = [v_idx[w] for w in subset_words]
    print(f"  {'':>6} " + " ".join(f"{w:>6}" for w in subset_words))
    for wi, word in zip(subset_idx, subset_words):
        row = M[wi, subset_idx]
        print(f"  {word:>6} " + " ".join(f"{int(v):>6}" for v in row))


# ── 5. PMI (Pointwise Mutual Information) ────────────────────────────────────
def pmi_demo():
    print("\n=== Pointwise Mutual Information (PMI) ===")
    print("  PMI(w1, w2) = log P(w1, w2) / (P(w1) · P(w2))")
    print("  Positive PMI (PPMI) = max(0, PMI)  — basis of GloVe")
    print()

    tokens_list = [tokenise(d) for d in CORPUS]
    window      = 2
    all_toks    = [t for doc in tokens_list for t in doc]
    vocab       = sorted(set(all_toks))
    v_idx       = {w: i for i, w in enumerate(vocab)}
    n_vocab     = len(vocab)
    N_total     = len(all_toks)

    word_cnt   = Counter(all_toks)
    pair_cnt   = defaultdict(float)
    N_pairs    = 0
    for tokens in tokens_list:
        for i, w in enumerate(tokens):
            for j in range(max(0, i-window), min(len(tokens), i+window+1)):
                if i != j:
                    pair_cnt[(w, tokens[j])] += 1
                    N_pairs += 1

    # PPMI for a few pairs
    pairs = [("cat", "dog"), ("cat", "mat"), ("cat", "sat"), ("dog", "mat")]
    print(f"  {'Pair':<18} PMI   PPMI")
    print(f"  {'─'*18} {'─'*5} {'─'*5}")
    for w1, w2 in pairs:
        p_w1 = word_cnt[w1] / N_total
        p_w2 = word_cnt[w2] / N_total
        p12  = pair_cnt[(w1, w2)] / max(N_pairs, 1)
        pmi  = math.log(p12 / (p_w1 * p_w2 + 1e-10) + 1e-10)
        ppmi = max(0, pmi)
        print(f"  ({w1}, {w2}):{'':<5} {pmi:>5.2f} {ppmi:>5.2f}")


if __name__ == "__main__":
    bag_of_words()
    tfidf_scratch()
    ngrams_demo()
    cooccurrence_matrix()
    pmi_demo()
