"""
Working Example: NLP Tasks
Covers sentiment analysis, NER, question answering, summarisation,
machine translation, and text similarity — implemented with sklearn/numpy.
"""
import re, math
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score,
                             classification_report, pairwise)


def tokenise(text):
    return re.sub(r"[^\w\s]", "", text.lower()).split()


# ── 1. Sentiment Analysis ────────────────────────────────────────────────────
def sentiment_analysis():
    print("=== Sentiment Analysis ===")
    texts = [
        ("I absolutely loved this product! Amazing quality.", 1),
        ("Fantastic experience, highly recommend to everyone.", 1),
        ("Great value for money, exceeded my expectations.", 1),
        ("Brilliant! Will definitely buy again.", 1),
        ("Wonderful service and very fast delivery.", 1),
        ("Terrible product, broke after one day.", 0),
        ("Worst purchase ever. Complete waste of money.", 0),
        ("Very disappointed. Poor quality and slow shipping.", 0),
        ("Do not buy this. It stopped working immediately.", 0),
        ("Awful experience. Customer service was useless.", 0),
    ]
    X, y = zip(*texts)

    vec  = TfidfVectorizer(ngram_range=(1, 2))
    X_v  = vec.fit_transform(X)
    clf  = LogisticRegression(C=1.0, random_state=0, max_iter=500)
    clf.fit(X_v, y)
    preds = clf.predict(X_v)
    print(f"  Train accuracy: {accuracy_score(y, preds):.4f}")

    # Test on new examples
    test = [
        "This is a fantastic item that I love",
        "Very bad, would not recommend to anyone",
        "The product is okay, nothing special",
    ]
    X_test = vec.transform(test)
    probs  = clf.predict_proba(X_test)
    print()
    for t, prob in zip(test, probs):
        label = "positive" if prob[1] > 0.5 else "negative"
        print(f"  '{t[:45]}' → {label} ({prob[1]:.3f})")

    print()
    print("  Tasks in sentiment analysis:")
    print("    Binary: positive/negative")
    print("    Fine-grained: 1-5 star rating")
    print("    Aspect-level: sentiment per aspect (service, food, ambience)")


# ── 2. Named Entity Recognition (rule-based) ─────────────────────────────────
def ner_demo():
    print("\n=== Named Entity Recognition (NER) ===")

    # Simple gazetteer-based NER
    PERSON_NAMES = {"Barack", "Obama", "Steve", "Jobs", "Elon", "Musk",
                    "Jeff", "Bezos", "Bill", "Gates", "Tim", "Cook"}
    ORG_NAMES    = {"Apple", "Google", "Microsoft", "Amazon", "Tesla",
                    "OpenAI", "Meta", "Netflix", "Twitter"}
    LOCATION     = {"New", "York", "London", "Paris", "Silicon", "Valley",
                    "California", "United", "States", "Washington"}

    def simple_ner(text):
        tokens = text.split()
        labels = []
        i = 0
        while i < len(tokens):
            w = re.sub(r"[^\w]", "", tokens[i])
            if w in PERSON_NAMES:
                if i + 1 < len(tokens) and re.sub(r"[^\w]", "", tokens[i+1]) in PERSON_NAMES:
                    labels.append((tokens[i], "B-PER"))
                    labels.append((tokens[i+1], "I-PER"))
                    i += 2
                else:
                    labels.append((tokens[i], "B-PER")); i += 1
            elif w in ORG_NAMES:
                labels.append((tokens[i], "B-ORG")); i += 1
            elif w in LOCATION:
                if i + 1 < len(tokens) and re.sub(r"[^\w]", "", tokens[i+1]) in LOCATION:
                    labels.append((tokens[i], "B-LOC"))
                    labels.append((tokens[i+1], "I-LOC"))
                    i += 2
                else:
                    labels.append((tokens[i], "B-LOC")); i += 1
            else:
                labels.append((tokens[i], "O")); i += 1
        return labels

    sentences = [
        "Barack Obama visited Apple headquarters in Silicon Valley",
        "Elon Musk founded Tesla and SpaceX in California",
        "Steve Jobs and Bill Gates shaped the tech industry",
        "Google and Microsoft are competitors in New York",
    ]
    for sent in sentences:
        entities = [(w, t) for w, t in simple_ner(sent) if t != "O"]
        print(f"  Text:     {sent}")
        print(f"  Entities: {entities}")
        print()

    print("  NER evaluation metric: Entity-level F1 (exact span match)")
    print("  State-of-art: BERT-CRF, spaCy en_core_web_trf ~90 F1 on CoNLL-2003")


# ── 3. Question Answering (extractive) ───────────────────────────────────────
def question_answering():
    print("\n=== Extractive Question Answering ===")
    print("  Given: context paragraph + question")
    print("  Output: span (start, end) within the context")
    print()

    context = """
    Python is a high-level, general-purpose programming language. Its design
    philosophy emphasises code readability, using significant indentation. Python
    is dynamically typed and garbage-collected. It supports multiple programming
    paradigms, including structured, object-oriented, and functional programming.
    Guido van Rossum began working on Python in the late 1980s as a successor to
    the ABC programming language. Python 3.0 was released in 2008.
    """.strip().replace("\n", " ")

    questions = [
        "When was Python 3.0 released?",
        "Who created Python?",
        "What is Python's design philosophy?",
    ]

    # Simple keyword extraction for QA (not real span extraction)
    def simple_qa(ctx, q):
        q_words   = set(tokenise(q)) - {"what", "when", "who", "is", "the", "a", "of"}
        sentences = re.split(r"[.!?]", ctx)
        best_s, best_n = ctx[:100], 0
        for s in sentences:
            s_words = set(tokenise(s))
            overlap = len(q_words & s_words)
            if overlap > best_n:
                best_n, best_s = overlap, s.strip()
        return best_s

    for q in questions:
        ans = simple_qa(context, q)
        print(f"  Q: {q}")
        print(f"  A: {ans[:100]}...")
        print()

    print("  Real extractive QA (BERT):")
    print("    Input:  [CLS] question [SEP] context [SEP]")
    print("    Output: start logits (T,) + end logits (T,)")
    print("    Span:   argmax(start) to argmax(end)")
    print("  Dataset: SQuAD 2.0  Metric: Exact Match, F1")


# ── 4. Text Summarisation ─────────────────────────────────────────────────────
def text_summarisation():
    print("\n=== Text Summarisation ===")
    print("  Extractive: select important sentences from original text")
    print("  Abstractive: generate new text (seq2seq, LLM)")
    print()

    text = """
    Machine learning is a subset of artificial intelligence that enables
    systems to automatically learn and improve from experience without being
    explicitly programmed. It focuses on developing computer programs that can
    access data and use it to learn for themselves. The process begins with
    observations or data, such as examples, direct experience, or instruction.
    Machine learning algorithms include supervised learning, unsupervised
    learning, and reinforcement learning. Deep learning is a subfield that uses
    neural networks with many layers to learn complex patterns. Today, machine
    learning powers many applications including recommendation systems, image
    recognition, and natural language processing.
    """.strip()

    # Extractive: TF-IDF sentence scoring
    sentences = re.split(r"(?<=[.!?])\s+", text)
    vec   = TfidfVectorizer()
    tfidf = vec.fit_transform(sentences).toarray()
    # Score each sentence by mean TF-IDF
    scores = tfidf.mean(axis=1)
    top_k  = 2
    top_idx = scores.argsort()[::-1][:top_k]
    top_idx = sorted(top_idx)

    print("  Original text:")
    print(f"    {len(sentences)} sentences, {len(text.split())} words")
    print()
    print(f"  Extractive summary (top-{top_k} sentences by TF-IDF score):")
    for i in top_idx:
        print(f"    [{i+1}] {sentences[i][:100].strip()}...")

    print()
    print("  Metrics: ROUGE-1, ROUGE-2, ROUGE-L (recall-oriented n-gram overlap)")
    print("  Models: T5, BART, Pegasus, LLaMA-summarise")


# ── 5. Text Similarity ───────────────────────────────────────────────────────
def text_similarity():
    print("\n=== Text Similarity ===")

    sentences = [
        "The cat sat on the mat",
        "A cat is sitting on a mat",
        "The dog ran in the park",
        "Machine learning is a field of AI",
        "Artificial intelligence includes machine learning",
    ]

    vec   = TfidfVectorizer()
    X     = vec.fit_transform(sentences).toarray()
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X_n   = X / (norms + 1e-9)
    sim   = X_n @ X_n.T

    print(f"  {'Pair':<55} Cosine Sim")
    print(f"  {'─'*55} {'─'*10}")
    pairs = [(0,1), (0,2), (3,4), (0,3)]
    for i, j in pairs:
        print(f"  '{sentences[i][:25]}' ↔ '{sentences[j][:25]}' {sim[i,j]:.4f}")

    print()
    print("  Methods for semantic similarity:")
    methods = [
        ("TF-IDF cosine",    "Lexical overlap; misses synonyms"),
        ("BM25",             "Improved TF-IDF with term saturation"),
        ("SBERT",            "Sentence-BERT fine-tuned for similarity"),
        ("Cross-encoder",    "BERT on (s1, s2) concatenated; slower, more accurate"),
        ("OpenAI embeddings","text-embedding-3-large; best for retrieval"),
    ]
    for m, d in methods:
        print(f"    {m:<22} {d}")


if __name__ == "__main__":
    sentiment_analysis()
    ner_demo()
    question_answering()
    text_summarisation()
    text_similarity()
