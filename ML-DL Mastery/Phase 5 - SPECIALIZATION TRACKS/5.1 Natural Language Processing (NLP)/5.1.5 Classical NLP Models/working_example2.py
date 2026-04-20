"""
Working Example 2: Classical NLP Models — Naive Bayes, Logistic Regression, SVM text classification
======================================================================================================
Text classification on 20 newsgroups subset with TF-IDF features.

Run:  python working_example2.py
"""
from pathlib import Path
try:
    import numpy as np
    from sklearn.datasets import fetch_20newsgroups
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    from sklearn.metrics import classification_report, accuracy_score
except ImportError:
    raise SystemExit("pip install scikit-learn numpy")

import re

OUTPUT = Path(__file__).parent / "output"
OUTPUT.mkdir(exist_ok=True)

def demo():
    print("=== Classical NLP Text Classification ===")
    cats = ["sci.space", "rec.sport.hockey", "talk.politics.guns", "comp.graphics"]
    train = fetch_20newsgroups(subset="train", categories=cats, remove=("headers","footers","quotes"))
    test  = fetch_20newsgroups(subset="test",  categories=cats, remove=("headers","footers","quotes"))

    vect = TfidfVectorizer(max_features=10000, sublinear_tf=True, ngram_range=(1,2))
    X_tr = vect.fit_transform(train.data)
    X_te = vect.transform(test.data)

    results = {}
    for name, model in [
        ("Naive Bayes", MultinomialNB(alpha=0.1)),
        ("Logistic Regression", LogisticRegression(max_iter=1000, C=1.0)),
        ("LinearSVC", LinearSVC(max_iter=2000, C=0.5)),
    ]:
        model.fit(X_tr, train.target)
        acc = accuracy_score(test.target, model.predict(X_te))
        results[name] = acc
        print(f"\n  {name}: {acc:.4f}")

    best = max(results, key=results.get)
    print(f"\nBest model: {best} ({results[best]:.4f})")
    (OUTPUT / "nlp_classical_results.txt").write_text(
        "\n".join(f"{k}: {v:.4f}" for k,v in results.items()))
    print("Saved nlp_classical_results.txt")

def demo_bigram_hmm():
    """Bigram HMM POS tagger with Viterbi decoding (pure numpy)."""
    print("\n=== Bigram HMM POS Tagger (Viterbi) ===")
    tags  = ["NN", "VB", "DT", "JJ"]
    words = ["the", "cat", "sat", "dog", "runs", "big", "a", "red"]
    tag2i  = {t: i for i, t in enumerate(tags)}
    word2i = {w: i for i, w in enumerate(words)}
    T, W = len(tags), len(words)

    train_sents = [
        (["the", "cat", "sat"],        ["DT", "NN", "VB"]),
        (["a", "big", "dog", "runs"],  ["DT", "JJ", "NN", "VB"]),
        (["the", "red", "cat", "sat"], ["DT", "JJ", "NN", "VB"]),
    ]

    # Counts with add-1 smoothing
    pi_c = np.ones(T)
    A_c  = np.ones((T, T))
    B_c  = np.ones((T, W))

    for ws, ts in train_sents:
        ti = [tag2i[t] for t in ts]
        pi_c[ti[0]] += 1
        for k in range(len(ti) - 1):
            A_c[ti[k], ti[k + 1]] += 1
        for k, w in enumerate(ws):
            if w in word2i:
                B_c[ti[k], word2i[w]] += 1

    pi = pi_c / pi_c.sum()
    A  = A_c  / A_c.sum(axis=1, keepdims=True)
    B  = B_c  / B_c.sum(axis=1, keepdims=True)

    # Viterbi decoding
    test_words = ["the", "big", "cat"]
    N = len(test_words)
    vit  = np.full((T, N), -np.inf)
    back = np.zeros((T, N), dtype=int)

    w0 = word2i.get(test_words[0], 0)
    vit[:, 0] = np.log(pi) + np.log(B[:, w0])

    for t in range(1, N):
        wt = word2i.get(test_words[t], 0)
        trans = vit[:, t - 1, np.newaxis] + np.log(A)  # (T, T): trans[prev, next]
        best_prev = np.argmax(trans, axis=0)             # best prev state per next state
        back[:, t] = best_prev
        vit[:, t]  = trans[best_prev, np.arange(T)] + np.log(B[:, wt])

    # Backtrack
    path = [0] * N
    path[-1] = int(np.argmax(vit[:, -1]))
    for t in range(N - 2, -1, -1):
        path[t] = back[path[t + 1], t + 1]

    predicted = [tags[i] for i in path]
    print(f"  Words : {test_words}")
    print(f"  Tags  : {predicted}")


def demo_maxent_text():
    """MaxEnt (logistic regression) with manual boolean feature engineering on 20newsgroups."""
    print("\n=== MaxEnt Text Classifier (Manual Boolean Features) ===")
    cats = ["sci.space", "rec.sport.hockey", "talk.politics.guns", "comp.graphics"]
    train = fetch_20newsgroups(subset="train", categories=cats,
                               remove=("headers", "footers", "quotes"))
    test  = fetch_20newsgroups(subset="test",  categories=cats,
                               remove=("headers", "footers", "quotes"))

    keywords = [
        "space", "nasa", "orbit", "shuttle",
        "hockey", "nhl", "puck", "goal",
        "gun", "weapon", "firearm", "rifle",
        "graphics", "image", "pixel", "render",
    ]

    def make_features(docs):
        rows = []
        for doc in docs:
            doc_lower = doc.lower()
            tokens = set(doc_lower.split())
            row = [1 if kw in tokens else 0 for kw in keywords]
            n = len(doc_lower.split())
            row += [1 if n < 50 else 0, 1 if 50 <= n < 200 else 0, 1 if n >= 200 else 0]
            rows.append(row)
        return np.array(rows, dtype=float)

    X_tr = make_features(train.data)
    X_te = make_features(test.data)

    clf = LogisticRegression(max_iter=500, C=1.0, solver="lbfgs")
    clf.fit(X_tr, train.target)
    acc = accuracy_score(test.target, clf.predict(X_te))
    print(f"  MaxEnt accuracy (manual features): {acc:.4f}")

    feat_names = keywords + ["short_doc", "med_doc", "long_doc"]
    print("  Top 3 features per class:")
    for ci, cat in enumerate(cats):
        top3 = np.argsort(clf.coef_[ci])[::-1][:3]
        print(f"    {cat.split('.')[-1]:20s}: {[feat_names[i] for i in top3]}")


def demo_naive_chunker():
    """Regex-based NP chunker (DT? JJ* NN+) applied to POS-tagged toy sentences."""
    print("\n=== Naive NP Chunker (Regex-based) ===")

    tagged_sents = [
        [("the", "DT"), ("big", "JJ"), ("cat", "NN"), ("sat", "VB")],
        [("a", "DT"), ("red", "JJ"), ("dog", "NN"), ("chased", "VB"),
         ("the", "DT"), ("cat", "NN")],
        [("happy", "JJ"), ("children", "NN"), ("run", "VB"), ("fast", "JJ")],
        [("the", "DT"), ("machine", "NN"), ("learning", "NN"),
         ("model", "NN"), ("works", "VB")],
    ]

    # NP grammar: DT? JJ* NN+  (each tag followed by a space for boundary matching)
    np_pattern = re.compile(r"(?:DT )?(?:JJ )*(?:NN )+")

    def chunk_np(tagged):
        tag_str = " ".join(tag for _, tag in tagged) + " "
        offsets = []
        pos = 0
        for _, tag in tagged:
            offsets.append(pos)
            pos += len(tag) + 1
        chunks = []
        for m in np_pattern.finditer(tag_str):
            s_idx = sum(1 for o in offsets if o < m.start())
            e_idx = sum(1 for o in offsets if o < m.end())
            phrase = [tagged[i][0] for i in range(s_idx, e_idx)]
            if phrase:
                chunks.append("[NP " + " ".join(phrase) + "]")
        return chunks

    for i, sent in enumerate(tagged_sents):
        words = " ".join(w for w, _ in sent)
        nps = chunk_np(sent)
        print(f"  Sent {i + 1}: {words}")
        print(f"    Chunks: {nps if nps else 'none'}")


if __name__ == "__main__":
    demo()
    demo_bigram_hmm()
    demo_maxent_text()
    demo_naive_chunker()
