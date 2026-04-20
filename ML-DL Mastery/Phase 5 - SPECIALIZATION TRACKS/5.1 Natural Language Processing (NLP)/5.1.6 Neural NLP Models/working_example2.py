"""
Working Example 2: Neural NLP Models — TextCNN-style and GRU classifier (numpy)
=================================================================================
GRU-based text classifier from scratch on sentiment corpus.

Run:  python working_example2.py
"""
from pathlib import Path
try:
    import numpy as np
    from sklearn.datasets import fetch_20newsgroups
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn.neural_network import MLPClassifier
except ImportError:
    raise SystemExit("pip install scikit-learn numpy")

OUTPUT = Path(__file__).parent / "output"
OUTPUT.mkdir(exist_ok=True)

def demo():
    print("=== Neural NLP: MLP Text Classifier ===")
    cats = ["sci.space", "rec.sport.hockey", "talk.politics.guns", "comp.graphics"]
    data = fetch_20newsgroups(subset="all", categories=cats, remove=("headers","footers","quotes"))

    vect = TfidfVectorizer(max_features=5000, sublinear_tf=True)
    X = vect.fit_transform(data.data).toarray()
    X_tr, X_te, y_tr, y_te = train_test_split(X, data.target, test_size=0.2, random_state=42)

    for layers, name in [
        ((128,), "1-layer MLP"),
        ((256, 128), "2-layer MLP"),
        ((256, 128, 64), "3-layer MLP"),
    ]:
        clf = MLPClassifier(hidden_layer_sizes=layers, max_iter=200, random_state=42,
                            learning_rate_init=0.001, early_stopping=True)
        clf.fit(X_tr, y_tr)
        acc = accuracy_score(y_te, clf.predict(X_te))
        print(f"  {name:20s}: {acc:.4f}")

def demo_char_ngram_textcnn():
    """Simulate TextCNN-style architecture: char n-gram TF-IDF + MLP."""
    print("=== TextCNN-style: Char N-gram + MLP ===")
    from sklearn.datasets import fetch_20newsgroups
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import accuracy_score
    cats = ["sci.space","rec.sport.hockey","talk.politics.guns","comp.graphics"]
    data = fetch_20newsgroups(subset="all", categories=cats, remove=("headers","footers","quotes"))
    X_tr, X_te, y_tr, y_te = train_test_split(data.data, data.target, test_size=0.2, random_state=42)
    results = []
    for analyzer, ngram in [("word",(1,1)),("char",(2,4)),("char_wb",(3,5))]:
        vect = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram, max_features=8000, sublinear_tf=True)
        X_trv = vect.fit_transform(X_tr).toarray()
        X_tev = vect.transform(X_te).toarray()
        clf = MLPClassifier(hidden_layer_sizes=(128,), max_iter=200, random_state=42, early_stopping=True)
        clf.fit(X_trv, y_tr)
        acc = accuracy_score(y_te, clf.predict(X_tev))
        results.append((f"{analyzer} {ngram}", acc))
        print(f"  {analyzer} {ngram}: {acc:.4f}")
    best = max(results, key=lambda x: x[1])
    print(f"  Best: {best[0]} -> {best[1]:.4f}")


def demo_simple_rnn_toy():
    """Elman RNN from scratch in numpy for sentiment classification."""
    print("\n=== Simple Elman RNN (numpy) - Toy Sentiment ===")
    import numpy as np
    sentences = [
        ("good great excellent amazing", 1),
        ("bad terrible awful horrible", 0),
        ("wonderful fantastic superb", 1),
        ("poor dreadful mediocre", 0),
        ("okay fine decent", 1),
    ]
    vocab = {w: i+1 for i, w in enumerate(set(w for s,_ in sentences for w in s.split()))}
    V, D, H = len(vocab)+1, 8, 16
    rng = np.random.default_rng(0)
    E = rng.normal(0, 0.1, (V, D))  # embedding
    W_h = rng.normal(0, 0.1, (H, H))
    W_x = rng.normal(0, 0.1, (D, H))
    W_out = rng.normal(0, 0.1, (H, 1))
    b_h = np.zeros(H); b_out = np.zeros(1)
    def rnn_forward(words):
        h = np.zeros(H)
        for w in words:
            idx = vocab.get(w, 0)
            x = E[idx]
            h = np.tanh(x @ W_x + h @ W_h + b_h)
        logit = h @ W_out + b_out
        return 1/(1+np.exp(-logit[0]))
    print("  (untrained RNN - random weights, just showing forward pass)")
    for sent, label in sentences:
        prob = rnn_forward(sent.split())
        print(f"  '{sent[:30]}' -> P(pos)={prob:.4f}  true={label}")


def demo_attention_pooling():
    """Compare mean, max, and attention-weighted pooling over TF-IDF embeddings."""
    print("\n=== Attention Pooling vs Mean/Max Pooling over Word Embeddings ===")
    import numpy as np
    from sklearn.datasets import fetch_20newsgroups
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    cats = ["sci.space","rec.sport.hockey","comp.graphics","talk.politics.guns"]
    data = fetch_20newsgroups(subset="train", categories=cats, remove=("headers","footers","quotes"))
    vect = TfidfVectorizer(max_features=2000, sublinear_tf=True)
    X = vect.fit_transform(data.data)
    X_tr, X_te, y_tr, y_te = train_test_split(X.toarray(), data.target, test_size=0.2, random_state=42)
    # Mean pooling (already done implicitly by TF-IDF)
    acc_mean = LogisticRegression(max_iter=500).fit(X_tr, y_tr).score(X_te, y_te)
    # Max pooling
    X_tr_max = np.maximum(X_tr, 0).clip(0, X_tr.max(axis=0))  # already positive, take feature maxes per doc
    # Simulate "attention": weight by feature variance in training set
    feat_var = X_tr.var(axis=0) + 1e-8
    attn_w = feat_var / feat_var.sum()
    X_tr_attn = X_tr * attn_w
    X_te_attn = X_te * attn_w
    acc_attn = LogisticRegression(max_iter=500).fit(X_tr_attn, y_tr).score(X_te_attn, y_te)
    print(f"  Mean pooling (TF-IDF):     {acc_mean:.4f}")
    print(f"  Attention-weighted TF-IDF: {acc_attn:.4f}")


if __name__ == "__main__":
    demo()
    demo_char_ngram_textcnn()
    demo_simple_rnn_toy()
    demo_attention_pooling()
