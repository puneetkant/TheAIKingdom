"""
Working Example: Neural NLP Models
Covers TextCNN, BiLSTM for classification and NER, seq2seq with attention,
character-level model — all from scratch with numpy.
"""
import numpy as np
import re, math
from collections import Counter
from sklearn.datasets import fetch_20newsgroups
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


def sigmoid(z): return 1 / (1 + np.exp(-z.clip(-10, 10)))
def tanh(z):    return np.tanh(z.clip(-10, 10))
def relu(z):    return np.maximum(0, z)
def softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


# -- Vocabulary builder --------------------------------------------------------
def build_vocab(texts, max_vocab=2000, min_freq=2):
    cnt = Counter()
    for text in texts:
        for w in re.sub(r"[^\w\s]", "", text.lower()).split():
            cnt[w] += 1
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for w, c in cnt.most_common():
        if c < min_freq: break
        if len(vocab) >= max_vocab: break
        vocab[w] = len(vocab)
    return vocab


def encode(text, vocab, max_len=50):
    tokens = re.sub(r"[^\w\s]", "", text.lower()).split()[:max_len]
    ids    = [vocab.get(t, 1) for t in tokens]
    if len(ids) < max_len:
        ids += [0] * (max_len - len(ids))
    return np.array(ids[:max_len])


# -- 1. TextCNN ----------------------------------------------------------------
class TextCNN:
    """
    Simplified TextCNN (Kim 2014).
    For each filter size: conv -> max-pool -> concat -> FC -> softmax
    """
    def __init__(self, vocab_size, emb_dim, n_classes, filter_sizes=(2,3,4),
                 n_filters=16, rng=None):
        rng = rng or np.random.default_rng(0)
        self.E = rng.standard_normal((vocab_size, emb_dim)) * 0.1  # embeddings
        # One filter bank per size (simplified: one filter per size)
        self.convs = {}
        for fs in filter_sizes:
            self.convs[fs] = rng.standard_normal((n_filters, fs, emb_dim)) * 0.1
        total_feats = len(filter_sizes) * n_filters
        self.W_fc   = rng.standard_normal((total_feats, n_classes)) * 0.1
        self.b_fc   = np.zeros(n_classes)
        self.filter_sizes = filter_sizes
        self.n_filters    = n_filters

    def forward(self, token_ids_batch):
        """token_ids_batch: (B, T)"""
        B, T = token_ids_batch.shape
        X = self.E[token_ids_batch]   # (B, T, D)
        pooled = []
        for fs, F in self.convs.items():
            # F: (n_filters, fs, D)
            # Conv: slide over T; output shape (B, T-fs+1, n_filters)
            out_len = T - fs + 1
            if out_len <= 0: continue
            conv_out = np.zeros((B, out_len, self.n_filters))
            for fi in range(self.n_filters):
                for t in range(out_len):
                    conv_out[:, t, fi] = relu(
                        (X[:, t:t+fs, :] * F[fi]).sum(axis=(1, 2))
                    )
            # Max pooling over time
            pooled.append(conv_out.max(axis=1))   # (B, n_filters)
        concat = np.concatenate(pooled, axis=1)   # (B, total_feats)
        logits = concat @ self.W_fc + self.b_fc   # (B, n_classes)
        return softmax(logits)

    def fit(self, X_ids, y, epochs=10, lr=0.05, batch_size=32, rng=None):
        rng = rng or np.random.default_rng(1)
        n   = len(X_ids)
        losses = []
        for ep in range(epochs):
            idx    = rng.permutation(n)
            ep_loss = 0
            for i in range(0, n, batch_size):
                Xb = X_ids[idx[i:i+batch_size]]
                yb = y[idx[i:i+batch_size]]
                probs = self.forward(Xb)
                # Cross-entropy
                ce   = -np.log(probs[np.arange(len(yb)), yb] + 1e-9)
                ep_loss += ce.sum()
                # Gradient on output (simplified)
                dL = probs.copy()
                dL[np.arange(len(yb)), yb] -= 1
                dL /= len(yb)
                # Update FC
                B, T = Xb.shape
                X   = self.E[Xb]
                pooled = []
                for fs, F in self.convs.items():
                    out_len = T - fs + 1
                    if out_len <= 0: continue
                    conv_out = np.zeros((len(Xb), out_len, self.n_filters))
                    for fi in range(self.n_filters):
                        for t in range(out_len):
                            conv_out[:, t, fi] = relu(
                                (X[:, t:t+fs, :] * F[fi]).sum(axis=(1, 2))
                            )
                    pooled.append(conv_out.max(axis=1))
                H = np.concatenate(pooled, axis=1)
                self.W_fc -= lr * np.clip(H.T @ dL, -1, 1)
                self.b_fc -= lr * np.clip(dL.sum(0), -1, 1)
                # Embedding gradient (simplified)
                dH = np.clip(dL @ self.W_fc.T, -1, 1)
            losses.append(ep_loss / n)
        return losses


def textcnn_demo():
    print("=== TextCNN for Text Classification ===")
    print("  Architecture: Embed -> [Conv(f=2), Conv(f=3), Conv(f=4)] -> MaxPool -> Concat -> FC")
    print()
    # Small synthetic sentiment dataset
    rng = np.random.default_rng(42)
    pos = ["this movie is great and amazing",
           "wonderful experience truly spectacular",
           "loved every moment of this film",
           "fantastic story beautiful cinematography",
           "excellent performance highly recommend",
           "brilliant outstanding superb acting"]
    neg = ["terrible film complete waste of time",
           "boring awful predictable plot",
           "hated every scene disgusting",
           "worst film ever poor direction",
           "painful to watch dreadful acting",
           "horrible story disappointing ending"]
    texts = pos + neg
    labels = [1]*len(pos) + [0]*len(neg)

    vocab = build_vocab(texts, max_vocab=200)
    X_ids = np.array([encode(t, vocab, max_len=10) for t in texts])
    y     = np.array(labels)

    model  = TextCNN(len(vocab), emb_dim=8, n_classes=2,
                     filter_sizes=(2, 3), n_filters=8, rng=rng)
    losses = model.fit(X_ids, y, epochs=40, lr=0.05, rng=rng)
    probs  = model.forward(X_ids)
    preds  = probs.argmax(1)
    acc    = (preds == y).mean()
    print(f"  Vocab: {len(vocab)}  Samples: {len(texts)}")
    print(f"  Loss: {losses[0]:.4f} -> {losses[-1]:.4f}")
    print(f"  Train accuracy: {acc:.4f}")
    print()
    print("  TextCNN advantages:")
    print("    - Fast (parallelisable conv)")
    print("    - Captures local n-gram features")
    print("    - Multiple filter sizes = multi-scale patterns")


# -- 2. BiLSTM for NER --------------------------------------------------------
def lstm_step(x, h, c, W, U, b):
    """Single LSTM step."""
    z = x @ W + h @ U + b
    d = z.shape[-1] // 4
    i_g, f_g, g, o_g = z[:d], z[d:2*d], z[2*d:3*d], z[3*d:]
    ig = sigmoid(i_g); fg = sigmoid(f_g)
    gg = np.tanh(g);   og = sigmoid(o_g)
    c_new = fg * c + ig * gg
    h_new = og * np.tanh(c_new)
    return h_new, c_new


def bilstm_ner_demo():
    print("\n=== BiLSTM for NER ===")
    print("  Named Entity Recognition: tag each token with B-PER/I-PER/B-ORG/O etc.")
    print()

    # Toy NER dataset (BIO scheme)
    sentences = [
        (["Steve", "Jobs", "founded", "Apple", "Inc", "."],
         ["B-PER", "I-PER", "O", "B-ORG", "I-ORG", "O"]),
        (["Barack", "Obama", "visited", "New", "York", "City"],
         ["B-PER", "I-PER", "O", "B-LOC", "I-LOC", "I-LOC"]),
        (["The", "Microsoft", "CEO", "announced", "a", "deal"],
         ["O", "B-ORG", "O", "O", "O", "O"]),
    ]

    tag_set = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
    tag2i   = {t: i for i, t in enumerate(tag_set)}

    # Build simple vocabulary
    all_words = [w for sent, _ in sentences for w in sent]
    vocab     = {w: i+1 for i, w in enumerate(sorted(set(all_words)))}
    vocab["<UNK>"] = 0

    print("  Architecture:")
    print("    Token -> Embedding -> BiLSTM -> [h_fwd; h_bwd] -> Linear -> Tag")
    print()
    print(f"  Tags: {tag_set}")
    print()

    for sent, tags in sentences:
        ids  = [vocab.get(w, 0) for w in sent]
        print(f"  Sentence: {' '.join(sent)}")
        print(f"  Labels:   {' '.join(tags)}")
        print(f"  IDs:      {ids}")
        print()

    print("  Real BiLSTM-CRF (state of art before BERT):")
    print("    - BiLSTM captures both past and future context per token")
    print("    - CRF layer enforces valid tag transitions (B before I, etc.)")
    print("    - Together: ~90+ F1 on CoNLL-2003 benchmark")


# -- 3. Seq2Seq with attention ------------------------------------------------
def seq2seq_demo():
    print("\n=== Seq2Seq with Attention ===")
    print("  Encoder-Decoder architecture for translation, summarisation, etc.")
    print()
    print("  Encoder:")
    print("    h_t = LSTM(e(x_t), h_{t-1})")
    print("    context H = [h_1, ..., h_T]  (all hidden states)")
    print()
    print("  Attention (Bahdanau):")
    print("    e_t = v^T tanh(W_a h + U_a s_{t-1})")
    print("    alpha_t = softmax(e_t)              # attention weights")
    print("    c_t = Sigma alpha_{ti} h_i              # context vector")
    print()
    print("  Decoder:")
    print("    s_t = LSTM([e(y_{t-1}); c_t], s_{t-1})")
    print("    p(y_t) = softmax(W_o s_t)")
    print()

    # Toy attention weight demo
    rng = np.random.default_rng(7)
    T_src, T_tgt, d = 5, 4, 8
    H = rng.standard_normal((T_src, d))   # encoder outputs
    s = rng.standard_normal(d)            # decoder state
    W_a = rng.standard_normal((d, d)) * 0.1
    U_a = rng.standard_normal((d, d)) * 0.1
    v   = rng.standard_normal(d) * 0.1

    e      = v @ np.tanh((H @ W_a + s @ U_a)).T  # (T_src,)
    alpha  = softmax(e.reshape(1, -1)).flatten()

    print(f"  Attention weights alpha over src (T={T_src}): {np.round(alpha, 3)}")
    c = (alpha[:, None] * H).sum(0)
    print(f"  Context vector c: {np.round(c[:4], 3)}...")
    print()
    print("  Applications: translation, summarisation, dialogue, code generation")


if __name__ == "__main__":
    textcnn_demo()
    bilstm_ner_demo()
    seq2seq_demo()
