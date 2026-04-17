"""
Working Example: RNN Applications
Covers sequence-to-sequence models, encoder-decoder architecture, text
generation, time series forecasting, and named entity recognition simulation.
"""
import numpy as np
from collections import defaultdict
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_rnn_apps")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def sigmoid(z): return 1 / (1 + np.exp(-z.clip(-500, 500)))
def tanh(z):    return np.tanh(z)
def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


# ── Minimal char-level RNN ────────────────────────────────────────────────────
class CharRNN:
    """Minimal char-level RNN (Karpathy 2015 style)."""
    def __init__(self, vocab_size, hidden_size=64, rng=None):
        rng  = rng or np.random.default_rng(0)
        s    = 0.01
        H    = hidden_size
        V    = vocab_size
        self.W_xh = rng.standard_normal((V, H)) * s
        self.W_hh = rng.standard_normal((H, H)) * s
        self.b_h  = np.zeros(H)
        self.W_hy = rng.standard_normal((H, V)) * s
        self.b_y  = np.zeros(V)
        self.H    = H
        self.V    = V

    def forward(self, inputs, h):
        xs, hs, ys, ps = [], [], [h], [], []
        for x_idx in inputs:
            x = np.zeros(self.V); x[x_idx] = 1
            h = tanh(x @ self.W_xh + h @ self.W_hh + self.b_h)
            y = h @ self.W_hy + self.b_y
            p = softmax(y)
            xs.append(x); hs.append(h); ys.append(y); ps.append(p)
        return xs, hs[:-1], ys, ps, h

    def sample(self, seed_idx, h, length, temperature=1.0):
        x = np.zeros(self.V); x[seed_idx] = 1
        out = [seed_idx]
        for _ in range(length - 1):
            h = tanh(x @ self.W_xh + h @ self.W_hh + self.b_h)
            y = h @ self.W_hy + self.b_y
            y = y / temperature
            p = softmax(y)
            idx = np.random.choice(self.V, p=p)
            x = np.zeros(self.V); x[idx] = 1
            out.append(idx)
        return out


# ── 1. Character-level language model ────────────────────────────────────────
def char_language_model():
    print("=== Character-Level Language Model ===")
    text = "the quick brown fox jumps over the lazy dog " * 3
    chars = sorted(set(text))
    c2i   = {c: i for i, c in enumerate(chars)}
    i2c   = {i: c for c, i in c2i.items()}
    V     = len(chars)
    T     = len(text) - 1

    print(f"  Text length: {len(text)}  Vocab: {V} chars")
    print(f"  Chars: {''.join(chars[:20])}...")

    rng   = np.random.default_rng(0)
    rnn   = CharRNN(V, hidden_size=64, rng=rng)
    inputs  = [c2i[c] for c in text[:-1]]
    targets = [c2i[c] for c in text[1:]]

    h = np.zeros(64)
    smooth_loss = -np.log(1/V) * T   # init
    losses = []

    for epoch in range(30):
        pos    = 0
        h_prev = np.zeros(64)
        epoch_loss = 0
        while pos + 25 <= len(inputs):
            chunk_in  = inputs[pos:pos+25]
            chunk_out = targets[pos:pos+25]
            xs, hs, ys, ps, h_new = rnn.forward(chunk_in, h_prev.copy())

            loss = -sum(np.log(ps[t][chunk_out[t]] + 1e-9) for t in range(len(chunk_in)))
            epoch_loss += loss

            # Simple gradient approximation (RTRL style — just for demo)
            lr = 0.02
            for t in reversed(range(len(chunk_in))):
                dy = ps[t].copy()
                dy[chunk_out[t]] -= 1
                dW_hy = np.outer(hs[t], dy)
                rnn.W_hy -= lr * np.clip(dW_hy, -5, 5)
                rnn.b_y  -= lr * np.clip(dy, -5, 5)
            h_prev = h_new
            pos   += 25
        losses.append(epoch_loss / len(inputs))

    print(f"\n  Training 30 epochs:")
    print(f"  Start loss: {losses[0]:.4f}")
    print(f"  End loss:   {losses[-1]:.4f}")

    # Sample
    seed = c2i['t']
    generated = rnn.sample(seed, np.zeros(64), length=50, temperature=0.8)
    generated_text = ''.join(i2c[i] for i in generated)
    print(f"\n  Generated text (T=0.8): '{generated_text}'")
    print(f"  (After only 30 epochs on short text — quality limited)")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(losses, lw=2); ax.set(xlabel="Epoch", ylabel="Loss",
           title="Char-RNN Training Loss"); ax.grid(True, alpha=0.3)
    path = os.path.join(OUTPUT_DIR, "charlm_loss.png")
    plt.tight_layout(); plt.savefig(path, dpi=90); plt.close()
    print(f"  Loss plot: {path}")


# ── 2. Encoder–Decoder (seq2seq) ──────────────────────────────────────────────
def seq2seq_architecture():
    print("\n=== Encoder–Decoder (Seq2Seq) Architecture ===")
    print("  Encoder: reads source sequence → context vector c = h_T")
    print("  Decoder: generates target sequence conditioned on c")
    print()
    print("  Architecture:")
    print("  x_1..x_T → [ENCODER LSTM] → c  ← final hidden state")
    print("              c, <sos> → [DECODER LSTM] → y_1")
    print("              c, y_1   → [DECODER LSTM] → y_2")
    print("              ...      → [DECODER LSTM] → <eos>")
    print()
    print("  Limitations of fixed context vector:")
    print("    Long sequences → bottleneck in single c vector")
    print("    Solution: attention mechanism (allows decoder to look at all encoder states)")

    rng     = np.random.default_rng(10)
    src_len = 6
    tgt_len = 4
    d       = 5
    H       = 8
    V       = 10  # toy vocab

    # Simulate encoder
    enc_input  = rng.standard_normal((src_len, d))
    W_enc = rng.standard_normal((d + H, H)) * 0.01
    b_enc = np.zeros(H)
    h = np.zeros(H)
    for t in range(src_len):
        h = tanh(np.concatenate([enc_input[t], h]) @ W_enc + b_enc)
    context = h.copy()

    # Simulate decoder (single pass)
    W_dec = rng.standard_normal((H + H, H)) * 0.01   # [context + h_prev] → h
    b_dec = np.zeros(H)
    W_out = rng.standard_normal((H, V)) * 0.01
    b_out = np.zeros(V)
    h_dec = context.copy()
    print(f"\n  Encoder output (context) shape: {context.shape}")
    for t in range(tgt_len):
        h_dec = tanh(np.concatenate([context, h_dec]) @ W_dec + b_dec)
        logit = h_dec @ W_out + b_out
        pred  = logit.argmax()
        print(f"  Decoder step {t+1}: predicted token={pred}")


# ── 3. Time series forecasting ────────────────────────────────────────────────
def time_series_forecasting():
    print("\n=== Time Series Forecasting with RNN ===")
    rng  = np.random.default_rng(20)
    T    = 200
    t    = np.linspace(0, 4*np.pi, T)
    y    = np.sin(t) + 0.5*np.sin(3*t) + rng.normal(0, 0.1, T)

    # Create sliding window dataset
    seq_len = 10
    X_windows = np.array([y[i:i+seq_len]   for i in range(T - seq_len)])
    y_targets = np.array([y[i + seq_len]    for i in range(T - seq_len)])

    n_train = int(len(X_windows) * 0.8)
    X_train, X_test = X_windows[:n_train], X_windows[n_train:]
    y_train, y_test = y_targets[:n_train], y_targets[n_train:]

    print(f"  Signal length: {T}  Seq len: {seq_len}")
    print(f"  Train/Test: {n_train}/{len(X_test)} windows")

    # Simple linear RNN-style regression (Reservoir Computing toy)
    H   = 20
    W_in  = rng.standard_normal((1, H)) * 0.5
    W_res = rng.standard_normal((H, H)) * 0.1
    # Scale reservoir to spectral radius 0.9
    sr = max(abs(np.linalg.eigvals(W_res)))
    W_res = W_res / sr * 0.9

    def run_reservoir(X_batch):
        """Echo State Network-style forward pass (untrained reservoir)."""
        states = []
        for x_seq in X_batch:
            h = np.zeros(H)
            for x in x_seq:
                h = tanh(np.array([x]) @ W_in + h @ W_res)
            states.append(h)
        return np.array(states)

    S_train = run_reservoir(X_train)
    S_test  = run_reservoir(X_test)

    # Ridge regression readout
    lam = 1e-3
    W_out = np.linalg.solve(S_train.T @ S_train + lam * np.eye(H), S_train.T @ y_train)
    y_pred = S_test @ W_out

    mae  = np.mean(np.abs(y_pred - y_test))
    rmse = np.sqrt(np.mean((y_pred - y_test)**2))
    print(f"  Echo State Network: MAE={mae:.4f}  RMSE={rmse:.4f}")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(y_test[:50], label="True", lw=2)
    ax.plot(y_pred[:50], label="Predicted", lw=2, linestyle="--")
    ax.set(xlabel="Time step", ylabel="Value",
           title="Time Series Forecast (Echo State Network)"); ax.legend()
    ax.grid(True, alpha=0.3)
    path = os.path.join(OUTPUT_DIR, "ts_forecast.png")
    plt.tight_layout(); plt.savefig(path, dpi=90); plt.close()
    print(f"  Forecast plot: {path}")


# ── 4. RNN for sequence tagging (NER simulation) ──────────────────────────────
def sequence_tagging():
    print("\n=== Sequence Tagging: Named Entity Recognition (NER) ===")
    print("  Many-to-Many: one label per token")
    print("  BIO tagging scheme:")
    print("    B-PER: Beginning of PERSON entity")
    print("    I-PER: Inside of PERSON entity")
    print("    B-ORG: Beginning of ORG")
    print("    I-ORG: Inside of ORG")
    print("    O:     Outside (not an entity)")
    print()
    print("  Example:")
    tokens = ["Steve", "Jobs", "founded",  "Apple", "Inc", "."]
    labels = ["B-PER","I-PER", "O",        "B-ORG","I-ORG","O"]
    for tok, lab in zip(tokens, labels):
        print(f"    {tok:<12}: {lab}")

    print()
    print("  Architecture: BiLSTM-CRF")
    print("    1. Token embeddings (or character CNN + word embedding)")
    print("    2. BiLSTM → context-aware representations")
    print("    3. CRF layer → globally optimal label sequence")
    print("       (enforces tag constraints: B before I, no I-ORG after B-PER)")
    print()
    print("  Metrics: entity-level F1 (seqeval library)")


# ── 5. Applications summary ───────────────────────────────────────────────────
def applications_summary():
    print("\n=== RNN Applications Summary ===")
    apps = [
        ("Language modelling",    "next word/char prediction",       "GPT (now Transformer)"),
        ("Machine translation",   "seq2seq + attention",             "Google Translate early"),
        ("Speech recognition",    "acoustic → phoneme → word",       "DeepSpeech"),
        ("Text generation",       "char/word level sampling",        "Karpathy's char-RNN"),
        ("Sentiment analysis",    "sequence → sentiment class",      "BiLSTM classifier"),
        ("NER / POS tagging",     "BIO tagging + BiLSTM-CRF",       "Stanford NLP"),
        ("Time series forecast",  "sliding window → next value",     "LSTNet, ES-RNN"),
        ("Music generation",      "note sequence modelling",         "Magenta LSTM"),
        ("Video captioning",      "CNN features + LSTM decoder",     "Show and Tell"),
        ("Anomaly detection",     "reconstruction error on seq",     "LSTM-VAE"),
    ]
    print(f"  {'Application':<25} {'Approach':<35} {'Example'}")
    for app, approach, example in apps:
        print(f"  {app:<25} {approach:<35} {example}")


if __name__ == "__main__":
    char_language_model()
    seq2seq_architecture()
    time_series_forecasting()
    sequence_tagging()
    applications_summary()
