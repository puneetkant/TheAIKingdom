"""
Working Example 2: RNN Applications — sequence classification, time series prediction
=======================================================================================
Uses sine/cosine sequences for classification and next-step prediction proxy.

Run:  python working_example2.py
"""
from pathlib import Path
try:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import accuracy_score
except ImportError:
    raise SystemExit("pip install numpy matplotlib scikit-learn")

OUTPUT = Path(__file__).parent / "output"
OUTPUT.mkdir(exist_ok=True)

sigmoid = lambda x: 1/(1+np.exp(-np.clip(x,-500,500)))

class SimpleRNN:
    def __init__(self, inp, hid, out, seed=42):
        rng = np.random.default_rng(seed); s = 0.1
        self.Wxh = rng.standard_normal((inp, hid)) * s
        self.Whh = rng.standard_normal((hid, hid)) * s
        self.bh  = np.zeros(hid)
        self.Why = rng.standard_normal((hid, out)) * s
        self.by  = np.zeros(out)
        self.hid = hid

    def forward_seq(self, xs):
        h = np.zeros(self.hid)
        for x in xs:
            h = np.tanh(x @ self.Wxh + h @ self.Whh + self.bh)
        return h @ self.Why + self.by, h

def make_sequence_dataset(n=500, seq_len=15):
    """Binary classification: sin wave (label 0) vs cos wave (label 1)."""
    rng = np.random.default_rng(0)
    X, y = [], []
    for _ in range(n):
        phase = rng.uniform(0, 2*np.pi)
        freq  = rng.uniform(0.5, 2.0)
        t = np.linspace(0, 2*np.pi, seq_len)
        cls = rng.integers(0, 2)
        seq = (np.sin(freq*t+phase) if cls == 0 else np.cos(freq*t+phase))
        seq += rng.standard_normal(seq_len) * 0.2
        X.append(seq.reshape(-1, 1)); y.append(cls)
    return X, np.array(y)

def demo():
    print("=== RNN Sequence Classification ===")
    X, y = make_sequence_dataset(500, 15)
    split = 400
    X_tr, y_tr = X[:split], y[:split]
    X_te, y_te = X[split:], y[split:]

    rnn = SimpleRNN(1, 16, 1)
    lr = 0.02; epochs = 30
    losses = []

    for ep in range(epochs):
        ep_loss = 0
        rng = np.random.default_rng(ep)
        idx = rng.permutation(split)
        for i in idx:
            logit, h = rnn.forward_seq(X_tr[i])
            p = sigmoid(logit[0]); label = float(y_tr[i])
            p = np.clip(p, 1e-7, 1-1e-7)
            ep_loss += -(label*np.log(p) + (1-label)*np.log(1-p))
            # Gradient of output layer only (simplified BPTT)
            dL = p - label
            rnn.Why -= lr * h.reshape(-1,1) * dL
            rnn.by  -= lr * np.array([dL])
        losses.append(ep_loss / split)
        if (ep+1) % 10 == 0:
            print(f"  Epoch {ep+1:3d}: loss={losses[-1]:.4f}")

    preds = []
    for xs in X_te:
        logit, _ = rnn.forward_seq(xs)
        preds.append(1 if sigmoid(logit[0]) > 0.5 else 0)
    acc = accuracy_score(y_te, preds)
    print(f"  Test accuracy: {acc:.4f}")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(losses); ax.set_xlabel("Epoch"); ax.set_ylabel("BCE Loss")
    ax.set_title("RNN Sequence Classification Training")
    plt.tight_layout(); plt.savefig(OUTPUT / "rnn_applications.png"); plt.close()
    print("  Saved rnn_applications.png")

def demo_next_step_prediction():
    """One-step-ahead prediction on a noisy sine wave."""
    print("\n=== Next-Step Prediction ===")
    rng = np.random.default_rng(7)
    t = np.linspace(0, 6 * np.pi, 300)
    signal = np.sin(t) + rng.normal(0, 0.1, len(t))
    # Build (x, y) pairs: predict step i+1 from steps i-4..i
    look_back = 5
    X_seq, y_seq = [], []
    for i in range(look_back, len(signal) - 1):
        X_seq.append(signal[i-look_back:i].reshape(-1, 1))
        y_seq.append(signal[i])
    X_seq = X_seq; y_seq = np.array(y_seq)

    rnn = SimpleRNN(1, 16, 1)
    split = int(0.7 * len(X_seq))
    lr = 0.02
    for ep in range(20):
        for i in range(split):
            logit, h = rnn.forward_seq(X_seq[i])
            target = float(y_seq[i])
            dL = (logit[0] - target)
            rnn.Why -= lr * h.reshape(-1, 1) * dL
            rnn.by  -= lr * np.array([dL])

    preds = [rnn.forward_seq(X_seq[i])[0][0] for i in range(split, len(X_seq))]
    mse = float(np.mean((np.array(preds) - y_seq[split:]) ** 2))
    print(f"  Test MSE (next-step): {mse:.4f}")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(y_seq[split:], label="True", color="steelblue", lw=1)
    ax.plot(preds, label="Predicted", color="tomato", lw=1, linestyle="--")
    ax.set(xlabel="Step", ylabel="Value", title="RNN Next-Step Prediction")
    ax.legend(); plt.tight_layout()
    plt.savefig(OUTPUT / "rnn_prediction.png"); plt.close()
    print("  Saved rnn_prediction.png")


def demo_sequence_length_effect():
    """Show how sequence length affects RNN classification accuracy."""
    print("\n=== Sequence Length Effect ===")
    seq_lengths = [5, 10, 15, 20, 30]
    for seq_len in seq_lengths:
        X, y = make_sequence_dataset(400, seq_len)
        split = 320
        rnn2 = SimpleRNN(1, 16, 1)
        lr = 0.02
        for ep in range(20):
            rng2 = np.random.default_rng(ep)
            for i in rng2.permutation(split):
                logit, h = rnn2.forward_seq(X[i])
                p = sigmoid(logit[0]); label = float(y[i])
                p = np.clip(p, 1e-7, 1-1e-7)
                dL = p - label
                rnn2.Why -= lr * h.reshape(-1,1) * dL
                rnn2.by  -= lr * np.array([dL])
        preds2 = [1 if sigmoid(rnn2.forward_seq(X[i])[0][0]) > 0.5 else 0
                  for i in range(split, len(X))]
        from sklearn.metrics import accuracy_score as acc_fn
        acc = acc_fn(y[split:], preds2)
        print(f"  seq_len={seq_len:3d}: test_acc={acc:.4f}")


if __name__ == "__main__":
    demo()
    demo_next_step_prediction()
    demo_sequence_length_effect()
