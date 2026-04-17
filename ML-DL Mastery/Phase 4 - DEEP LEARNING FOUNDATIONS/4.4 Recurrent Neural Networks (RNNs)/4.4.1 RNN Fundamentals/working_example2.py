"""
Working Example 2: RNN Fundamentals — vanilla RNN from scratch on sine wave
=============================================================================
Manual RNN cell, sequence processing, gradient through time demo.

Run:  python working_example2.py
"""
from pathlib import Path
try:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit("pip install numpy matplotlib")

OUTPUT = Path(__file__).parent / "output"
OUTPUT.mkdir(exist_ok=True)

class VanillaRNN:
    """Single-layer RNN: h_t = tanh(W_hh * h_{t-1} + W_xh * x_t + b_h)."""
    def __init__(self, input_size, hidden_size, output_size, seed=42):
        rng = np.random.default_rng(seed)
        s = 0.01
        self.W_xh = rng.standard_normal((input_size, hidden_size)) * s
        self.W_hh = rng.standard_normal((hidden_size, hidden_size)) * s
        self.b_h  = np.zeros(hidden_size)
        self.W_hy = rng.standard_normal((hidden_size, output_size)) * s
        self.b_y  = np.zeros(output_size)
        self.hidden_size = hidden_size

    def forward(self, xs):
        """xs: list of (1,) inputs. Returns predictions and hidden states."""
        h = np.zeros(self.hidden_size)
        hs, ys = [], []
        for x in xs:
            h = np.tanh(x @ self.W_xh + h @ self.W_hh + self.b_h)
            y = h @ self.W_hy + self.b_y
            hs.append(h.copy()); ys.append(y.copy())
        return np.array(ys), hs

def make_sine_sequences(n=300, seq_len=20, pred_steps=1):
    t = np.linspace(0, 4*np.pi, n+pred_steps)
    s = np.sin(t)
    X, y = [], []
    for i in range(n - seq_len):
        X.append(s[i:i+seq_len].reshape(-1, 1))
        y.append(s[i+seq_len])
    return X, np.array(y)

def demo():
    print("=== Vanilla RNN on Sine Wave Prediction ===")
    X_seqs, y_true = make_sine_sequences(300, seq_len=10)
    rnn = VanillaRNN(input_size=1, hidden_size=16, output_size=1)

    preds = []
    for xs in X_seqs:
        ys, _ = rnn.forward(xs)
        preds.append(ys[-1, 0])
    preds = np.array(preds)

    mse = np.mean((preds - y_true)**2)
    print(f"  Random init MSE: {mse:.4f}  (before training — expected ~0.5)")

    # Show hidden state dynamics for one sequence
    ys, hs = rnn.forward(X_seqs[0])
    hs_arr = np.array(hs)
    print(f"  Sequence length: {len(X_seqs[0])}  Hidden state shape per step: {hs_arr.shape}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    t = np.arange(len(y_true))
    axes[0].plot(t[:80], y_true[:80], label="True"); axes[0].plot(t[:80], preds[:80], label="Pred")
    axes[0].set_title("RNN Sine Prediction (random init)"); axes[0].legend()
    axes[1].imshow(hs_arr.T, aspect="auto", cmap="RdBu")
    axes[1].set_title("Hidden State Heatmap (seq 0)"); axes[1].set_xlabel("Time step")
    plt.tight_layout(); plt.savefig(OUTPUT / "rnn_fundamentals.png"); plt.close()
    print("  Saved rnn_fundamentals.png")

if __name__ == "__main__":
    demo()
