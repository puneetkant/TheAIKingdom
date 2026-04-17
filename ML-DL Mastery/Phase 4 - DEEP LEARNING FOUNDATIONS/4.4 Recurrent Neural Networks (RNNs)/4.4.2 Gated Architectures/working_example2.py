"""
Working Example 2: Gated Architectures — LSTM and GRU from scratch
====================================================================
Manual LSTM and GRU cells, gradient flow comparison.

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

sigmoid = lambda x: 1/(1+np.exp(-np.clip(x,-500,500)))

class LSTMCell:
    """LSTM cell: forget / input / output gates + cell state."""
    def __init__(self, input_size, hidden_size, seed=42):
        rng = np.random.default_rng(seed); s = 0.1
        n = input_size + hidden_size
        self.Wf = rng.standard_normal((n, hidden_size)) * s; self.bf = np.ones(hidden_size)  # forget bias=1
        self.Wi = rng.standard_normal((n, hidden_size)) * s; self.bi = np.zeros(hidden_size)
        self.Wg = rng.standard_normal((n, hidden_size)) * s; self.bg = np.zeros(hidden_size)
        self.Wo = rng.standard_normal((n, hidden_size)) * s; self.bo = np.zeros(hidden_size)
        self.h = np.zeros(hidden_size); self.c = np.zeros(hidden_size)

    def step(self, x):
        xh = np.concatenate([x, self.h])
        f = sigmoid(xh @ self.Wf + self.bf)
        i = sigmoid(xh @ self.Wi + self.bi)
        g = np.tanh(xh @ self.Wg + self.bg)
        o = sigmoid(xh @ self.Wo + self.bo)
        self.c = f * self.c + i * g
        self.h = o * np.tanh(self.c)
        return self.h.copy()

class GRUCell:
    """GRU cell: reset / update gates."""
    def __init__(self, input_size, hidden_size, seed=42):
        rng = np.random.default_rng(seed); s = 0.1
        n = input_size + hidden_size
        self.Wz = rng.standard_normal((n, hidden_size)) * s; self.bz = np.zeros(hidden_size)
        self.Wr = rng.standard_normal((n, hidden_size)) * s; self.br = np.zeros(hidden_size)
        self.Wn = rng.standard_normal((n, hidden_size)) * s; self.bn = np.zeros(hidden_size)
        self.h = np.zeros(hidden_size)

    def step(self, x):
        xh = np.concatenate([x, self.h])
        z = sigmoid(xh @ self.Wz + self.bz)
        r = sigmoid(xh @ self.Wr + self.br)
        xrh = np.concatenate([x, r * self.h])
        n = np.tanh(xrh @ self.Wn + self.bn)
        self.h = (1 - z) * self.h + z * n
        return self.h.copy()

def demo():
    print("=== LSTM vs GRU cell comparison ===")
    t = np.linspace(0, 8*np.pi, 200)
    xs = np.sin(t)

    for Cell, name in [(LSTMCell, "LSTM"), (GRUCell, "GRU")]:
        cell = Cell(1, 16)
        hs = np.array([cell.step(np.array([x])) for x in xs])
        print(f"  {name}: hidden state mean={hs.mean():.4f}  std={hs.std():.4f}")

    # Gate value trace for LSTM
    lstm = LSTMCell(1, 8)
    cs, hs_trace = [], []
    for x in xs:
        lstm.step(np.array([x]))
        cs.append(lstm.c.copy()); hs_trace.append(lstm.h.copy())
    cs = np.array(cs); hs_trace = np.array(hs_trace)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    axes[0].plot(xs, label="Input (sin)"); axes[0].set_title("LSTM Input"); axes[0].legend()
    axes[1].imshow(cs.T, aspect="auto", cmap="RdBu", vmin=-1, vmax=1)
    axes[1].set_title("LSTM Cell State (c_t) — 8 units over time")
    axes[1].set_xlabel("Time step")
    plt.tight_layout(); plt.savefig(OUTPUT / "lstm_gru.png"); plt.close()
    print("  Saved lstm_gru.png")

if __name__ == "__main__":
    demo()
