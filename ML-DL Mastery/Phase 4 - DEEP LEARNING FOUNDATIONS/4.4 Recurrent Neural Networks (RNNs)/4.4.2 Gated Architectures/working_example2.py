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

def demo_gate_values():
    """Trace individual gate activations through a sequence."""
    print("\n=== LSTM Gate Value Traces ===")
    np.random.seed(0)
    seq_len = 50
    t = np.linspace(0, 4 * np.pi, seq_len)
    xs = np.sin(t)  # input signal

    lstm = LSTMCell(1, 4, seed=99)
    f_vals, i_vals, o_vals, c_vals = [], [], [], []
    for x in xs:
        xh = np.concatenate([np.array([x]), lstm.h])
        f = sigmoid(xh @ lstm.Wf + lstm.bf)
        i = sigmoid(xh @ lstm.Wi + lstm.bi)
        g = np.tanh(xh @ lstm.Wg + lstm.bg)
        o = sigmoid(xh @ lstm.Wo + lstm.bo)
        lstm.c = f * lstm.c + i * g
        lstm.h = o * np.tanh(lstm.c)
        f_vals.append(f[0]); i_vals.append(i[0])
        o_vals.append(o[0]); c_vals.append(lstm.c[0])

    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(xs, label="Input (sin)", color="steelblue")
    axes[0].set_ylabel("Input"); axes[0].legend()
    axes[1].plot(f_vals, label="Forget gate", color="tomato")
    axes[1].plot(i_vals, label="Input gate",  color="green", linestyle="--")
    axes[1].plot(o_vals, label="Output gate", color="purple", linestyle=":")
    axes[1].set_ylabel("Gate (0-1)"); axes[1].legend()
    axes[2].plot(c_vals, label="Cell state c[0]", color="darkorange")
    axes[2].set_ylabel("Cell state"); axes[2].set_xlabel("Time step"); axes[2].legend()
    plt.suptitle("LSTM Gate Values over Sequence")
    plt.tight_layout()
    plt.savefig(OUTPUT / "lstm_gate_traces.png"); plt.close()
    print("  Saved lstm_gate_traces.png")


def demo_gru_vs_lstm_params():
    """Compare parameter counts for LSTM vs GRU at various hidden sizes."""
    print("\n=== LSTM vs GRU Parameter Count ===")
    input_size = 1
    hidden_sizes = [8, 16, 32, 64, 128]
    print(f"  {'Hidden':>8s} {'LSTM params':>14s} {'GRU params':>13s} {'Ratio':>8s}")
    for h in hidden_sizes:
        n = input_size + h
        lstm_params = 4 * (n * h + h)   # 4 gates: Wf,Wi,Wg,Wo + biases
        gru_params  = 3 * (n * h + h)   # 3 gates: Wz,Wr,Wn + biases
        print(f"  {h:>8d} {lstm_params:>14,d} {gru_params:>13,d} {lstm_params/gru_params:>8.2f}x")


if __name__ == "__main__":
    demo()
    demo_gate_values()
    demo_gru_vs_lstm_params()
