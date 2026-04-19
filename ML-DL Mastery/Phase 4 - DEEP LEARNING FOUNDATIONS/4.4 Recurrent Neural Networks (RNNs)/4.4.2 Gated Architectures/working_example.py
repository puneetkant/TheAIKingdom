"""
Working Example: Gated Architectures (LSTM and GRU)
Covers LSTM cell mechanics, GRU, peephole connections, bidirectional RNNs,
and comparisons.  All implemented from scratch with numpy.
"""
import numpy as np
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_gated")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def sigmoid(z): return 1 / (1 + np.exp(-z.clip(-500, 500)))
def tanh(z):    return np.tanh(z)


# -- LSTM cell (single step) ---------------------------------------------------
class LSTMCell:
    """
    Long Short-Term Memory cell.
    Gates: i=input, f=forget, o=output, g=cell gate
    c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
    h_t = o_t ⊙ tanh(c_t)
    """
    def __init__(self, input_size, hidden_size, rng=None):
        rng   = rng or np.random.default_rng(0)
        s     = np.sqrt(2 / (input_size + hidden_size))
        D     = input_size + hidden_size
        # All 4 gate weights packed in a single matrix for efficiency
        self.W = rng.standard_normal((D, 4 * hidden_size)) * s
        self.b = np.zeros(4 * hidden_size)
        # Forget gate bias initialised to 1 (helps long-term memory early on)
        self.b[hidden_size:2*hidden_size] = 1.0
        self.H = hidden_size

    def step(self, x, h_prev, c_prev):
        """Single LSTM step."""
        H   = self.H
        xh  = np.concatenate([x, h_prev])
        z   = xh @ self.W + self.b   # (4H,)
        i   = sigmoid(z[:H])
        f   = sigmoid(z[H:2*H])
        o   = sigmoid(z[2*H:3*H])
        g   = tanh(z[3*H:])
        c   = f * c_prev + i * g
        h   = o * tanh(c)
        gate_values = {"i": i.mean(), "f": f.mean(), "o": o.mean(),
                       "g": g.mean(), "c_norm": np.linalg.norm(c)}
        return h, c, gate_values

    def forward_seq(self, X_seq):
        """Process full sequence; return all h, c."""
        T   = X_seq.shape[0]
        H   = self.H
        h   = np.zeros(H)
        c   = np.zeros(H)
        hs, cs = [], []
        for t in range(T):
            h, c, _ = self.step(X_seq[t], h, c)
            hs.append(h.copy())
            cs.append(c.copy())
        return np.array(hs), np.array(cs)


# -- GRU cell (single step) ----------------------------------------------------
class GRUCell:
    """
    Gated Recurrent Unit cell.
    z = update gate (replaces forget + input gate)
    r = reset gate
    h_t = (1-z) ⊙ h_{t-1} + z ⊙ tanh(W_x x_t + W_h (r ⊙ h_{t-1}) + b)
    """
    def __init__(self, input_size, hidden_size, rng=None):
        rng  = rng or np.random.default_rng(0)
        s    = np.sqrt(2 / (input_size + hidden_size))
        D    = input_size + hidden_size
        # z and r gates (2H)
        self.W_zr = rng.standard_normal((D, 2 * hidden_size)) * s
        self.b_zr = np.zeros(2 * hidden_size)
        # candidate gate
        self.W_h  = rng.standard_normal((D, hidden_size)) * s
        self.b_h  = np.zeros(hidden_size)
        self.H    = hidden_size

    def step(self, x, h_prev):
        H   = self.H
        xh  = np.concatenate([x, h_prev])
        zr  = sigmoid(xh @ self.W_zr + self.b_zr)
        z   = zr[:H]
        r   = zr[H:]
        xrh = np.concatenate([x, r * h_prev])
        h_cand = tanh(xrh @ self.W_h + self.b_h)
        h = (1 - z) * h_prev + z * h_cand
        return h, {"z": z.mean(), "r": r.mean()}

    def forward_seq(self, X_seq):
        T = X_seq.shape[0]
        h = np.zeros(self.H)
        hs = []
        for t in range(T):
            h, _ = self.step(X_seq[t], h)
            hs.append(h.copy())
        return np.array(hs)


# -- 1. LSTM mechanics ---------------------------------------------------------
def lstm_mechanics():
    print("=== LSTM Cell Mechanics ===")
    print("  LSTM = 4 gates running in parallel:")
    print("    Forget gate f:  sigma(W_f·[h,x]+b_f) — what to discard from cell")
    print("    Input gate i:   sigma(W_i·[h,x]+b_i) — what new info to store")
    print("    Cell gate g:    tanh(W_g·[h,x]+b_g) — candidate new content")
    print("    Output gate o:  sigma(W_o·[h,x]+b_o) — what to expose as h")
    print()
    print("  Cell state update:")
    print("    c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t")
    print("    h_t = o_t ⊙ tanh(c_t)")
    print()
    print("  Cell state c is the 'memory highway' — gradients flow with less decay")

    rng   = np.random.default_rng(1)
    lstm  = LSTMCell(input_size=4, hidden_size=8, rng=rng)
    x     = rng.standard_normal(4)
    h, c, gates = lstm.step(x, np.zeros(8), np.zeros(8))
    print(f"\n  Single step example (input_size=4, hidden_size=8):")
    print(f"  Gate means: i={gates['i']:.3f}  f={gates['f']:.3f}  o={gates['o']:.3f}  g={gates['g']:.3f}")
    print(f"  h norm: {np.linalg.norm(h):.4f}   c norm: {gates['c_norm']:.4f}")


# -- 2. GRU mechanics ----------------------------------------------------------
def gru_mechanics():
    print("\n=== GRU Cell Mechanics ===")
    print("  GRU simplifies LSTM: merges forget+input into one update gate")
    print("  Update gate z:  sigma(W_z·[x,h])  — balance between past and new")
    print("  Reset gate r:   sigma(W_r·[x,h])  — how much past to use for candidate")
    print("  Candidate:      tanh(W_h·[x, r⊙h])")
    print("  h_t = (1-z)⊙h_{t-1} + z⊙h_t")
    print()
    print("  Fewer parameters than LSTM (no separate cell state)")

    rng = np.random.default_rng(2)
    gru = GRUCell(input_size=4, hidden_size=8, rng=rng)
    x   = rng.standard_normal(4)
    h, gates = gru.step(x, np.zeros(8))
    print(f"\n  Single step: z={gates['z']:.3f}  r={gates['r']:.3f}  h_norm={np.linalg.norm(h):.4f}")


# -- 3. Long-term dependency capture -------------------------------------------
def long_term_dependency():
    print("\n=== Long-Term Dependency: LSTM vs Vanilla RNN ===")
    rng = np.random.default_rng(3)
    T   = 50
    input_size = 4

    # Vanilla RNN gradient norms
    W_rnn  = rng.standard_normal((input_size + 8, 8)) * 0.5
    grad_norms_rnn = []
    grad = np.eye(8)
    for _ in range(T):
        dW = 0.5 * W_rnn[input_size:].T   # tanh' ~= 0.5
        grad = dW @ grad
        grad_norms_rnn.append(np.linalg.norm(grad))

    # LSTM cell-state gradient (dominated by forget gate, not matrix multiply)
    f = 0.7   # typical forget gate value
    grad_norms_lstm = [f**t for t in range(1, T+1)]

    print(f"  Gradient norm after T={T} steps:")
    print(f"  Vanilla RNN: {grad_norms_rnn[-1]:.4e}")
    print(f"  LSTM (f=0.7): {grad_norms_lstm[-1]:.4e}")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(range(1, T+1), grad_norms_rnn, label="Vanilla RNN", lw=2)
    ax.semilogy(range(1, T+1), grad_norms_lstm, label="LSTM (approx)", lw=2)
    ax.set(xlabel="Steps back in time", ylabel="Gradient norm (log)",
           title="Gradient Flow: Vanilla RNN vs LSTM")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "gradient_flow.png")
    plt.savefig(path, dpi=90); plt.close()
    print(f"  Plot saved: {path}")


# -- 4. Bidirectional RNN ------------------------------------------------------
def bidirectional_rnn():
    print("\n=== Bidirectional RNN ===")
    print("  Process sequence left->right AND right->left; concatenate hidden states")
    print("  h_t = [h_forward_t; h_backward_t]  (size = 2 x hidden_size)")
    print()
    print("  Use cases: NLP (see full sentence context), not suitable for forecasting")

    rng = np.random.default_rng(4)
    T, d = 8, 3
    H    = 6
    X    = rng.standard_normal((T, d))

    fwd_lstm = LSTMCell(d, H, rng)
    bwd_lstm = LSTMCell(d, H, rng)

    hs_fwd, _ = fwd_lstm.forward_seq(X)
    hs_bwd, _ = bwd_lstm.forward_seq(X[::-1])  # reverse time
    hs_bwd    = hs_bwd[::-1]                   # flip back

    hs_bidir = np.concatenate([hs_fwd, hs_bwd], axis=-1)
    print(f"\n  Sequence: T={T}, input_size={d}, hidden_size={H}")
    print(f"  Fwd hidden: {hs_fwd.shape}")
    print(f"  Bwd hidden: {hs_bwd.shape}")
    print(f"  BiDir out:  {hs_bidir.shape} (2×H per time step)")


# -- 5. Stacked LSTM -----------------------------------------------------------
def stacked_lstm():
    print("\n=== Stacked (Deep) LSTM ===")
    print("  Stack multiple LSTM layers; output h of layer l is input to layer l+1")
    print()
    rng = np.random.default_rng(5)
    T, d = 10, 4
    X    = rng.standard_normal((T, d))

    layer_sizes = [d, 16, 8, 4]
    h_seq = X
    for layer_idx in range(len(layer_sizes) - 1):
        in_sz  = layer_sizes[layer_idx]
        out_sz = layer_sizes[layer_idx + 1]
        lstm   = LSTMCell(in_sz, out_sz, rng)
        hs, _  = lstm.forward_seq(h_seq)
        h_seq  = hs
        print(f"  Layer {layer_idx+1} ({in_sz}->{out_sz}): output shape {hs.shape}")

    print(f"\n  Final representation: shape {h_seq.shape}")


# -- 6. LSTM vs GRU comparison ------------------------------------------------
def lstm_gru_comparison():
    print("\n=== LSTM vs GRU Comparison ===")
    print(f"  {'Aspect':<25} {'LSTM':<35} {'GRU'}")
    rows = [
        ("Gates",             "Forget, Input, Output, Cell",  "Update, Reset (2 gates)"),
        ("State",             "h + c (two vectors)",          "h only (one vector)"),
        ("Parameters",        "4×(d+H)×H + 4H",              "3×(d+H)×H + 3H"),
        ("Memory capacity",   "Higher (separate cell)",       "Lower"),
        ("Training speed",    "Slower (more params)",         "Faster"),
        ("Short sequences",   "Good",                         "Often comparable"),
        ("Long sequences",    "Strong (explicit cell state)", "Strong (empirically)"),
        ("Vanishing gradient","Better than vanilla RNN",      "Better than vanilla RNN"),
        ("Recommended when",  "Long-range deps, NLP, speech","Less data, faster training"),
    ]
    for name, lstm_val, gru_val in rows:
        print(f"  {name:<25} {lstm_val:<35} {gru_val}")

    # Parameter count comparison
    d, H = 128, 256
    lstm_params = 4 * (d + H) * H + 4*H
    gru_params  = 3 * (d + H) * H + 3*H
    print(f"\n  Example (d=128, H=256):")
    print(f"  LSTM params: {lstm_params:,}")
    print(f"  GRU params:  {gru_params:,}  ({gru_params/lstm_params*100:.0f}% of LSTM)")


if __name__ == "__main__":
    lstm_mechanics()
    gru_mechanics()
    long_term_dependency()
    bidirectional_rnn()
    stacked_lstm()
    lstm_gru_comparison()
