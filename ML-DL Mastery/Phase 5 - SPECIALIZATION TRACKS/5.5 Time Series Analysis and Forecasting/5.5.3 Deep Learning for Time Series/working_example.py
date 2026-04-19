"""
Working Example: Deep Learning for Time Series
Covers 1D-CNN, LSTM/GRU forecasting, Transformer-based models,
N-BEATS, PatchTST, and multi-step forecasting approaches.
"""
import numpy as np
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_dl_ts")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def softmax(z):
    e = np.exp(z - z.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


# -- Helper: generate dataset --------------------------------------------------
def gen_dataset(T=500, lookback=24, horizon=12, seed=0):
    rng = np.random.default_rng(seed)
    t   = np.arange(T)
    y   = (0.1*t + 50
           + 8*np.sin(2*np.pi*t/12)
           + 3*np.sin(2*np.pi*t/6)
           + rng.normal(0, 1.5, T))
    # Normalise
    mu, sigma = y.mean(), y.std()
    y = (y - mu) / sigma

    X, Y = [], []
    for i in range(T - lookback - horizon + 1):
        X.append(y[i:i+lookback])
        Y.append(y[i+lookback:i+lookback+horizon])
    return np.array(X), np.array(Y)


# -- 1. LSTM for time series ---------------------------------------------------
class LSTMCell:
    """Single LSTM cell — numpy implementation."""
    def __init__(self, input_dim, hidden_dim, seed=0):
        rng  = np.random.default_rng(seed)
        s    = 1.0 / np.sqrt(hidden_dim)
        d    = input_dim + hidden_dim
        self.Wf = rng.uniform(-s, s, (d, hidden_dim))
        self.Wi = rng.uniform(-s, s, (d, hidden_dim))
        self.Wg = rng.uniform(-s, s, (d, hidden_dim))
        self.Wo = rng.uniform(-s, s, (d, hidden_dim))
        self.bf = np.zeros(hidden_dim)
        self.bi = np.zeros(hidden_dim)
        self.bg = np.zeros(hidden_dim)
        self.bo = np.zeros(hidden_dim)

    def forward_seq(self, x_seq):
        """x_seq: (T, input_dim)"""
        T, d = x_seq.shape
        hd   = self.Wf.shape[1]
        h    = np.zeros(hd); c = np.zeros(hd)
        hs   = []
        for t in range(T):
            xh  = np.concatenate([x_seq[t], h])
            f   = 1/(1+np.exp(-(xh @ self.Wf + self.bf)))
            i_  = 1/(1+np.exp(-(xh @ self.Wi + self.bi)))
            g   = np.tanh(xh @ self.Wg + self.bg)
            o   = 1/(1+np.exp(-(xh @ self.Wo + self.bo)))
            c   = f*c + i_*g
            h   = o * np.tanh(c)
            hs.append(h.copy())
        return np.array(hs), h, c


def lstm_demo():
    print("=== LSTM Time Series Forecasting ===")
    print()
    print("  LSTM gate equations:")
    print("    f_t = sigma(W_f·[h_{t-1}, x_t] + b_f)   — forget gate")
    print("    i_t = sigma(W_i·[h_{t-1}, x_t] + b_i)   — input gate")
    print("    g_t = tanh(W_g·[h_{t-1}, x_t] + b_g) — cell gate")
    print("    o_t = sigma(W_o·[h_{t-1}, x_t] + b_o)   — output gate")
    print("    c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t")
    print("    h_t = o_t ⊙ tanh(c_t)")
    print()

    lookback = 24; horizon = 1
    X, Y = gen_dataset(T=300, lookback=lookback, horizon=horizon)
    X_tr, Y_tr = X[:200], Y[:200]
    X_te, Y_te = X[200:], Y[200:]
    print(f"  Dataset: {len(X_tr)} train, {len(X_te)} test  lookback={lookback}")

    cell   = LSTMCell(input_dim=1, hidden_dim=16)
    Wout   = np.random.default_rng(1).standard_normal((16, 1)) * 0.1
    lr     = 0.005; losses = []

    for epoch in range(30):
        preds = []
        for i in range(len(X_tr)):
            seq       = X_tr[i].reshape(-1, 1)
            hs, h, _  = cell.forward_seq(seq)
            pred      = h @ Wout
            preds.append(pred.squeeze())
        preds  = np.array(preds)
        targets = Y_tr.squeeze()
        mse    = ((preds - targets)**2).mean()
        losses.append(mse)

        # Crude weight update (gradient descent)
        Wout  -= lr * (preds - targets).mean()

        if epoch % 10 == 0:
            print(f"  Epoch {epoch+1:>2}: MSE={mse:.4f}")

    # Test
    preds_te = []
    for i in range(len(X_te)):
        seq      = X_te[i].reshape(-1, 1)
        _, h, _  = cell.forward_seq(seq)
        pred     = h @ Wout
        preds_te.append(pred.squeeze())
    test_mse = ((np.array(preds_te) - Y_te.squeeze())**2).mean()
    print(f"  Test MSE: {test_mse:.4f}")


# -- 2. 1D CNN for time series -------------------------------------------------
def cnn1d_demo():
    print("\n=== 1D-CNN Time Series Forecasting ===")
    print()
    print("  Causal 1D convolution: each output depends only on past inputs")
    print("  Dilated 1D conv: receptive field grows exponentially")
    print("  WaveNet-style: d=1,2,4,8,... -> large context window")
    print()

    # Simple 1D convolution
    rng     = np.random.default_rng(0)
    T_in    = 24
    n_filt  = 8; k = 3

    kernel  = rng.standard_normal((n_filt, 1, k)) * 0.1   # (out, in, k)
    x_in    = rng.standard_normal(T_in)

    # Causal conv (pad left)
    out = np.zeros((n_filt, T_in))
    padded = np.concatenate([np.zeros(k-1), x_in])
    for f in range(n_filt):
        for t in range(T_in):
            out[f, t] = padded[t:t+k] @ kernel[f, 0]

    print(f"  Input: ({T_in},)  Kernel: ({n_filt}, 1, {k})")
    print(f"  Output: {out.shape}  (same length, causal)")
    print(f"  Output max per filter: {out.max(axis=1).round(3)}")

    print()
    print("  TCN (Temporal Convolutional Network):")
    print("    Stack of dilated causal convolutions + residual connections")
    print("    Receptive field = (k-1) * 2^{L-1} * n_stacks + 1")
    rf = (k-1) * (2**4 - 1) * 2 + 1
    print(f"    k={k}, L=4 layers, 2 stacks -> RF={rf}")


# -- 3. Transformer-based models -----------------------------------------------
def transformer_ts():
    print("\n=== Transformer Models for Time Series ===")
    models = [
        ("Informer",      2021, "ProbSparse attention; long-range; O(T log T)"),
        ("Autoformer",    2021, "Auto-Correlation mechanism; series decomp layer"),
        ("FEDformer",     2022, "Frequency-enhanced decomposed attention"),
        ("PatchTST",      2023, "Time series patching; channel-independent; SOTA"),
        ("iTransformer",  2024, "Inverted: variate tokens; multivariate focus"),
        ("TimesNet",      2023, "1D->2D transformation; CNNs on 2D spectrum"),
        ("N-BEATS",       2020, "Pure MLP; basis expansion; interpretable"),
        ("N-HiTS",        2022, "Multi-scale sampling + interpolation; SOTA"),
        ("TimesFM",       2024, "Google; foundation model; 200M params"),
        ("Moirai",        2024, "Salesforce; universal forecasting model"),
        ("Chronos",       2024, "Amazon; tokenised time series LM"),
    ]
    print(f"  {'Model':<14} {'Year'} {'Notes'}")
    print(f"  {'-'*14} {'-'*4} {'-'*55}")
    for m, y, d in models:
        print(f"  {m:<14} {y}  {d}")
    print()
    print("  PatchTST key ideas:")
    print("    Divide time series into patches (P=16 tokens per patch)")
    print("    Each patch = one token (like image patches in ViT)")
    print("    Channel-independent: each variate processed separately")
    print("    Pre-training: masked patch prediction (MPM)")
    print()

    # Simulate patching
    T_in = 96; P = 16; stride = 8
    n_patches = (T_in - P) // stride + 1
    print(f"  Input: L={T_in}  Patch={P}  Stride={stride}  n_patches={n_patches}")

    # N-BEATS overview
    print()
    print("  N-BEATS (Neural Basis Expansion Analysis):")
    print("    Residual blocks: FC layers -> backcast theta_b and forecast theta_f")
    print("    Basis expansion: backcast = Sigma theta_b · v_b(t)  (e.g. polynomials/Fourier)")
    print("    Stack: Trend stack (polynomial bases) + Seasonality stack (Fourier)")
    print("    Fully interpretable; no domain knowledge required")


if __name__ == "__main__":
    lstm_demo()
    cnn1d_demo()
    transformer_ts()
