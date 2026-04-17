"""
Working Example 2: Transformer Architecture Deep Dive — multi-head attention, positional encoding
===================================================================================================
Implements scaled dot-product attention and multi-head attention from scratch.

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

def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)

def scaled_dot_product_attention(Q, K, V, mask=None):
    """Q, K, V: (batch, heads, seq, d_k)"""
    d_k = Q.shape[-1]
    scores = (Q @ K.swapaxes(-2, -1)) / np.sqrt(d_k)
    if mask is not None:
        scores = np.where(mask, scores, -1e9)
    attn = softmax(scores)
    return attn @ V, attn

def multi_head_attention(X, W_q, W_k, W_v, W_o, n_heads):
    """X: (seq, d_model)"""
    seq, d_model = X.shape
    d_k = d_model // n_heads
    Q = X @ W_q; K = X @ W_k; V = X @ W_v
    # Reshape to (1, n_heads, seq, d_k)
    Q = Q.reshape(seq, n_heads, d_k).transpose(1, 0, 2)[None]
    K = K.reshape(seq, n_heads, d_k).transpose(1, 0, 2)[None]
    V = V.reshape(seq, n_heads, d_k).transpose(1, 0, 2)[None]
    out, attn = scaled_dot_product_attention(Q, K, V)
    out = out[0].transpose(1, 0, 2).reshape(seq, d_model)
    return out @ W_o, attn[0]

def sinusoidal_pe(seq_len, d_model):
    PE = np.zeros((seq_len, d_model))
    pos = np.arange(seq_len)[:, None]
    div = np.exp(np.arange(0, d_model, 2) * (-np.log(10000) / d_model))
    PE[:, 0::2] = np.sin(pos * div); PE[:, 1::2] = np.cos(pos * div)
    return PE

def demo():
    print("=== Transformer Architecture Deep Dive ===")
    np.random.seed(3)
    seq, d_model, n_heads = 8, 16, 4
    d_k = d_model // n_heads
    X = np.random.randn(seq, d_model)
    W_q = np.random.randn(d_model, d_model) * 0.1
    W_k = np.random.randn(d_model, d_model) * 0.1
    W_v = np.random.randn(d_model, d_model) * 0.1
    W_o = np.random.randn(d_model, d_model) * 0.1
    out, attn = multi_head_attention(X, W_q, W_k, W_v, W_o, n_heads)
    print(f"  Input: {X.shape}  Output: {out.shape}")
    print(f"  Attention: {attn.shape}  (heads × seq × seq)")

    PE = sinusoidal_pe(32, d_model)
    print(f"  Positional encoding shape: {PE.shape}")

    fig, axes = plt.subplots(1, 2, figsize=(9, 3))
    axes[0].imshow(attn[0], cmap="Blues"); axes[0].set_title("Attention (head 0)")
    axes[0].set_xlabel("Key"); axes[0].set_ylabel("Query")
    axes[1].imshow(PE.T, aspect="auto", cmap="RdBu"); axes[1].set_title("Sinusoidal PE")
    axes[1].set_xlabel("Position"); axes[1].set_ylabel("Dimension")
    plt.tight_layout(); plt.savefig(OUTPUT / "transformer_arch.png"); plt.close()
    print("  Saved transformer_arch.png")

if __name__ == "__main__":
    demo()
