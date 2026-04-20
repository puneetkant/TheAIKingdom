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

def demo_causal_mask():
    """Show how a causal (upper-triangular) mask prevents attending to future tokens."""
    print("\n=== Causal Mask ===" )
    seq = 6
    # mask[i,j] = True if position i is allowed to attend to position j
    mask = np.tril(np.ones((seq, seq), dtype=bool))
    print("  Causal mask (True = allowed):")
    for row in mask:
        print("  ", [int(v) for v in row])

    # Show attention scores before/after masking
    np.random.seed(7)
    scores = np.random.randn(seq, seq)
    masked_scores = np.where(mask, scores, -1e9)
    attn = softmax(masked_scores)
    print(f"  Attn row 0 (should be ~1.0 at pos 0): {attn[0].round(3)}")
    print(f"  Attn row 3 (attends to pos 0-3):      {attn[3].round(3)}")

    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    axes[0].imshow(mask, cmap="Greens")
    axes[0].set_title("Causal Mask"); axes[0].set_xlabel("Key"); axes[0].set_ylabel("Query")
    axes[1].imshow(attn, cmap="Blues")
    axes[1].set_title("Masked Attention Weights")
    axes[1].set_xlabel("Key"); axes[1].set_ylabel("Query")
    plt.tight_layout()
    plt.savefig(OUTPUT / "causal_mask.png", dpi=100); plt.close()
    print("  Saved causal_mask.png")


def demo_head_specialization():
    """Different attention heads can capture different token relationships."""
    print("\n=== Head Specialization ===")
    np.random.seed(42)
    seq, d_model, n_heads = 6, 8, 2
    X = np.random.randn(seq, d_model)
    # Two independent sets of projection weights -> different patterns
    head_attns = []
    for h in range(n_heads):
        np.random.seed(h * 10)
        Wq = np.random.randn(d_model, d_model) * 0.5
        Wk = np.random.randn(d_model, d_model) * 0.5
        Wv = np.random.randn(d_model, d_model) * 0.1
        Wo = np.random.randn(d_model, d_model) * 0.1
        _, attn = multi_head_attention(X, Wq, Wk, Wv, Wo, n_heads)
        head_attns.append(attn[h])  # (seq, seq)

    fig, axes = plt.subplots(1, n_heads, figsize=(8, 3))
    for h, ax in enumerate(axes):
        ax.imshow(head_attns[h], cmap="Blues")
        ax.set_title(f"Head {h} Attention")
        ax.set_xlabel("Key"); ax.set_ylabel("Query")
    plt.suptitle("Attention Head Specialization", y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT / "head_specialization.png", dpi=100); plt.close()
    print("  Saved head_specialization.png")
    # Measure diversity: correlation between heads
    corr = np.corrcoef(head_attns[0].ravel(), head_attns[1].ravel())[0, 1]
    print(f"  Correlation between head 0 and head 1 attention maps: {corr:.3f}")


if __name__ == "__main__":
    demo()
    demo_causal_mask()
    demo_head_specialization()
