"""
Working Example 2: Attention Fundamentals — scaled dot-product attention from scratch
=======================================================================================
Manual Q, K, V attention, softmax scores, context vector computation.

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
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x); return e / e.sum(axis=axis, keepdims=True)

def scaled_dot_product_attention(Q, K, V, mask=None):
    """Q: (seq_q, d_k)  K: (seq_k, d_k)  V: (seq_k, d_v)"""
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)          # (seq_q, seq_k)
    if mask is not None:
        scores = np.where(mask, -1e9, scores)
    weights = softmax(scores, axis=-1)        # (seq_q, seq_k)
    context = weights @ V                     # (seq_q, d_v)
    return context, weights

def demo_attention():
    print("=== Scaled Dot-Product Attention ===")
    rng = np.random.default_rng(42)
    seq_len, d_k, d_v = 5, 8, 8

    Q = rng.standard_normal((seq_len, d_k))
    K = rng.standard_normal((seq_len, d_k))
    V = rng.standard_normal((seq_len, d_v))

    context, weights = scaled_dot_product_attention(Q, K, V)
    print(f"  Q shape: {Q.shape}  K shape: {K.shape}  V shape: {V.shape}")
    print(f"  Attention weights shape: {weights.shape}")
    print(f"  Context vector shape: {context.shape}")
    print(f"  Weight row sum: {weights.sum(axis=1).round(4)}  (should all be 1.0)")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    im = axes[0].imshow(weights, cmap="Blues", vmin=0, vmax=1)
    axes[0].set_title("Attention Weights"); axes[0].set_xlabel("Key pos"); axes[0].set_ylabel("Query pos")
    plt.colorbar(im, ax=axes[0])
    axes[1].imshow(context, cmap="RdBu", aspect="auto")
    axes[1].set_title("Context Vectors"); axes[1].set_xlabel("d_v dim"); axes[1].set_ylabel("Query pos")
    plt.tight_layout(); plt.savefig(OUTPUT / "attention.png"); plt.close()
    print("  Saved attention.png")

def demo_causal_mask():
    print("\n=== Causal (autoregressive) Mask ===")
    seq_len = 5
    mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
    rng = np.random.default_rng(0)
    Q = rng.standard_normal((seq_len, 4)); K = Q.copy(); V = rng.standard_normal((seq_len, 4))
    _, w = scaled_dot_product_attention(Q, K, V, mask=mask)
    print("  Causal attention weights (lower triangle only):")
    print(np.round(w, 3))

def demo_multi_head_attention():
    """Show multi-head attention as parallel attention over sub-spaces."""
    print("\n=== Multi-Head Attention ===")
    rng = np.random.default_rng(1)
    seq_len, d_model, n_heads = 6, 16, 4
    d_k = d_model // n_heads

    X = rng.standard_normal((seq_len, d_model))
    all_weights = []
    for h in range(n_heads):
        Wq = rng.standard_normal((d_model, d_k)) * 0.1
        Wk = rng.standard_normal((d_model, d_k)) * 0.1
        Wv = rng.standard_normal((d_model, d_k)) * 0.1
        Q = X @ Wq; K = X @ Wk; V = X @ Wv
        _, weights = scaled_dot_product_attention(Q, K, V)
        all_weights.append(weights)
        print(f"  Head {h+1}: max attention weight = {weights.max():.4f}  "
              f"entropy = {-(weights * np.log(weights + 1e-9)).sum(axis=-1).mean():.4f}")

    # Different heads attend to different patterns
    print(f"  Pairwise correlation of head weights: "
          f"{np.corrcoef([w.ravel() for w in all_weights]).mean():.4f} (lower = more diverse)")


def demo_attention_score_temperature():
    """Show effect of temperature (sqrt(d_k) scaling) on attention sharpness."""
    print("\n=== Temperature Effect on Attention ===")
    rng = np.random.default_rng(42)
    Q = rng.standard_normal((4, 16)); K = rng.standard_normal((4, 16)); V = rng.standard_normal((4, 8))
    raw_scores = Q @ K.T
    for temp in [1, 4, 8, 16]:
        scaled = raw_scores / temp
        w = softmax(scaled)
        entropy = -(w * np.log(w + 1e-9)).sum(axis=-1).mean()
        print(f"  temp=sqrt({temp:2d}^2)={temp:2d}: weight entropy={entropy:.4f}  "
              f"({'sharp' if entropy < 1.0 else 'diffuse'} attention)")


if __name__ == "__main__":
    demo_attention()
    demo_causal_mask()
    demo_multi_head_attention()
    demo_attention_score_temperature()
