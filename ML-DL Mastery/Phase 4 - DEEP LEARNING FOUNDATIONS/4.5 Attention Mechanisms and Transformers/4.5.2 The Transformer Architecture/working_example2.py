"""
Working Example 2: The Transformer Architecture — multi-head attention + FFN block
=====================================================================================
Manual implementation of transformer encoder block: MHA + LayerNorm + FFN.

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

def layer_norm(x, eps=1e-6):
    return (x - x.mean(axis=-1, keepdims=True)) / (x.std(axis=-1, keepdims=True) + eps)

def multi_head_attention(X, W_q, W_k, W_v, W_o, n_heads):
    """X: (seq, d_model)  Returns: (seq, d_model)"""
    seq, d_model = X.shape; d_k = d_model // n_heads
    Q = X @ W_q; K = X @ W_k; V = X @ W_v  # (seq, d_model) each

    # Split heads
    Q = Q.reshape(seq, n_heads, d_k); K = K.reshape(seq, n_heads, d_k); V = V.reshape(seq, n_heads, d_k)
    heads = []
    for h in range(n_heads):
        scores = Q[:, h, :] @ K[:, h, :].T / np.sqrt(d_k)
        w = softmax(scores); heads.append(w @ V[:, h, :])
    concat = np.concatenate(heads, axis=-1)  # (seq, d_model)
    return concat @ W_o

def ffn(x, W1, b1, W2, b2):
    return np.maximum(0, x @ W1 + b1) @ W2 + b2

def transformer_encoder_block(X, params):
    d = X.shape[1]
    # Self-attention + residual + norm
    attn_out = multi_head_attention(X, params["Wq"], params["Wk"], params["Wv"], params["Wo"], params["n_heads"])
    X = layer_norm(X + attn_out)
    # FFN + residual + norm
    ff_out = ffn(X, params["W1"], params["b1"], params["W2"], params["b2"])
    X = layer_norm(X + ff_out)
    return X

def demo():
    print("=== Transformer Encoder Block ===")
    rng = np.random.default_rng(42)
    seq_len, d_model, n_heads, d_ff = 6, 16, 2, 32
    d_k = d_model // n_heads

    params = {
        "Wq": rng.standard_normal((d_model, d_model)) * 0.1,
        "Wk": rng.standard_normal((d_model, d_model)) * 0.1,
        "Wv": rng.standard_normal((d_model, d_model)) * 0.1,
        "Wo": rng.standard_normal((d_model, d_model)) * 0.1,
        "W1": rng.standard_normal((d_model, d_ff)) * 0.1,
        "b1": np.zeros(d_ff),
        "W2": rng.standard_normal((d_ff, d_model)) * 0.1,
        "b2": np.zeros(d_model),
        "n_heads": n_heads,
    }
    X = rng.standard_normal((seq_len, d_model))
    out = transformer_encoder_block(X, params)
    print(f"  Input:  {X.shape}")
    print(f"  Output: {out.shape}  (same — transformer preserves sequence shape)")
    print(f"  LayerNorm mean: {out.mean(axis=-1).round(4)}  (~=0 after norm)")
    print(f"  LayerNorm std:  {out.std(axis=-1).round(4)}  (~=1 after norm)")

def demo_positional_encoding():
    """Sinusoidal positional encoding as described in 'Attention Is All You Need'."""
    print("\n=== Positional Encoding ===")
    def get_positional_encoding(seq_len, d_model):
        PE = np.zeros((seq_len, d_model))
        pos = np.arange(seq_len)[:, None]
        div = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        PE[:, 0::2] = np.sin(pos * div)
        PE[:, 1::2] = np.cos(pos * div[:d_model//2])
        return PE

    seq_len, d_model = 8, 16
    PE = get_positional_encoding(seq_len, d_model)
    print(f"  PE shape: {PE.shape}  (seq_len={seq_len}, d_model={d_model})")
    # Adjacent positions should be similar
    for i in range(1, 4):
        sim = (PE[0] @ PE[i]) / (np.linalg.norm(PE[0]) * np.linalg.norm(PE[i]))
        print(f"  cosine(pos 0, pos {i}): {sim:.4f}")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(PE.T, aspect="auto", cmap="RdBu")
    ax.set_xlabel("Position"); ax.set_ylabel("Embedding dim")
    ax.set_title("Sinusoidal Positional Encoding")
    plt.tight_layout(); plt.savefig(OUTPUT / "positional_encoding.png"); plt.close()
    print("  Saved positional_encoding.png")


def demo_stacked_blocks():
    """Pass data through 4 stacked transformer encoder blocks."""
    print("\n=== Stacked Transformer Blocks ===")
    rng = np.random.default_rng(99)
    seq_len, d_model, n_heads, d_ff, n_layers = 6, 16, 2, 32, 4

    def make_params():
        return {
            "Wq": rng.standard_normal((d_model, d_model)) * 0.1,
            "Wk": rng.standard_normal((d_model, d_model)) * 0.1,
            "Wv": rng.standard_normal((d_model, d_model)) * 0.1,
            "Wo": rng.standard_normal((d_model, d_model)) * 0.1,
            "W1": rng.standard_normal((d_model, d_ff)) * 0.1,
            "b1": np.zeros(d_ff),
            "W2": rng.standard_normal((d_ff, d_model)) * 0.1,
            "b2": np.zeros(d_model),
            "n_heads": n_heads,
        }

    X = rng.standard_normal((seq_len, d_model))
    print(f"  Input  std: {X.std():.4f}")
    for i in range(n_layers):
        X = transformer_encoder_block(X, make_params())
        print(f"  After block {i+1}: std={X.std():.4f}  mean={X.mean():.4f}")


if __name__ == "__main__":
    demo()
    demo_positional_encoding()
    demo_stacked_blocks()
