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

if __name__ == "__main__":
    demo()
