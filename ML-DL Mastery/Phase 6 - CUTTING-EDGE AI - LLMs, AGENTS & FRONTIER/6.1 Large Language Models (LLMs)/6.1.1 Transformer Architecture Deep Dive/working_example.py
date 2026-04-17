"""
Working Example: Transformer Architecture Deep Dive
Covers multi-head attention, positional encoding, layer norm,
KV cache, and architectural variants in modern LLMs.
"""
import numpy as np
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_transformer")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)

def layer_norm(x, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps)


# ── 1. Scaled dot-product attention ──────────────────────────────────────────
def scaled_dot_product_attention():
    print("=== Scaled Dot-Product Attention ===")
    print()
    print("  Attention(Q, K, V) = softmax(QK^T / √d_k) V")
    print()
    rng = np.random.default_rng(0)
    T = 5; d_k = 8; d_v = 8
    Q = rng.normal(0, 1, (T, d_k))
    K = rng.normal(0, 1, (T, d_k))
    V = rng.normal(0, 1, (T, d_v))

    scores = Q @ K.T / np.sqrt(d_k)        # (T, T)
    # Causal mask (for decoder self-attention)
    mask = np.triu(np.full((T, T), -1e9), k=1)
    scores += mask
    weights = softmax(scores)               # (T, T)
    out = weights @ V                       # (T, d_v)

    print(f"  Q, K, V shapes: ({T},{d_k})")
    print(f"  Attention weights (row = query, col = key):")
    for row in weights:
        print("  ", " ".join(f"{w:.3f}" for w in row))
    print(f"  Output shape: {out.shape}")
    print()
    print("  Why scale by √d_k?")
    print("    Dot products grow in magnitude with d_k → softmax saturates")
    print("    Scaling keeps gradients healthy; originally from 'Attention is All You Need'")
    return weights


# ── 2. Multi-head attention ───────────────────────────────────────────────────
def multi_head_attention():
    print("\n=== Multi-Head Attention ===")
    print()
    print("  MHA = Concat(head_1, ..., head_h) W_O")
    print("  head_i = Attention(Q W_Qi, K W_Ki, V W_Vi)")
    print()
    rng = np.random.default_rng(0)
    n_heads = 4; d_model = 32; T = 6
    d_k = d_model // n_heads   # 8

    X = rng.normal(0, 1, (T, d_model))
    W_Q = rng.normal(0, 0.1, (n_heads, d_model, d_k))
    W_K = rng.normal(0, 0.1, (n_heads, d_model, d_k))
    W_V = rng.normal(0, 0.1, (n_heads, d_model, d_k))
    W_O = rng.normal(0, 0.1, (n_heads * d_k, d_model))

    heads = []
    for h in range(n_heads):
        Q = X @ W_Q[h]   # (T, d_k)
        K = X @ W_K[h]
        V = X @ W_V[h]
        scores = softmax(Q @ K.T / np.sqrt(d_k))
        heads.append(scores @ V)   # (T, d_k)

    concat = np.concatenate(heads, axis=-1)   # (T, d_model)
    out = concat @ W_O                         # (T, d_model)
    print(f"  Input: ({T}, {d_model}), heads={n_heads}, d_k={d_k}")
    print(f"  Each head output: ({T}, {d_k})")
    print(f"  Concatenated: ({T}, {d_model})")
    print(f"  MHA output: {out.shape}")
    print()
    print("  Different attention heads learn different patterns:")
    print("    Head 1: syntactic dependencies")
    print("    Head 2: co-reference")
    print("    Head 3: next-token prediction")
    print("    Head 4: positional relationships")


# ── 3. Positional encoding ────────────────────────────────────────────────────
def positional_encoding():
    print("\n=== Positional Encoding ===")
    print()
    print("  Sinusoidal (original Transformer):")
    print("    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))")
    print("    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))")
    print()
    d_model = 16; T = 10
    PE = np.zeros((T, d_model))
    pos = np.arange(T)[:, None]
    i   = np.arange(d_model // 2)[None, :]
    div = np.power(10000, 2*i / d_model)
    PE[:, 0::2] = np.sin(pos / div)
    PE[:, 1::2] = np.cos(pos / div)
    print(f"  PE matrix shape: {PE.shape}")
    print(f"  PE[0,:4] = {PE[0,:4].round(4)}")
    print(f"  PE[9,:4] = {PE[9,:4].round(4)}")
    print()
    print("  Positional encoding variants:")
    variants = [
        ("Sinusoidal",  "Fixed; extrapolates poorly"),
        ("Learned abs.","Trainable; BERT, GPT-2; no extrapolation"),
        ("RoPE",        "Rotary; LLaMA, GPT-NeoX; relative by construction; ∞ context"),
        ("ALiBi",       "Add linear bias; MPT, BLOOM; no positional params"),
        ("xPos",        "Extended RoPE; better long context; length generalisation"),
    ]
    for v, d in variants:
        print(f"  {v:<14} {d}")


# ── 4. LLM architecture variants ─────────────────────────────────────────────
def architecture_variants():
    print("\n=== LLM Architecture Variants ===")
    print()
    print("  Normalisation placement:")
    print("    Post-LN (original): X → MHA → Add+Norm → FFN → Add+Norm")
    print("    Pre-LN (GPT-3+):    X → Norm → MHA → Add → Norm → FFN → Add")
    print("    RMSNorm: variance-only norm; LLaMA; cheaper; often better")
    print()
    print("  FFN variants:")
    ffn_vars = [
        ("Standard FFN",  "Linear(4d) → GELU → Linear(d); GPT-2"),
        ("SwiGLU",        "Linear(8d/3) × σ(Linear(8d/3)) → Linear(d); LLaMA"),
        ("GeGLU",         "GELU gating; PaLM; similar to SwiGLU"),
        ("Mixture of Exp.","Sparse FFN; only top-k experts active; Mixtral"),
    ]
    for f, d in ffn_vars:
        print(f"  {f:<18} {d}")
    print()
    print("  Attention variants:")
    attn_vars = [
        ("MHA",  "Multi-head attention; all heads unique Q,K,V"),
        ("MQA",  "Multi-query; 1 K,V head shared; faster; PaLM"),
        ("GQA",  "Grouped-query; G shared K,V per G query heads; LLaMA-2/3"),
        ("FlashAttn","Fused kernel; IO-aware; 2-4× faster; same output"),
        ("Sliding window","Local attention per layer; Mistral; long context"),
    ]
    for a, d in attn_vars:
        print(f"  {a:<14} {d}")
    print()
    # KV cache demo
    print("  KV Cache: store K, V from previous tokens; avoid recomputation")
    print("  Memory = 2 × n_layers × n_heads × d_head × seq_len × bytes")
    for seq_len in [1024, 8192, 128_000]:
        n_layers=32; n_kv_heads=8; d_head=128; bytes_=2  # GQA BF16
        mem_gb = 2 * n_layers * n_kv_heads * d_head * seq_len * bytes_ / 1e9
        print(f"    seq={seq_len:>7}: KV cache = {mem_gb:.3f} GB")


if __name__ == "__main__":
    scaled_dot_product_attention()
    multi_head_attention()
    positional_encoding()
    architecture_variants()
