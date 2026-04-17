"""
Working Example: Attention Fundamentals
Covers additive (Bahdanau) and multiplicative (Luong) attention,
self-attention, multi-head attention, and attention visualisation.
"""
import numpy as np
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_attention")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def softmax(z, axis=-1):
    e = np.exp(z - z.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


# ── 1. Motivation ─────────────────────────────────────────────────────────────
def motivation():
    print("=== Why Attention? ===")
    print("  Seq2Seq bottleneck: all source info compressed into single context c")
    print("  Long sequences lose detail; decoder has no direct path to early tokens")
    print()
    print("  Attention: let decoder look at ALL encoder hidden states")
    print("  Weighted sum of encoder states; weights learned from (query, key) similarity")
    print()
    print("  Key intuition:")
    print("    When translating 'cat', attend strongly to source token 'chat'")
    print("    When translating 'eat', attend strongly to 'mange'")
    print("    The model learns WHICH source tokens are relevant for each target token")


# ── 2. Bahdanau additive attention ───────────────────────────────────────────
def bahdanau_attention(query, keys, W_q, W_k, v):
    """
    Additive (Bahdanau) attention:
    score(q, k_i) = v^T · tanh(W_q·q + W_k·k_i)
    alpha = softmax(scores)
    context = Σ alpha_i · k_i
    """
    T_src, _ = keys.shape
    scores = []
    for k in keys:
        z = np.tanh(query @ W_q + k @ W_k)
        scores.append(z @ v)
    scores  = np.array(scores)          # (T_src,)
    alpha   = softmax(scores, axis=0)   # attention weights
    context = alpha @ keys               # (d_keys,)
    return context, alpha, scores


def bahdanau_demo():
    print("\n=== Bahdanau (Additive) Attention ===")
    print("  score(q, k) = v^T · tanh(W_q·q + W_k·k)")
    print("  Context = Σ softmax(scores) · values")

    rng = np.random.default_rng(1)
    T_src = 5; d_enc = 6; d_dec = 4; d_attn = 8

    encoder_states = rng.standard_normal((T_src, d_enc))
    decoder_query  = rng.standard_normal(d_dec)
    W_q = rng.standard_normal((d_dec, d_attn)) * 0.1
    W_k = rng.standard_normal((d_enc, d_attn)) * 0.1
    v   = rng.standard_normal(d_attn) * 0.1

    context, alpha, scores = bahdanau_attention(decoder_query, encoder_states, W_q, W_k, v)
    print(f"\n  Encoder states: {encoder_states.shape}")
    print(f"  Decoder query:  {decoder_query.shape}")
    print(f"  Attention weights: {alpha.round(4)}")
    print(f"  (Sum = {alpha.sum():.4f})")
    print(f"  Most attended token: {alpha.argmax()}")
    print(f"  Context vector shape: {context.shape}")


# ── 3. Luong multiplicative attention ─────────────────────────────────────────
def luong_attention(query, keys, W_score=None, mode="dot"):
    """
    Luong attention:
    dot:      score = q^T · k
    general:  score = q^T · W · k
    concat:   (Bahdanau style — not shown here)
    """
    if mode == "dot":
        scores = keys @ query           # (T_src,)
    else:  # general
        scores = keys @ W_score @ query
    alpha   = softmax(scores, axis=0)
    context = alpha @ keys
    return context, alpha


def luong_demo():
    print("\n=== Luong (Multiplicative / Dot-Product) Attention ===")
    print("  score(q, k) = q^T · k           (dot)")
    print("  score(q, k) = q^T · W · k       (general)")
    print("  Scaled: divide by √d_k to prevent softmax saturation")

    rng = np.random.default_rng(2)
    T_src = 6; d = 8
    keys  = rng.standard_normal((T_src, d))
    query = rng.standard_normal(d)

    ctx_dot, alpha_dot = luong_attention(query, keys, mode="dot")
    print(f"\n  Dot attention weights: {alpha_dot.round(4)}")

    # Scaled dot-product
    scores_scaled = (keys @ query) / np.sqrt(d)
    alpha_scaled  = softmax(scores_scaled)
    print(f"  Scaled dot weights:    {alpha_scaled.round(4)}")
    print(f"\n  Scaling by √d={d**0.5:.2f} prevents softmax from becoming one-hot")


# ── 4. Self-attention ─────────────────────────────────────────────────────────
def self_attention(X, W_Q, W_K, W_V, mask=None):
    """
    Scaled dot-product self-attention.
    Q = X·W_Q, K = X·W_K, V = X·W_V
    Attention = softmax(Q·K^T / √d_k) · V
    """
    Q = X @ W_Q   # (T, d_k)
    K = X @ W_K   # (T, d_k)
    V = X @ W_V   # (T, d_v)
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)   # (T, T)
    if mask is not None:
        scores = scores + mask * -1e9  # causal mask
    alpha = softmax(scores, axis=-1)   # (T, T)
    out   = alpha @ V                  # (T, d_v)
    return out, alpha


def self_attention_demo():
    print("\n=== Self-Attention ===")
    print("  Every token attends to every other token in the same sequence")
    print("  Q = X·W_Q   K = X·W_K   V = X·W_V")
    print("  Attn = softmax(QK^T / √d_k) · V")

    rng  = np.random.default_rng(3)
    T, d = 7, 16
    d_k  = d_v = 8
    X    = rng.standard_normal((T, d))
    W_Q  = rng.standard_normal((d, d_k)) * 0.1
    W_K  = rng.standard_normal((d, d_k)) * 0.1
    W_V  = rng.standard_normal((d, d_v)) * 0.1

    out, alpha = self_attention(X, W_Q, W_K, W_V)
    print(f"\n  Input X:  {X.shape}")
    print(f"  Output:   {out.shape}")
    print(f"  Attention map (T×T): {alpha.shape}")
    print(f"\n  Attention matrix (row = query, col = key):")
    print(np.round(alpha, 3))


# ── 5. Causal (masked) self-attention ─────────────────────────────────────────
def causal_attention_demo():
    print("\n=== Causal (Masked) Self-Attention ===")
    print("  Used in autoregressive models (GPT): token t can only attend to t'≤t")
    print("  Mask future positions with -∞ before softmax")

    rng  = np.random.default_rng(4)
    T, d = 5, 8
    d_k  = 4
    X    = rng.standard_normal((T, d))
    W_Q  = rng.standard_normal((d, d_k)) * 0.1
    W_K  = rng.standard_normal((d, d_k)) * 0.1
    W_V  = rng.standard_normal((d, d_k)) * 0.1

    causal_mask = np.triu(np.ones((T, T)), k=1)  # 1 = future positions to mask
    out, alpha  = self_attention(X, W_Q, W_K, W_V, mask=causal_mask)

    print(f"\n  Causal mask (upper triangle = masked out):")
    print(causal_mask.astype(int))
    print(f"\n  Resulting attention weights (lower triangular = causal):")
    print(np.round(alpha, 3))


# ── 6. Multi-head attention ───────────────────────────────────────────────────
def multi_head_attention(X, W_Qs, W_Ks, W_Vs, W_O):
    """Multi-head attention: parallel heads, then concat → project."""
    heads = []
    for W_Q, W_K, W_V in zip(W_Qs, W_Ks, W_Vs):
        h, _ = self_attention(X, W_Q, W_K, W_V)
        heads.append(h)
    concat = np.concatenate(heads, axis=-1)    # (T, n_heads × d_v)
    out    = concat @ W_O                       # (T, d_model)
    return out, heads

def multi_head_demo():
    print("\n=== Multi-Head Attention ===")
    print("  Multiple attention heads allow attending to different aspects simultaneously")
    print("  head_i = Attention(X·W_Qi, X·W_Ki, X·W_Vi)")
    print("  MultiHead = Concat(head_1,...,head_h) · W_O")
    print()
    print("  Different heads might learn:")
    print("    Head 1: syntactic relationships (subject-verb)")
    print("    Head 2: co-reference (pronoun → noun)")
    print("    Head 3: semantic similarity")

    rng = np.random.default_rng(5)
    T, d_model = 6, 16
    n_heads    = 4
    d_k        = d_model // n_heads   # 4
    X          = rng.standard_normal((T, d_model))

    W_Qs = [rng.standard_normal((d_model, d_k)) * 0.1 for _ in range(n_heads)]
    W_Ks = [rng.standard_normal((d_model, d_k)) * 0.1 for _ in range(n_heads)]
    W_Vs = [rng.standard_normal((d_model, d_k)) * 0.1 for _ in range(n_heads)]
    W_O  = rng.standard_normal((n_heads * d_k, d_model)) * 0.1

    out, heads = multi_head_attention(X, W_Qs, W_Ks, W_Vs, W_O)
    print(f"\n  Input:  {X.shape}  (T={T}, d_model={d_model})")
    print(f"  n_heads={n_heads}, d_k={d_k}")
    print(f"  Each head output: {heads[0].shape}")
    print(f"  Concat: ({T}, {n_heads*d_k})")
    print(f"  Final output: {out.shape}")
    print(f"\n  Parameters per head: {d_model*d_k} (Q) + {d_model*d_k} (K) + {d_model*d_k} (V)")
    total = n_heads * 3 * d_model * d_k + n_heads * d_k * d_model
    print(f"  Total MHA params: {total}")


# ── 7. Visualise attention map ────────────────────────────────────────────────
def visualise_attention():
    rng    = np.random.default_rng(6)
    tokens = ["The", "cat", "sat", "on", "the", "mat"]
    T, d   = len(tokens), 16
    d_k    = 8
    X      = rng.standard_normal((T, d))
    W_Q    = rng.standard_normal((d, d_k)) * 0.5
    W_K    = rng.standard_normal((d, d_k)) * 0.5
    W_V    = rng.standard_normal((d, d_k)) * 0.1
    _, alpha = self_attention(X, W_Q, W_K, W_V)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(alpha, cmap="Blues", vmin=0, vmax=alpha.max())
    ax.set_xticks(range(T)); ax.set_xticklabels(tokens, rotation=45, ha="right")
    ax.set_yticks(range(T)); ax.set_yticklabels(tokens)
    plt.colorbar(im, ax=ax)
    ax.set_title("Self-Attention Map"); ax.set_xlabel("Key"); ax.set_ylabel("Query")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "attention_map.png")
    plt.savefig(path, dpi=90); plt.close()
    print(f"\n  Attention map saved: {path}")


if __name__ == "__main__":
    motivation()
    bahdanau_demo()
    luong_demo()
    self_attention_demo()
    causal_attention_demo()
    multi_head_demo()
    visualise_attention()
