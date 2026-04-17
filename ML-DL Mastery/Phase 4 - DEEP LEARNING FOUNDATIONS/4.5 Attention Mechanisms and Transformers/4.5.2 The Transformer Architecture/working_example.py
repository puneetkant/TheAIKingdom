"""
Working Example: The Transformer Architecture
Covers positional encoding, encoder block, decoder block, full Transformer,
and the original "Attention Is All You Need" architecture.
"""
import numpy as np
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_transformer")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def softmax(z, axis=-1):
    e = np.exp(z - z.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)

def layer_norm(x, eps=1e-6):
    m = x.mean(axis=-1, keepdims=True)
    s = x.std(axis=-1, keepdims=True) + eps
    return (x - m) / s

def relu(z): return np.maximum(0, z)


# ── 1. Positional Encoding ───────────────────────────────────────────────────
def positional_encoding(T, d_model):
    """
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    PE   = np.zeros((T, d_model))
    pos  = np.arange(T).reshape(-1, 1)
    dims = np.arange(0, d_model, 2)
    div  = np.power(10000, dims / d_model)
    PE[:, 0::2] = np.sin(pos / div)
    PE[:, 1::2] = np.cos(pos / div[:d_model//2])
    return PE


def pe_demo():
    print("=== Positional Encoding ===")
    print("  Transformers have no recurrence — position info must be injected")
    print("  PE(pos, 2i)   = sin(pos / 10000^(2i/d))")
    print("  PE(pos, 2i+1) = cos(pos / 10000^(2i/d))")
    print()
    print("  Properties:")
    print("    Unique encoding for each position")
    print("    PE · PE^T ≈ function of (pos1 - pos2) → relative position info")
    print("    Works for unseen sequence lengths")

    T, d = 20, 64
    PE   = positional_encoding(T, d)
    print(f"\n  PE shape: {PE.shape}  (T={T}, d={d})")
    print(f"  First 5 positions, first 6 dims:")
    print(np.round(PE[:5, :6], 3))

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(PE, cmap='RdBu', aspect='auto')
    plt.colorbar(im, ax=ax)
    ax.set(xlabel="Embedding dimension", ylabel="Position",
           title="Positional Encoding Matrix")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "positional_encoding.png")
    plt.savefig(path, dpi=90); plt.close()
    print(f"  Plot saved: {path}")


# ── 2. Scaled Dot-Product Attention ──────────────────────────────────────────
def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k    = Q.shape[-1]
    scores = Q @ K.swapaxes(-2, -1) / np.sqrt(d_k)
    if mask is not None:
        scores = scores + mask * -1e9
    alpha  = softmax(scores, axis=-1)
    return alpha @ V, alpha


# ── 3. Multi-Head Attention layer ────────────────────────────────────────────
class MultiHeadAttention:
    def __init__(self, d_model, n_heads, rng):
        self.h  = n_heads
        self.dk = d_model // n_heads
        self.W_Q = rng.standard_normal((d_model, d_model)) * 0.02
        self.W_K = rng.standard_normal((d_model, d_model)) * 0.02
        self.W_V = rng.standard_normal((d_model, d_model)) * 0.02
        self.W_O = rng.standard_normal((d_model, d_model)) * 0.02
        self.d   = d_model

    def _split_heads(self, X):
        """(T, d_model) → (n_heads, T, d_k)."""
        T  = X.shape[0]
        X  = X.reshape(T, self.h, self.dk)
        return X.transpose(1, 0, 2)   # (h, T, dk)

    def forward(self, Q_in, K_in, V_in, mask=None):
        Q = self._split_heads(Q_in @ self.W_Q)   # (h, T, dk)
        K = self._split_heads(K_in @ self.W_K)
        V = self._split_heads(V_in @ self.W_V)
        # Apply attention per head
        heads = []
        for i in range(self.h):
            h_out, _ = scaled_dot_product_attention(Q[i], K[i], V[i], mask)
            heads.append(h_out)
        concat = np.concatenate(heads, axis=-1)   # (T, d_model)
        return concat @ self.W_O


# ── 4. Feed-Forward sublayer ─────────────────────────────────────────────────
class FeedForward:
    def __init__(self, d_model, d_ff, rng):
        self.W1 = rng.standard_normal((d_model, d_ff)) * 0.02
        self.b1 = np.zeros(d_ff)
        self.W2 = rng.standard_normal((d_ff, d_model)) * 0.02
        self.b2 = np.zeros(d_model)

    def forward(self, X):
        return relu(X @ self.W1 + self.b1) @ self.W2 + self.b2


# ── 5. Encoder Block ─────────────────────────────────────────────────────────
class EncoderBlock:
    """Self-Attn → Add&Norm → FFN → Add&Norm."""
    def __init__(self, d_model, n_heads, d_ff, rng):
        self.mha = MultiHeadAttention(d_model, n_heads, rng)
        self.ffn = FeedForward(d_model, d_ff, rng)

    def forward(self, X, mask=None):
        attn_out = self.mha.forward(X, X, X, mask)
        X2 = layer_norm(X + attn_out)
        ff_out = self.ffn.forward(X2)
        X3 = layer_norm(X2 + ff_out)
        return X3


# ── 6. Decoder Block ─────────────────────────────────────────────────────────
class DecoderBlock:
    """Masked Self-Attn → Cross-Attn → FFN (each with Add&Norm)."""
    def __init__(self, d_model, n_heads, d_ff, rng):
        self.self_attn  = MultiHeadAttention(d_model, n_heads, rng)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, rng)
        self.ffn        = FeedForward(d_model, d_ff, rng)

    def forward(self, Y, enc_out, src_mask=None, tgt_mask=None):
        attn1 = self.self_attn.forward(Y, Y, Y, tgt_mask)
        Y2    = layer_norm(Y + attn1)
        attn2 = self.cross_attn.forward(Y2, enc_out, enc_out, src_mask)
        Y3    = layer_norm(Y2 + attn2)
        ff    = self.ffn.forward(Y3)
        return layer_norm(Y3 + ff)


# ── 7. Full Transformer ───────────────────────────────────────────────────────
def transformer_demo():
    print("\n=== Full Transformer (Encoder-Decoder) ===")
    rng     = np.random.default_rng(42)
    d_model = 32
    n_heads = 4
    d_ff    = 64
    n_enc   = 2   # encoder layers
    n_dec   = 2   # decoder layers
    T_src   = 6
    T_tgt   = 4
    V       = 20  # vocab size

    # Token embeddings
    Emb = rng.standard_normal((V, d_model)) * 0.02
    PE  = positional_encoding(max(T_src, T_tgt), d_model)

    src_ids = rng.integers(0, V, T_src)
    tgt_ids = rng.integers(0, V, T_tgt)

    # Embedding + PE
    enc_in = Emb[src_ids] + PE[:T_src]
    dec_in = Emb[tgt_ids] + PE[:T_tgt]

    # Causal mask for decoder self-attention
    causal_mask = np.triu(np.ones((T_tgt, T_tgt)), k=1)

    # Encoder stack
    X = enc_in
    encoders = [EncoderBlock(d_model, n_heads, d_ff, rng) for _ in range(n_enc)]
    for enc in encoders:
        X = enc.forward(X)
    enc_out = X
    print(f"  Encoder output: {enc_out.shape}")

    # Decoder stack
    Y = dec_in
    decoders = [DecoderBlock(d_model, n_heads, d_ff, rng) for _ in range(n_dec)]
    for dec in decoders:
        Y = dec.forward(Y, enc_out, tgt_mask=causal_mask)
    print(f"  Decoder output: {Y.shape}")

    # Final projection to vocab
    W_proj = rng.standard_normal((d_model, V)) * 0.02
    logits = Y @ W_proj   # (T_tgt, V)
    preds  = logits.argmax(axis=-1)
    print(f"  Logits:    {logits.shape}")
    print(f"  Predicted token ids: {preds}")


# ── 8. Architecture overview ─────────────────────────────────────────────────
def architecture_overview():
    print("\n=== Transformer Architecture ('Attention Is All You Need', Vaswani 2017) ===")
    print()
    print("  ENCODER (left)              DECODER (right)")
    print("  ─────────────────────────   ─────────────────────────────")
    print("  Input Embedding + PE        Output Embedding + PE")
    print("  ↓                           ↓")
    print("  × N:                        × N:")
    print("    Multi-Head Self-Attn        Masked MH Self-Attn")
    print("    Add & LayerNorm             Add & LayerNorm")
    print("    Feed-Forward (d_ff=2048)    MH Cross-Attn (Q=dec, K=V=enc)")
    print("    Add & LayerNorm             Add & LayerNorm")
    print("                               Feed-Forward (d_ff=2048)")
    print("                               Add & LayerNorm")
    print("                           ↓")
    print("                           Linear → Softmax → Output probs")
    print()
    print("  Original 'base' config:")
    config = [
        ("d_model",    512,   "embedding dimension"),
        ("n_heads",    8,     "attention heads"),
        ("d_k = d_v",  64,    "d_model / n_heads"),
        ("d_ff",       2048,  "FF inner dimension"),
        ("N (layers)", 6,     "encoder and decoder layers"),
        ("Dropout",    0.1,   "applied to attention weights and sublayers"),
        ("Parameters", "65M", "base model"),
    ]
    print(f"  {'Param':<14} {'Value':<10} {'Notes'}")
    for name, val, note in config:
        print(f"  {name:<14} {str(val):<10} {note}")


# ── 9. Complexity comparison ─────────────────────────────────────────────────
def complexity_comparison():
    print("\n=== Complexity: Self-Attention vs RNN vs CNN ===")
    print(f"  {'Operation':<22} {'Complexity per layer':<30} {'Sequential ops'}")
    rows = [
        ("Self-Attention",   "O(T²·d)",                    "O(1)   — fully parallel"),
        ("RNN",              "O(T·d²)",                    "O(T)   — sequential"),
        ("Dilated CNN",      "O(T·k·d) per layer",         "O(1)   — parallel"),
        ("Self-Attn (local)","O(T·r·d)  r=window",         "O(1)   — parallel"),
    ]
    for name, comp, seq in rows:
        print(f"  {name:<22} {comp:<30} {seq}")
    print()
    print("  Self-attention is O(T²) — quadratic in sequence length!")
    print("  Efficient Attention variants (Linformer, Performer) reduce to O(T·log T)")


if __name__ == "__main__":
    pe_demo()
    transformer_demo()
    architecture_overview()
    complexity_comparison()
