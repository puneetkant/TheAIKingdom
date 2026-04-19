"""
Working Example 2: Transformer Variants — BERT vs GPT vs T5 architecture comparison
======================================================================================
Demonstrates the encoder-only, decoder-only, encoder-decoder paradigms.

Run:  python working_example2.py
"""
from pathlib import Path
try:
    import numpy as np
except ImportError:
    raise SystemExit("pip install numpy")

def demo():
    print("=== Transformer Variants Comparison ===\n")
    variants = [
        {
            "name": "BERT (Encoder-only)",
            "masking": "Bidirectional (all positions see all)",
            "pretraining": "Masked Language Modeling + NSP",
            "use_case": "Classification, NER, QA (understanding)",
            "key_models": "BERT, RoBERTa, DistilBERT, ALBERT, DeBERTa",
        },
        {
            "name": "GPT (Decoder-only)",
            "masking": "Causal (each token sees only past)",
            "pretraining": "Next token prediction (CLM)",
            "use_case": "Text generation, chat, code (generation)",
            "key_models": "GPT-2, GPT-3/4, LLaMA, Mistral, Falcon",
        },
        {
            "name": "T5 / BART (Encoder-Decoder)",
            "masking": "Encoder: bidirectional; Decoder: causal",
            "pretraining": "Span masking / Denoising",
            "use_case": "Translation, summarization, seq2seq",
            "key_models": "T5, BART, Pegasus, mT5",
        },
    ]
    for v in variants:
        print(f"  {'='*55}")
        for k, val in v.items():
            print(f"  {k:15s}: {val}")
    print()

    # Causal mask vs no mask (quick numpy demo)
    print("=== Attention mask demo ===")
    n = 5
    # Causal (GPT)
    causal_mask = np.tril(np.ones((n, n)))
    # Bidirectional (BERT)
    bidi_mask = np.ones((n, n))
    print("  Causal mask (GPT/decoder):\n", causal_mask.astype(int))
    print("  Bidirectional (BERT/encoder):\n", bidi_mask.astype(int))

def demo_encoder_only_pattern():
    """Encoder-only (BERT-style): bidirectional self-attention block."""
    print("=== Encoder-Only (BERT-style) Transformer Block ===")
    import numpy as np
    np.random.seed(42)
    T, D, H = 8, 32, 4  # seq_len, d_model, heads
    dh = D // H
    x = np.random.randn(T, D)
    # Multi-head self-attention (full, no mask)
    W_qkv = np.random.randn(D, 3*D) * 0.1
    qkv = x @ W_qkv
    Q, K, V = qkv[:,:D], qkv[:,D:2*D], qkv[:,2*D:]
    # Reshape to heads
    Q = Q.reshape(T, H, dh).transpose(1,0,2)  # (H,T,dh)
    K = K.reshape(T, H, dh).transpose(1,0,2)
    V = V.reshape(T, H, dh).transpose(1,0,2)
    scores = Q @ K.transpose(0,2,1) / (dh**0.5)  # (H,T,T) - NO mask
    attn = np.exp(scores - scores.max(-1, keepdims=True))
    attn /= attn.sum(-1, keepdims=True)
    out = (attn @ V).transpose(1,0,2).reshape(T, D)
    # FFN
    W1 = np.random.randn(D, 4*D)*0.1; W2 = np.random.randn(4*D, D)*0.1
    ffn_out = np.maximum(0, out @ W1) @ W2
    result = out + ffn_out
    attn_entropy = -np.sum(attn * np.log(attn + 1e-9), axis=-1).mean()
    print(f"  Input:  ({T}, {D})")
    print(f"  Output: {result.shape}")
    print(f"  Avg attention entropy: {attn_entropy:.4f} (full bidirectional)")


def demo_decoder_only_pattern():
    """Decoder-only (GPT-style): causal masked self-attention."""
    print("\n=== Decoder-Only (GPT-style) Causal Self-Attention ===")
    import numpy as np
    np.random.seed(0)
    T, D = 8, 16
    x = np.random.randn(T, D)
    W_q = np.random.randn(D, D)*0.1
    W_k = np.random.randn(D, D)*0.1
    W_v = np.random.randn(D, D)*0.1
    Q = x @ W_q; K = x @ W_k; V = x @ W_v
    scores = Q @ K.T / (D**0.5)
    # Causal mask: upper triangle = -inf
    mask = np.triu(np.full((T, T), -1e9), k=1)
    scores += mask
    attn = np.exp(scores - scores.max(-1, keepdims=True))
    attn /= attn.sum(-1, keepdims=True)
    out = attn @ V
    print(f"  Causal mask zeros future: {(attn[0, 1:] < 1e-8).all()}")  # token 0 can't see 1+
    print(f"  Attn[3] sums to: {attn[3].sum():.4f}  (only tokens 0..3 visible)")
    print(f"  Output shape: {out.shape}")


def demo_cross_attention():
    """Cross-attention: queries from decoder, keys/values from encoder."""
    print("\n=== Cross-Attention (Encoder-Decoder) ===")
    import numpy as np
    np.random.seed(7)
    T_enc, T_dec, D = 10, 6, 16
    enc = np.random.randn(T_enc, D)  # encoder output
    dec = np.random.randn(T_dec, D)  # decoder states
    W_q = np.random.randn(D, D)*0.1
    W_k = np.random.randn(D, D)*0.1
    W_v = np.random.randn(D, D)*0.1
    Q = dec @ W_q   # (T_dec, D)
    K = enc @ W_k   # (T_enc, D)
    V = enc @ W_v   # (T_enc, D)
    scores = Q @ K.T / (D**0.5)  # (T_dec, T_enc)
    attn = np.exp(scores - scores.max(-1, keepdims=True))
    attn /= attn.sum(-1, keepdims=True)
    out = attn @ V  # (T_dec, D)
    peak = attn.argmax(-1)  # which encoder token each decoder token attends to most
    print(f"  Encoder len={T_enc}, Decoder len={T_dec}")
    print(f"  Cross-attn weights shape: {attn.shape}")
    print(f"  Peak encoder positions attended by decoder: {peak}")
    print(f"  Output shape: {out.shape}")


if __name__ == "__main__":
    demo()
    demo_encoder_only_pattern()
    demo_decoder_only_pattern()
    demo_cross_attention()
