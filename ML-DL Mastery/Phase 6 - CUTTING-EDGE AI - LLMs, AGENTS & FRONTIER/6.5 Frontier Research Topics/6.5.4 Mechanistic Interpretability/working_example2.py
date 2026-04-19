"""
Working Example 2: Mechanistic Interpretability
Attention head analysis and activation patching on a toy transformer.
Run: python working_example2.py
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


def attention(Q, K, V, mask=None):
    """Scaled dot-product attention."""
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    if mask is not None:
        scores = np.where(mask, scores, -1e9)
    weights = softmax(scores, axis=-1)
    return weights @ V, weights


def logit_lens(residual_stream, W_unembed, top_k=3):
    """Project intermediate residual to vocabulary logits."""
    logits = residual_stream @ W_unembed
    top_tokens = np.argsort(logits)[::-1][:top_k]
    return logits, top_tokens


def demo():
    print("=== Mechanistic Interpretability ===")
    rng = np.random.default_rng(42)

    T, d_model, d_k, n_heads = 8, 32, 8, 4
    n_vocab = 50

    # Random token embeddings (proxy)
    tokens = rng.integers(0, n_vocab, T)
    E = rng.standard_normal((n_vocab, d_model))
    residual = E[tokens]  # (T, d_model)

    # Multiple attention heads
    head_weights = []
    for h in range(n_heads):
        Wq = rng.standard_normal((d_model, d_k)) * 0.1
        Wk = rng.standard_normal((d_model, d_k)) * 0.1
        Wv = rng.standard_normal((d_model, d_k)) * 0.1
        Q = residual @ Wq
        K = residual @ Wk
        V = residual @ Wv
        # Causal mask
        mask = np.tril(np.ones((T, T), dtype=bool))
        _, attn_w = attention(Q, K, V, mask)
        head_weights.append(attn_w)

    # Logit lens: project residual at each position to vocab
    W_unembed = rng.standard_normal((d_model, n_vocab)) * 0.1
    logits_by_pos = []
    for pos in range(T):
        logits, top_toks = logit_lens(residual[pos], W_unembed, top_k=3)
        logits_by_pos.append(logits)
        print(f"  Pos {pos} (token {tokens[pos]}): top predicted next = {top_toks}")

    # Activation patching: measure effect of patching position 3
    patch_pos = 3
    patch_token = rng.integers(0, n_vocab)
    patched_residual = residual.copy()
    patched_residual[patch_pos] = E[patch_token]

    # Effect on output logits at final position
    orig_logits = residual[-1] @ W_unembed
    patch_logits = patched_residual[-1] @ W_unembed
    patch_effect = np.abs(patch_logits - orig_logits)

    print(f"\n  Activation patching at pos {patch_pos}: mean effect = {patch_effect.mean():.4f}")

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # Head 0 attention pattern
    im0 = axes[0][0].imshow(head_weights[0], cmap="Blues", vmin=0, vmax=1)
    axes[0][0].set(title="Head 0 Attention Pattern", xlabel="Key Position", ylabel="Query Position")
    plt.colorbar(im0, ax=axes[0][0])

    # All heads attention entropy
    entropies = []
    for hw in head_weights:
        ent = -np.sum(hw * np.log(hw + 1e-10), axis=-1)
        entropies.append(ent.mean())
    axes[0][1].bar(range(n_heads), entropies, color="steelblue")
    axes[0][1].set(xlabel="Head", ylabel="Mean Attention Entropy",
                   title="Attention Entropy per Head (higher=more spread)")
    axes[0][1].grid(True, axis="y", alpha=0.3)

    # Logit lens heatmap
    logit_matrix = np.array(logits_by_pos)  # (T, n_vocab)
    im2 = axes[1][0].imshow(logit_matrix.T, cmap="RdYlGn", aspect="auto",
                              vmin=logit_matrix.min(), vmax=logit_matrix.max())
    axes[1][0].set(xlabel="Position", ylabel="Vocab Token",
                   title="Logit Lens: Residual -> Vocab Logits")
    plt.colorbar(im2, ax=axes[1][0])

    # Patching effect across vocab
    axes[1][1].bar(range(min(20, n_vocab)), patch_effect[:20], color="tomato")
    axes[1][1].set(xlabel="Vocab Token (first 20)", ylabel="|Delta Logit|",
                   title=f"Activation Patching Effect (pos {patch_pos})")
    axes[1][1].grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT / "mechanistic_interp.png", dpi=100)
    plt.close()
    print("  Saved mechanistic_interp.png")


if __name__ == "__main__":
    demo()
