"""
Working Example 2: Context Length Extension
Compares RoPE vs sinusoidal positional encodings and visualises
attention patterns for long sequences.
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


def sinusoidal_encoding(seq_len, d_model):
    """Classic sinusoidal positional encoding."""
    pos = np.arange(seq_len)[:, None]
    div = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(pos * div)
    pe[:, 1::2] = np.cos(pos * div)
    return pe


def rope_encoding(seq_len, d_model):
    """Simplified RoPE: rotary positional encoding (even dims only)."""
    theta = np.arange(0, d_model, 2) / d_model
    freqs = 1.0 / (10000 ** theta)
    t = np.arange(seq_len)
    angles = np.outer(t, freqs)
    cos = np.cos(angles)
    sin = np.sin(angles)
    # Interleave cos/sin for visualisation
    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = cos
    pe[:, 1::2] = sin
    return pe


def demo():
    print("=== Context Length Extension: RoPE vs Sinusoidal ===")
    seq_len, d_model = 512, 64
    sin_pe = sinusoidal_encoding(seq_len, d_model)
    rope_pe = rope_encoding(seq_len, d_model)

    print(f"  Sequence length: {seq_len}, d_model: {d_model}")
    print(f"  Sinusoidal PE shape: {sin_pe.shape}")
    print(f"  RoPE PE shape:       {rope_pe.shape}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Heatmaps
    axes[0][0].imshow(sin_pe[:64, :32].T, aspect="auto", cmap="RdBu", vmin=-1, vmax=1)
    axes[0][0].set(title="Sinusoidal PE (first 64 positions, 32 dims)",
                   xlabel="Position", ylabel="Dimension")

    axes[0][1].imshow(rope_pe[:64, :32].T, aspect="auto", cmap="RdBu", vmin=-1, vmax=1)
    axes[0][1].set(title="RoPE PE (first 64 positions, 32 dims)",
                   xlabel="Position", ylabel="Dimension")

    # Cosine similarity decay with distance (extrapolation test)
    query_pos = 0
    sim_sin = [float(np.dot(sin_pe[query_pos], sin_pe[i]) /
                     (np.linalg.norm(sin_pe[query_pos]) * np.linalg.norm(sin_pe[i]) + 1e-8))
               for i in range(seq_len)]
    sim_rope = [float(np.dot(rope_pe[query_pos], rope_pe[i]) /
                      (np.linalg.norm(rope_pe[query_pos]) * np.linalg.norm(rope_pe[i]) + 1e-8))
                for i in range(seq_len)]

    axes[1][0].plot(sim_sin, color="steelblue", lw=1.5, label="Sinusoidal")
    axes[1][0].plot(sim_rope, color="tomato", lw=1.5, label="RoPE")
    axes[1][0].set(xlabel="Position", ylabel="Cosine Similarity to pos 0",
                   title="Position Similarity Decay")
    axes[1][0].legend()
    axes[1][0].grid(True, alpha=0.3)

    # Simulated attention pattern for long context
    rng = np.random.default_rng(42)
    attn_len = 64
    attn = rng.exponential(scale=1.0, size=(attn_len, attn_len))
    # Causal mask + local bias
    for i in range(attn_len):
        attn[i, i+1:] = 0
        attn[i, max(0, i-8):i+1] *= 3  # local attention boost
    attn = attn / attn.sum(axis=1, keepdims=True)
    im = axes[1][1].imshow(attn, cmap="Oranges", aspect="auto")
    axes[1][1].set(title="Simulated Causal Attention Pattern (local bias)",
                   xlabel="Key Position", ylabel="Query Position")
    plt.colorbar(im, ax=axes[1][1])

    plt.tight_layout()
    plt.savefig(OUTPUT / "context_length.png", dpi=100)
    plt.close()
    print("  Saved context_length.png")


if __name__ == "__main__":
    demo()
