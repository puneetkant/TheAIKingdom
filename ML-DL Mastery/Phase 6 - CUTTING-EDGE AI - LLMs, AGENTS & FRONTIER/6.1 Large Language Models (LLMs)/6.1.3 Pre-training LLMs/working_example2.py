"""
Working Example 2: Pre-training LLMs
Character-level bigram model demonstrating next-token prediction loss
and perplexity vs training steps.
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


def demo():
    print("=== Pre-training LLMs: Character-Level Bigram Model ===")
    rng = np.random.default_rng(42)
    vocab_size = 27  # a-z + space

    # Simulated bigram count matrix -> probability matrix
    counts = rng.integers(1, 100, size=(vocab_size, vocab_size)).astype(float)
    probs = counts / counts.sum(axis=1, keepdims=True)

    print(f"  Vocab size: {vocab_size}")
    print(f"  Prob matrix shape: {probs.shape}")
    print(f"  Sample row sums: {probs.sum(axis=1)[:3].round(4)}")

    # Simulate perplexity decay over training steps
    steps = np.arange(1, 201)
    noise = rng.normal(0, 0.3, len(steps))
    perplexity = 22 * np.exp(-0.025 * steps) + 5.5 + np.abs(noise)
    loss = np.log(perplexity)

    print(f"  Initial perplexity: {perplexity[0]:.2f}")
    print(f"  Final   perplexity: {perplexity[-1]:.2f}")

    # --- Plot 1: Training curves ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(steps, perplexity, color="steelblue", linewidth=1.8)
    axes[0].set(xlabel="Training Steps", ylabel="Perplexity",
                title="Perplexity vs Training Steps")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(steps, loss, color="tomato", linewidth=1.8)
    axes[1].set(xlabel="Training Steps", ylabel="Cross-Entropy Loss",
                title="Training Loss Curve")
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT / "pretraining_curves.png", dpi=100)
    plt.close()
    print("  Saved pretraining_curves.png")

    # --- Plot 2: Bigram heatmap ---
    fig2, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(probs[:12, :12], cmap="Blues")
    ax.set(title="Bigram Transition Probabilities (12×12 subset)",
           xlabel="Next Token", ylabel="Current Token")
    plt.colorbar(im, ax=ax, label="P(next | current)")
    plt.tight_layout()
    plt.savefig(OUTPUT / "bigram_heatmap.png", dpi=100)
    plt.close()
    print("  Saved bigram_heatmap.png")


def demo_scaling_laws():
    """Illustrate Chinchilla-style scaling: loss decreases with model size N."""
    print("\n=== Scaling Laws ===")
    rng = np.random.default_rng(7)
    # Loss ~ A / N^alpha  (simplified Chinchilla power-law form)
    N = np.logspace(6, 11, 50)  # 1M to 100B parameters
    alpha = 0.076
    A = 1.8
    loss_N = A / (N ** alpha) + rng.normal(0, 0.01, len(N))

    # Compute tokens for compute-optimal training: D ~ 20 * N
    D = 20 * N
    print(f"  Model 1B params: optimal tokens ~{20e9/1e9:.0f}B")
    print(f"  Model 7B params: optimal tokens ~{20*7e9/1e9:.0f}B")
    print(f"  Model 70B params: optimal tokens ~{20*70e9/1e9:.0f}B")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(N / 1e9, loss_N, color="steelblue", lw=2)
    axes[0].set(xscale="log", xlabel="Model Size (B params)",
                ylabel="Validation Loss", title="Loss vs Model Size (Scaling Law)")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(N / 1e9, D / 1e9, color="darkorange", lw=2)
    axes[1].set(xscale="log", yscale="log", xlabel="Model Size (B params)",
                ylabel="Optimal Tokens (B)", title="Chinchilla: Optimal Training Tokens")
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT / "scaling_laws.png", dpi=100); plt.close()
    print("  Saved scaling_laws.png")


def demo_data_quality_effect():
    """Show how corpus quality (noise level) affects perplexity."""
    print("\n=== Data Quality Effect ===")
    rng = np.random.default_rng(99)
    steps = np.arange(1, 101)
    quality_levels = {"High quality": 0.5, "Medium quality": 2.0, "Low quality": 5.0}
    plt.figure(figsize=(6, 4))
    for label, noise in quality_levels.items():
        ppl = 22 * np.exp(-0.025 * steps) + 5.5 + np.abs(rng.normal(0, noise, len(steps)))
        plt.plot(steps, ppl, lw=2, label=label)
        print(f"  {label}: final perplexity ~ {ppl[-1]:.2f}")
    plt.xlabel("Training Steps"); plt.ylabel("Perplexity")
    plt.title("Data Quality vs Perplexity")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT / "data_quality.png", dpi=100); plt.close()
    print("  Saved data_quality.png")


if __name__ == "__main__":
    demo()
    demo_scaling_laws()
    demo_data_quality_effect()
