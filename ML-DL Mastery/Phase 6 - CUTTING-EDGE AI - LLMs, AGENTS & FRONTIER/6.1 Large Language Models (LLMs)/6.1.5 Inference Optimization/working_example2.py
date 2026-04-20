"""
Working Example 2: Inference Optimization
Simulates KV-cache memory savings, batching throughput comparison,
and greedy vs beam search decoding.
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


def kv_cache_memory(seq_len, n_heads, head_dim, n_layers, dtype_bytes=2):
    """KV cache memory in MB for a given sequence length."""
    return 2 * seq_len * n_heads * head_dim * n_layers * dtype_bytes / (1024 ** 2)


def demo():
    print("=== Inference Optimization ===")

    # --- KV-cache memory savings ---
    n_heads, head_dim, n_layers = 32, 128, 32  # GPT-3 class
    seq_lengths = np.arange(128, 8193, 128)
    cache_mb = [kv_cache_memory(s, n_heads, head_dim, n_layers) for s in seq_lengths]
    no_cache_mb = [s * n_heads * head_dim * n_layers * 2 / (1024 ** 2) for s in seq_lengths]

    print(f"  KV-cache at 4096 tokens: {kv_cache_memory(4096, n_heads, head_dim, n_layers):.1f} MB")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].plot(seq_lengths, cache_mb, label="With KV-Cache", color="steelblue", lw=2)
    axes[0].fill_between(seq_lengths, cache_mb, alpha=0.2, color="steelblue")
    axes[0].set(xlabel="Sequence Length", ylabel="Memory (MB)",
                title="KV-Cache Memory vs Sequence Length")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # --- Batching throughput ---
    batch_sizes = [1, 2, 4, 8, 16, 32, 64]
    # Simulated tokens/sec (sub-linear scaling due to memory bandwidth)
    throughput = [b * 150 / (1 + 0.02 * b) for b in batch_sizes]
    axes[1].plot(batch_sizes, throughput, "o-", color="darkorange", lw=2)
    axes[1].set(xlabel="Batch Size", ylabel="Tokens / Second",
                title="Batching Throughput")
    axes[1].grid(True, alpha=0.3)
    print(f"  Throughput at batch=32: {throughput[5]:.1f} tok/s")

    # --- Greedy vs beam search comparison ---
    vocab_size = 50
    rng = np.random.default_rng(7)
    log_probs = rng.normal(-3, 1, vocab_size)
    top_k = np.argsort(log_probs)[-10:]
    beam_scores = log_probs[top_k]

    axes[2].bar(range(len(beam_scores)), np.sort(beam_scores)[::-1],
                color="#9b59b6", label="Beam candidates")
    axes[2].axhline(beam_scores.max(), color="red", linestyle="--", label="Greedy pick")
    axes[2].set(xlabel="Candidate Rank", ylabel="Log-Prob",
                title="Greedy vs Beam Search (top-10 tokens)")
    axes[2].legend()
    axes[2].grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT / "inference_optimization.png", dpi=100)
    plt.close()
    print("  Saved inference_optimization.png")


def demo_quantization_tradeoff():
    """Show model size vs accuracy tradeoff for different quantization levels."""
    print("\n=== Quantization Tradeoff ===")
    bits = [32, 16, 8, 4, 2]
    # Relative model size (FP32 = 1.0)
    size_ratio = [b / 32 for b in bits]
    # Simulated accuracy degradation (higher bits = better accuracy)
    accuracy = [95.0, 94.8, 94.2, 92.5, 85.0]
    for b, s, a in zip(bits, size_ratio, accuracy):
        print(f"  {b:2d}-bit: size={s:.2f}x  accuracy={a:.1f}%")

    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax2 = ax1.twinx()
    ax1.bar([str(b) for b in bits], size_ratio, color="steelblue", alpha=0.7, label="Relative Size")
    ax2.plot([str(b) for b in bits], accuracy, "o-", color="tomato", lw=2, label="Accuracy (%)")
    ax1.set_xlabel("Bit Width"); ax1.set_ylabel("Relative Model Size", color="steelblue")
    ax2.set_ylabel("Accuracy (%)", color="tomato")
    ax1.set_title("Quantization: Size vs Accuracy")
    plt.tight_layout()
    plt.savefig(OUTPUT / "quantization_tradeoff.png", dpi=100); plt.close()
    print("  Saved quantization_tradeoff.png")


def demo_speculative_decoding():
    """Simulate speculative decoding speedup vs acceptance rate."""
    print("\n=== Speculative Decoding ===")
    # Draft model proposes k tokens; accepted with probability p each
    k_values = [1, 2, 4, 8]
    acceptance_rates = np.linspace(0.5, 1.0, 50)
    plt.figure(figsize=(6, 4))
    for k in k_values:
        # Expected speedup = (1 - p^(k+1)) / ((1-p) * (1 + k * cost_ratio))
        cost_ratio = 0.1  # draft model is 10x cheaper
        speedup = (1 - acceptance_rates**(k+1)) / \
                  ((1 - acceptance_rates + 1e-9) * (1 + k * cost_ratio))
        speedup = np.clip(speedup, 1.0, None)
        plt.plot(acceptance_rates, speedup, lw=2, label=f"k={k}")
        print(f"  k={k}: speedup at p=0.9 ~ {float(speedup[np.argmin(np.abs(acceptance_rates-0.9))]):.2f}x")
    plt.xlabel("Draft Token Acceptance Rate")
    plt.ylabel("Speedup vs Autoregressive")
    plt.title("Speculative Decoding Speedup")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT / "speculative_decoding.png", dpi=100); plt.close()
    print("  Saved speculative_decoding.png")


if __name__ == "__main__":
    demo()
    demo_quantization_tradeoff()
    demo_speculative_decoding()
