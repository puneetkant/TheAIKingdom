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


if __name__ == "__main__":
    demo()
