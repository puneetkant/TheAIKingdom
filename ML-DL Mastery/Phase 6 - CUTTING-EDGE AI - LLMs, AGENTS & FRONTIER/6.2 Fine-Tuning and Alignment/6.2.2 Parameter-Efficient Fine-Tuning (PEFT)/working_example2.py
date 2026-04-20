"""
Working Example 2: Parameter-Efficient Fine-Tuning (PEFT) / LoRA
Demonstrates LoRA low-rank decomposition: rank-r approximation of a weight
matrix, trainable parameter count reduction vs full fine-tuning.
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


def lora_approx(W, rank):
    """Low-rank approximation of W via SVD (simulates LoRA A@B)."""
    U, S, Vt = np.linalg.svd(W, full_matrices=False)
    W_approx = (U[:, :rank] * S[:rank]) @ Vt[:rank, :]
    error = np.linalg.norm(W - W_approx, "fro") / np.linalg.norm(W, "fro")
    return W_approx, error


def demo():
    print("=== PEFT / LoRA: Low-Rank Decomposition ===")
    rng = np.random.default_rng(42)
    d_in, d_out = 768, 768  # Typical transformer hidden size
    W = rng.standard_normal((d_out, d_in))

    full_params = d_in * d_out
    ranks = [1, 2, 4, 8, 16, 32, 64, 128]
    errors, lora_params, reduction = [], [], []

    for r in ranks:
        _, err = lora_approx(W, r)
        errors.append(err)
        p = r * (d_in + d_out)
        lora_params.append(p)
        reduction.append(100 * (1 - p / full_params))
        print(f"  rank={r:4d}: approx_error={err:.4f}, "
              f"params={p:,} ({reduction[-1]:.1f}% reduction)")

    print(f"\n  Full fine-tuning params: {full_params:,}")

    # Plot 1: Approximation error vs rank
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].plot(ranks, errors, "o-", color="tomato", lw=2)
    axes[0].set(xlabel="LoRA Rank (r)", ylabel="Relative Frobenius Error",
                title="Reconstruction Error vs Rank")
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Trainable parameters vs rank
    axes[1].bar(range(len(ranks)), [p / 1000 for p in lora_params],
                color="steelblue", tick_label=[str(r) for r in ranks])
    axes[1].axhline(full_params / 1000, color="red", linestyle="--",
                    lw=2, label="Full fine-tune")
    axes[1].set(xlabel="LoRA Rank (r)", ylabel="Trainable Params (K)",
                title="Param Count vs Rank")
    axes[1].legend()
    axes[1].grid(True, axis="y", alpha=0.3)

    # Plot 3: Parameter reduction %
    axes[2].bar(range(len(ranks)), reduction, color="mediumseagreen",
                tick_label=[str(r) for r in ranks])
    axes[2].set(xlabel="LoRA Rank (r)", ylabel="Parameter Reduction (%)",
                title="PEFT Efficiency: Param Reduction")
    axes[2].grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT / "peft_lora.png", dpi=100)
    plt.close()
    print("  Saved peft_lora.png")


def demo_adapter_layers():
    """Adapter layers: small bottleneck modules inserted between transformer layers."""
    print("\n=== Adapter Layers ===")
    d_model = 768
    bottleneck_dims = [8, 16, 32, 64, 128]
    for b in bottleneck_dims:
        # Adapter: down-project (d_model -> b), up-project (b -> d_model)
        adapter_params = d_model * b + b + b * d_model + d_model
        pct = 100 * adapter_params / (d_model * d_model)
        print(f"  bottleneck={b:4d}: adapter params={adapter_params:7,}  ({pct:.2f}% of one FFN weight)")
    plt.figure(figsize=(5, 3))
    adapter_p = [d_model*b + b + b*d_model + d_model for b in bottleneck_dims]
    plt.plot(bottleneck_dims, [p/1000 for p in adapter_p], "o-", color="darkorange", lw=2)
    plt.xlabel("Bottleneck Dim"); plt.ylabel("Adapter Params (K)")
    plt.title("Adapter Layer Parameter Count")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT / "adapter_layers.png", dpi=100); plt.close()
    print("  Saved adapter_layers.png")


def demo_rank_ablation():
    """Compare LoRA reconstruction quality across ranks for a low-rank matrix."""
    print("\n=== Rank Ablation on Low-Rank Target ===")
    rng = np.random.default_rng(0)
    d = 256
    # Create a truly low-rank matrix (rank 4)
    true_rank = 4
    L = rng.standard_normal((d, true_rank))
    R = rng.standard_normal((true_rank, d))
    W_low = L @ R  # intrinsically rank-4
    ranks = [1, 2, 4, 8, 16, 32]
    for r in ranks:
        _, err = lora_approx(W_low, r)
        print(f"  rank={r:3d}: error={err:.6f}")
    errors = [lora_approx(W_low, r)[1] for r in ranks]
    plt.figure(figsize=(5, 3))
    plt.plot(ranks, errors, "o-", color="mediumseagreen", lw=2)
    plt.axvline(true_rank, color="red", linestyle="--", label=f"True rank={true_rank}")
    plt.xlabel("LoRA Rank"); plt.ylabel("Reconstruction Error")
    plt.title("Rank Ablation on Low-Rank Matrix")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT / "rank_ablation.png", dpi=100); plt.close()
    print("  Saved rank_ablation.png")


if __name__ == "__main__":
    demo()
    demo_adapter_layers()
    demo_rank_ablation()
