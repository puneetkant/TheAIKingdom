"""
Working Example 2: Compute Resources
FLOP estimator, GPU memory calculator, and training time projection.
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


def transformer_flops_per_token(d_model, n_heads, n_layers, seq_len, ffn_mult=4):
    """
    Estimate FLOPs per forward pass token for a transformer.
    Approximate: 2 * n_params per token (Kaplan et al. scaling law).
    """
    d_head = d_model // n_heads
    # Attention: Q,K,V projections + attn + output proj
    attn_flops = 4 * d_model * d_model + 2 * seq_len * d_model  # per layer
    # FFN: 2 linear layers
    ffn_flops = 2 * ffn_mult * d_model * d_model  # per layer
    return n_layers * (attn_flops + ffn_flops)


def count_params(d_model, n_heads, n_layers, vocab_size=50000, ffn_mult=4):
    """Estimate parameter count for a transformer model."""
    embedding = vocab_size * d_model
    attn_params = n_layers * 4 * d_model * d_model   # Q,K,V,O
    ffn_params = n_layers * 2 * ffn_mult * d_model * d_model
    return embedding + attn_params + ffn_params


def estimate_gpu_memory_gb(n_params, dtype_bytes=2, batch_size=1, seq_len=512,
                            d_model=512, optimizer_factor=12):
    """
    Estimate GPU memory for training.
    - Weights: n_params * dtype_bytes
    - Gradients: n_params * dtype_bytes
    - Optimizer states (Adam): n_params * 8 bytes (fp32 * 2 states)
    - Activations: approximate
    """
    weights = n_params * dtype_bytes
    grads = n_params * dtype_bytes
    optimizer = n_params * 8  # Adam fp32
    activations = batch_size * seq_len * d_model * dtype_bytes * 12  # rough
    total_bytes = weights + grads + optimizer + activations
    return total_bytes / 1e9


def training_time_hours(n_tokens, n_params, gpu_flops_per_sec=312e12, mfu=0.4):
    """
    Chinchilla training time estimate.
    FLOPs needed ~= 6 * N * D (N=params, D=tokens).
    """
    total_flops = 6 * n_params * n_tokens
    effective_flops = gpu_flops_per_sec * mfu
    return total_flops / effective_flops / 3600


def demo():
    print("=== Compute Resources: FLOP & Memory Estimator ===")

    configs = [
        {"name": "Small  (125M)", "d": 768,  "h": 12, "L": 12, "seq": 2048},
        {"name": "Medium (350M)", "d": 1024, "h": 16, "L": 24, "seq": 2048},
        {"name": "Large  (1.3B)", "d": 2048, "h": 16, "L": 24, "seq": 2048},
        {"name": "XL     (6.7B)", "d": 4096, "h": 32, "L": 32, "seq": 4096},
        {"name": "XXL    (70B)",  "d": 8192, "h": 64, "L": 80, "seq": 4096},
    ]

    print(f"\n  {'Model':15s} {'Params (M)':12s} {'FLOPS/tok (G)':14s} {'Mem 1GPU (GB)':14s} {'Train 1xH100 (hr)':18s}")
    print("  " + "-" * 78)

    params_list, flops_list, mem_list, time_list = [], [], [], []
    for cfg in configs:
        n_params = count_params(cfg["d"], cfg["h"], cfg["L"])
        flops = transformer_flops_per_token(cfg["d"], cfg["h"], cfg["L"], cfg["seq"])
        mem_gb = estimate_gpu_memory_gb(n_params, d_model=cfg["d"])
        # Chinchilla: D = 20*N tokens
        train_hrs = training_time_hours(20 * n_params, n_params)
        params_list.append(n_params / 1e6)
        flops_list.append(flops / 1e9)
        mem_list.append(mem_gb)
        time_list.append(train_hrs)
        print(f"  {cfg['name']:15s} {n_params/1e6:12.1f} {flops/1e9:14.2f} {mem_gb:14.2f} {train_hrs:18.1f}")

    # Scaling law: compute-optimal tokens vs model size
    n_params_range = np.logspace(7, 11, 100)
    chinchilla_tokens = 20 * n_params_range  # C-optimal

    # MFU sensitivity
    mfu_values = np.linspace(0.1, 0.9, 50)
    train_hrs_mfu = [training_time_hours(20 * 1.3e9, 1.3e9, mfu=mfu) for mfu in mfu_values]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # Params vs FLOPs
    axes[0][0].bar([c["name"] for c in configs], params_list, color="steelblue", alpha=0.8)
    axes[0][0].set(ylabel="Parameters (M)", title="Model Scale Comparison")
    axes[0][0].tick_params(axis="x", rotation=20)
    axes[0][0].set_yscale("log")
    axes[0][0].grid(True, axis="y", alpha=0.3)

    # GPU Memory
    axes[0][1].bar([c["name"] for c in configs], mem_list, color="tomato", alpha=0.8)
    axes[0][1].axhline(80, color="k", linestyle="--", alpha=0.5, label="A100 80GB")
    axes[0][1].set(ylabel="GPU Memory (GB)", title="Training Memory Estimate (BS=1)")
    axes[0][1].tick_params(axis="x", rotation=20)
    axes[0][1].legend()
    axes[0][1].grid(True, axis="y", alpha=0.3)

    # Chinchilla scaling
    axes[1][0].loglog(n_params_range / 1e9, chinchilla_tokens / 1e9,
                       color="mediumseagreen", lw=2)
    axes[1][0].set(xlabel="Model Size (B params)", ylabel="Optimal Tokens (B)",
                   title="Chinchilla Scaling: Optimal Token Budget")
    axes[1][0].grid(True, alpha=0.3)

    # MFU vs training time
    axes[1][1].plot(mfu_values, train_hrs_mfu, color="steelblue", lw=2)
    axes[1][1].axvline(0.4, color="red", linestyle="--", label="Typical MFU=0.4")
    axes[1][1].set(xlabel="Model FLOP Utilisation (MFU)", ylabel="Training Hours (1xH100)",
                   title="1.3B Model: Training Time vs MFU")
    axes[1][1].legend()
    axes[1][1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT / "compute_resources.png", dpi=100)
    plt.close()
    print("\n  Saved compute_resources.png")


def demo_multi_gpu_scaling():
    """Show training speedup and efficiency for different GPU counts."""
    print("\n=== Multi-GPU Scaling Efficiency ===")
    gpu_counts = [1, 2, 4, 8, 16, 32, 64]
    # Amdahl's law: parallel fraction p=0.95
    p = 0.95
    ideal_speedup  = np.array(gpu_counts, dtype=float)
    actual_speedup = np.array([1 / ((1-p) + p/n) for n in gpu_counts])
    efficiency     = actual_speedup / ideal_speedup * 100
    print(f"  {'GPUs':>6} {'Ideal':>8} {'Actual':>8} {'Efficiency':>12}")
    for n, ideal, actual, eff in zip(gpu_counts, ideal_speedup, actual_speedup, efficiency):
        print(f"  {n:>6} {ideal:>8.1f}x {actual:>8.2f}x {eff:>11.1f}%")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(gpu_counts, ideal_speedup,  "o--", color="gray",       lw=2, label="Ideal")
    axes[0].plot(gpu_counts, actual_speedup, "o-",  color="steelblue",  lw=2, label="Actual (Amdahl p=0.95)")
    axes[0].set(xlabel="GPU Count", ylabel="Speedup", title="Multi-GPU Speedup")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].plot(gpu_counts, efficiency, "o-", color="tomato", lw=2)
    axes[1].axhline(100, color="gray", linestyle="--", lw=1)
    axes[1].set(xlabel="GPU Count", ylabel="Efficiency (%)", title="Parallel Efficiency")
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT / "multi_gpu_scaling.png", dpi=100); plt.close()
    print("  Saved multi_gpu_scaling.png")


def demo_cost_comparison():
    """Compare cloud GPU instance costs for training a 1.3B model."""
    print("\n=== Cloud GPU Cost Comparison ===")
    instances = [
        {"name": "A100 80GB",  "flops_tflops": 312, "cost_per_hr": 3.06},
        {"name": "H100 80GB",  "flops_tflops": 989, "cost_per_hr": 9.80},
        {"name": "A10G 24GB",  "flops_tflops":  31, "cost_per_hr": 0.76},
        {"name": "T4 16GB",    "flops_tflops":  65, "cost_per_hr": 0.35},
        {"name": "RTX 4090",   "flops_tflops": 165, "cost_per_hr": 0.74},
    ]
    n_params, n_tokens = 1.3e9, 26e9  # Chinchilla-optimal
    total_flops = 6 * n_params * n_tokens
    mfu = 0.4
    print(f"  {'Instance':14s} {'Time (hr)':10s} {'Cost ($)':10s} {'$/TFLOP':10s}")
    for inst in instances:
        eff_tflops = inst["flops_tflops"] * 1e12 * mfu
        train_hr = total_flops / eff_tflops / 3600
        cost = train_hr * inst["cost_per_hr"]
        cost_per_tflop = inst["cost_per_hr"] / (inst["flops_tflops"] * mfu)
        print(f"  {inst['name']:14s} {train_hr:10.1f} {cost:10.2f} {cost_per_tflop:10.4f}")
    costs = [inst["cost_per_hr"] / (inst["flops_tflops"] * mfu) for inst in instances]
    plt.figure(figsize=(6, 3))
    plt.bar([i["name"] for i in instances], costs, color="steelblue", edgecolor="white")
    plt.xlabel("Instance"); plt.ylabel("$/Effective TFLOP-hr")
    plt.title("Cloud GPU Cost Efficiency (lower=better)")
    plt.xticks(rotation=20, ha="right"); plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT / "gpu_cost_comparison.png", dpi=100); plt.close()
    print("  Saved gpu_cost_comparison.png")


if __name__ == "__main__":
    demo()
    demo_multi_gpu_scaling()
    demo_cost_comparison()
