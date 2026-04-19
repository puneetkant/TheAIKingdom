"""
Working Example: State Space Models (SSMs)
Covers the Mamba architecture, S4, H3, linear attention alternatives,
and comparison to transformers.
"""
import numpy as np
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_ssm")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -- 1. Motivation -------------------------------------------------------------
def ssm_motivation():
    print("=== State Space Models (SSMs) ===")
    print()
    print("  Problem with Transformers:")
    print("    O(L²) attention complexity (L = sequence length)")
    print("    KV cache grows linearly with context")
    print("    Great for short-range, struggle with 100k+ tokens efficiently")
    print()
    print("  SSM promise:")
    print("    O(L) inference (constant state size regardless of history)")
    print("    O(L log L) training (parallel scan; similar to FFT)")
    print("    Recurrent-like memory; infinite theoretical context")
    print()
    print("  SSM family timeline:")
    models = [
        ("HiPPO",   "2020; Gu; optimal polynomial projection of history"),
        ("S4",      "2021; structured state space; HiPPO + diagonal init"),
        ("H3",      "2022; hybrid SSM+attention; fill attention gaps"),
        ("Hyena",   "2023; long convolution; sub-quadratic; no SSM"),
        ("Mamba",   "2023; selective SSM; data-dependent; SOTA"),
        ("Mamba-2", "2024; SSD; 8x faster; theoretical unification"),
        ("RWKV",    "2023; RNN+transformer hybrid; open; trained on 100B+"),
    ]
    for m, d in models:
        print(f"  {m:<10} {d}")


# -- 2. SSM mechanics ----------------------------------------------------------
def ssm_mechanics():
    print("\n=== SSM Mechanics ===")
    print()
    print("  Continuous-time SSM:")
    print("    h'(t) = A h(t) + B x(t)")
    print("    y(t)  = C h(t) + D x(t)")
    print("    h: hidden state; x: input; y: output; A,B,C,D: learnable")
    print()
    print("  Discrete-time (for sequences with step size Delta):")
    print("    A_bar = exp(Delta * A)")
    print("    B_bar = (Delta * A)^-1 (exp(Delta * A) - I) * Delta * B")
    print("    h_t   = A_bar * h_{t-1} + B_bar * x_t")
    print("    y_t   = C * h_t")
    print()

    # Simulate a simple SSM (1D state for clarity)
    N = 20  # sequence length
    A = np.array([-0.5])  # stable decay
    B = np.array([1.0])
    C = np.array([1.0])
    delta = 0.1

    A_bar = np.exp(delta * A)
    B_bar = (A_bar - 1) / A * B * delta

    h = np.zeros(1)
    x = np.sin(np.linspace(0, 4 * np.pi, N))
    outputs = []
    for xt in x:
        h = A_bar * h + B_bar * xt
        outputs.append(float(C @ h))

    print(f"  Simulated SSM output (1D state, sinusoidal input, N={N}):")
    print(f"  Input:  {' '.join(f'{v:+.2f}' for v in x[:10])} ...")
    print(f"  Output: {' '.join(f'{v:+.2f}' for v in outputs[:10])} ...")


# -- 3. Mamba: selective SSM ---------------------------------------------------
def mamba_architecture():
    print("\n=== Mamba: Selective State Space ===")
    print()
    print("  Key insight: make B, C, Delta input-dependent (selective)")
    print("  Unlike S4: constant A,B,C,D cannot focus on relevant tokens")
    print("  Mamba: Delta, B, C = linear projections of input x_t")
    print("         -> model chooses what to remember/forget per token")
    print()
    print("  Mamba block:")
    print("    Input (D_model) --------------------------------------+")
    print("    -> linear expand (2x) -> split:")
    print("      Branch 1: Conv1D -> SSM (selective scan) -> gate")
    print("      Branch 2: SiLU activation                  v")
    print("    -> elementwise multiply -> linear project -> output")
    print()
    print("  Advantages over transformers at inference:")
    advantages = [
        ("Constant memory", "State h is fixed size; no KV cache growth"),
        ("O(1) per step",   "Process each token in O(N * D_state) vs O(L * D)"),
        ("Long context",    "Efficient memory compression; no quadratic blowup"),
    ]
    for a, d in advantages:
        print(f"  {a:<18} {d}")
    print()
    print("  Mamba-2 / SSD:")
    print("    Restricts A to scalar matrix -> enables tensor parallelism")
    print("    8x faster than Mamba-1; theoretical connection to linear attention")
    print()
    print("  Hybrid models (best of both):")
    hybrids = [
        ("Jamba",    "AI21; Mamba + attention layers alternating"),
        ("Zamba",    "Zyphra; 7B; Mamba-2 + shared attention; efficient"),
        ("Falcon-H1","TII; hybrid Mamba-2 + attention; 0.5–34B"),
    ]
    for h, d in hybrids:
        print(f"  {h:<12} {d}")


# -- 4. Benchmark comparison ---------------------------------------------------
def ssm_benchmarks():
    print("\n=== SSM vs Transformer Benchmarks ===")
    print()
    print("  Memory usage at sequence length L (batch=1, D=4096):")
    print(f"  {'L':>8} {'Transformer (KV cache)':>24} {'SSM (state)':>14}")
    for L in [1_000, 10_000, 100_000, 1_000_000]:
        kv_mb  = L * 2 * 32 * 4096 * 2 / 1e6  # 32 layers, KV, float16
        ssm_mb = 32 * 16 * 4096 * 2 / 1e6     # 32 layers, D_state=16, fixed
        print(f"  {L:>8,} {kv_mb:>22.1f} MB {ssm_mb:>12.1f} MB")
    print()
    print("  Current status (2025):")
    print("  SSMs match transformers on short-context tasks")
    print("  Hybrids (Mamba+attn) often best trade-off")
    print("  Pure transformers still lead on very long context (needle-in-haystack)")


if __name__ == "__main__":
    ssm_motivation()
    ssm_mechanics()
    mamba_architecture()
    ssm_benchmarks()
