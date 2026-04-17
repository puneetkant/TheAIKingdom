"""
Working Example 2: Limit Theorems — LLN, CLT, Monte Carlo Convergence
======================================================================
Law of Large Numbers, Central Limit Theorem, CLT for sample means,
bootstrap CI, and Monte Carlo integration with error bounds.

Run:  python working_example2.py
"""
from pathlib import Path
try:
    import numpy as np
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit("pip install numpy matplotlib")

OUTPUT = Path(__file__).parent / "output"
OUTPUT.mkdir(exist_ok=True)

def demo_lln():
    print("=== Law of Large Numbers ===")
    np.random.seed(42)
    # Biased coin: P(H)=0.6
    p_true = 0.6
    ns = [10, 50, 100, 500, 1000, 5000, 10000]
    samples = np.random.binomial(1, p_true, max(ns))
    for n in ns:
        p_est = samples[:n].mean()
        print(f"  n={n:>6}: p̂ = {p_est:.4f}  |err| = {abs(p_est - p_true):.4f}")

    # Plot running average
    running = np.cumsum(samples) / np.arange(1, len(samples)+1)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogx(running, lw=1, color="steelblue", label="running mean")
    ax.axhline(p_true, color="r", ls="--", label=f"true p={p_true}")
    ax.set_xlabel("n"); ax.set_ylabel("Sample mean"); ax.legend(); ax.set_title("LLN")
    fig.savefig(OUTPUT / "lln.png", dpi=120, bbox_inches="tight")
    plt.close(fig); print(f"  Saved: lln.png")

def demo_clt():
    print("\n=== Central Limit Theorem ===")
    np.random.seed(0)
    # X ~ Exponential(1): E=1, Var=1
    # Sample mean of n draws: ~ N(1, 1/n)
    fig, axes = plt.subplots(1, 4, figsize=(12, 4))
    for ax, n in zip(axes, [1, 5, 30, 100]):
        samples = np.random.exponential(1, (10_000, n))
        means = samples.mean(axis=1)
        # Standardise
        z = (means - 1) / (1 / np.sqrt(n))
        ax.hist(z, bins=50, density=True, alpha=0.7, color="coral")
        # Overlay N(0,1)
        x = np.linspace(-4, 4, 200)
        ax.plot(x, np.exp(-0.5*x**2)/np.sqrt(2*np.pi), "k", lw=2)
        ax.set_title(f"n={n}"); ax.set_xlim(-4, 4)
    axes[0].set_ylabel("Density")
    plt.suptitle("CLT: Standardised sample mean of Exp(1)")
    plt.tight_layout()
    fig.savefig(OUTPUT / "clt.png", dpi=120, bbox_inches="tight")
    plt.close(fig); print(f"  Saved: clt.png")

def demo_bootstrap():
    print("\n=== Bootstrap Confidence Interval ===")
    np.random.seed(7)
    data = np.random.exponential(2.0, 50)    # true mean = 2
    obs_mean = data.mean()
    print(f"  Observed mean: {obs_mean:.4f}  (true: 2.0)")

    # Bootstrap
    B = 10_000
    boot_means = np.array([np.random.choice(data, len(data), replace=True).mean() for _ in range(B)])
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
    print(f"  95% Bootstrap CI: [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"  True mean in CI: {ci_low <= 2.0 <= ci_high}")

def demo_mc_integration():
    print("\n=== Monte Carlo Integration ===")
    np.random.seed(1)
    # ∫₀¹ x² dx = 1/3
    true_val = 1/3
    for N in [100, 1000, 10000, 100000]:
        x = np.random.uniform(0, 1, N)
        est = np.mean(x**2)
        se  = np.std(x**2) / np.sqrt(N)
        print(f"  N={N:>7}: est={est:.6f}  ±{1.96*se:.6f}  err={abs(est-true_val):.6f}")

if __name__ == "__main__":
    demo_lln()
    demo_clt()
    demo_bootstrap()
    demo_mc_integration()
