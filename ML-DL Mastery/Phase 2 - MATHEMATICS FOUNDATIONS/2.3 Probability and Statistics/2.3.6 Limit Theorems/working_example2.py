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
    import matplotlib
    matplotlib.use("Agg")
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
        print(f"  n={n:>6}: p = {p_est:.4f}  |err| = {abs(p_est - p_true):.4f}")

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
    # integral01 x² dx = 1/3
    true_val = 1/3
    for N in [100, 1000, 10000, 100000]:
        x = np.random.uniform(0, 1, N)
        est = np.mean(x**2)
        se  = np.std(x**2) / np.sqrt(N)
        print(f"  N={N:>7}: est={est:.6f}  ±{1.96*se:.6f}  err={abs(est-true_val):.6f}")

def demo_clt_dice():
    print("\n=== CLT: Sum of Dice Rolls ===")
    np.random.seed(10)
    N = 5000
    ns = [1, 2, 5, 10, 30]
    fig, axes = plt.subplots(1, len(ns), figsize=(15, 4))
    for ax, n in zip(axes, ns):
        rolls = np.random.randint(1, 7, size=(N, n))
        sums = rolls.sum(axis=1)
        bins = range(n, 6 * n + 2)
        ax.hist(sums, bins=list(bins), density=True, alpha=0.75,
                color="steelblue", edgecolor="white")
        ax.set_title(f"n={n}")
        ax.set_xlabel("Sum")
    axes[0].set_ylabel("Density")
    plt.suptitle("CLT: Sum of n fair dice (N=5000 trials each)")
    plt.tight_layout()
    fig.savefig(OUTPUT / "clt_dice.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: clt_dice.png")


def demo_lln_coin():
    print("\n=== LLN: Biased Coin (p=0.3) ===")
    np.random.seed(20)
    p = 0.3
    n_flips = 1000
    flips = np.random.binomial(1, p, n_flips)
    running_mean = np.cumsum(flips) / np.arange(1, n_flips + 1)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(running_mean, color="steelblue", lw=1, label="Running mean")
    ax.axhline(p, color="red", ls="--", lw=2, label=f"True p={p}")
    ax.set_xlabel("Number of flips")
    ax.set_ylabel("Estimated P(Heads)")
    ax.set_title("LLN: Running mean of biased coin flips (p=0.3)")
    ax.legend()
    fig.savefig(OUTPUT / "lln_convergence.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Final estimate: {running_mean[-1]:.4f}  (true: {p})")
    print("  Saved: lln_convergence.png")


def demo_clt_exponential():
    print("\n=== CLT: Averaging Exponential(1) Samples ===")
    np.random.seed(30)
    ns = [1, 5, 30]
    x_range = np.linspace(-4, 4, 200)
    normal_pdf = np.exp(-0.5 * x_range ** 2) / np.sqrt(2 * np.pi)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, n in zip(axes, ns):
        samples = np.random.exponential(1.0, size=(10_000, n))
        means = samples.mean(axis=1)
        # E[X]=1, Var[X]=1 => std of mean = 1/sqrt(n)
        z = (means - 1.0) / (1.0 / np.sqrt(n))
        ax.hist(z, bins=50, density=True, alpha=0.7, color="coral", label="Empirical")
        ax.plot(x_range, normal_pdf, "k-", lw=2, label="N(0,1)")
        ax.set_title(f"n={n}")
        ax.set_xlim(-4, 4)
        ax.set_xlabel("Standardised mean")
    axes[0].set_ylabel("Density")
    axes[0].legend(fontsize=8)
    plt.suptitle("CLT: Standardised mean of Exponential(1) samples")
    plt.tight_layout()
    fig.savefig(OUTPUT / "clt_exponential.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: clt_exponential.png")


if __name__ == "__main__":
    demo_lln()
    demo_clt()
    demo_bootstrap()
    demo_mc_integration()
    demo_clt_dice()
    demo_lln_coin()
    demo_clt_exponential()
