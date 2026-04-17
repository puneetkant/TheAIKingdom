"""
Working Example 2: Random Variables — PMF, PDF, CDF, Moments
=============================================================
Discrete (Bernoulli/Binomial/Poisson) and continuous (Uniform/Gaussian)
random variables; expectation, variance, MGF demo, simulation.

Run:  python working_example2.py
"""
import random, math
from pathlib import Path
try:
    import numpy as np
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit("pip install numpy matplotlib")

OUTPUT = Path(__file__).parent / "output"
OUTPUT.mkdir(exist_ok=True)

def demo_discrete():
    print("=== Discrete Random Variables ===")
    np.random.seed(42)

    # Binomial(n=10, p=0.3)
    n_trials, p = 10, 0.3
    samples = np.random.binomial(n_trials, p, size=10_000)
    E_X = np.mean(samples);   Var_X = np.var(samples)
    print(f"  Binomial(n={n_trials}, p={p}):")
    print(f"    E[X] = {E_X:.4f}  (theory: {n_trials*p:.4f})")
    print(f"    Var[X]= {Var_X:.4f}  (theory: {n_trials*p*(1-p):.4f})")

    # Poisson(λ=3)
    lam = 3
    samples_p = np.random.poisson(lam, size=10_000)
    print(f"  Poisson(λ={lam}):")
    print(f"    E[X] = {np.mean(samples_p):.4f}  Var[X]= {np.var(samples_p):.4f}  (both ≈ {lam})")

def demo_continuous():
    print("\n=== Continuous Random Variables ===")
    np.random.seed(0)

    # Gaussian
    mu, sigma = 2.0, 1.5
    samples = np.random.normal(mu, sigma, size=50_000)
    print(f"  Gaussian(μ={mu}, σ={sigma}):")
    print(f"    E[X] = {np.mean(samples):.4f}  std = {np.std(samples):.4f}")

    # CDF by simulation: P(X ≤ t)
    for t in [0.0, mu - sigma, mu, mu + sigma]:
        p = np.mean(samples <= t)
        print(f"    P(X ≤ {t:.1f}) ≈ {p:.4f}")

    # Plot PDF + CDF
    x = np.linspace(mu - 4*sigma, mu + 4*sigma, 300)
    pdf = np.exp(-0.5*((x-mu)/sigma)**2) / (sigma * math.sqrt(2*math.pi))
    cdf = 0.5 * (1 + np.array([math.erf((xi - mu)/(sigma*math.sqrt(2))) for xi in x]))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(samples, bins=80, density=True, alpha=0.5, color="steelblue")
    axes[0].plot(x, pdf, "r", lw=2); axes[0].set_title("Gaussian PDF")
    axes[1].plot(x, cdf, lw=2, color="purple"); axes[1].set_title("Gaussian CDF")
    fig.savefig(OUTPUT / "gaussian_pdf_cdf.png", dpi=120, bbox_inches="tight")
    plt.close(fig); print(f"  Saved: gaussian_pdf_cdf.png")

def demo_moments():
    print("\n=== Moments ===")
    np.random.seed(1)
    samples = np.random.exponential(scale=2.0, size=100_000)
    mean  = np.mean(samples)
    var   = np.var(samples)
    skew  = np.mean(((samples - mean) / np.std(samples))**3)
    kurt  = np.mean(((samples - mean) / np.std(samples))**4) - 3
    print(f"  Exponential(scale=2):")
    print(f"    E[X]   = {mean:.4f}  (theory: 2.0)")
    print(f"    Var[X] = {var:.4f}   (theory: 4.0)")
    print(f"    Skew   = {skew:.4f}  (theory: 2.0)")
    print(f"    Ex.Kurt= {kurt:.4f}  (theory: 6.0)")

if __name__ == "__main__":
    demo_discrete()
    demo_continuous()
    demo_moments()
