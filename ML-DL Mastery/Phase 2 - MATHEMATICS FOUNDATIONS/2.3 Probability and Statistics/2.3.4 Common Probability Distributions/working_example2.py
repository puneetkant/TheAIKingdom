"""
Working Example 2: Common Probability Distributions — ML Distributions Gallery
===============================================================================
Gaussian, Bernoulli, Categorical, Beta, Dirichlet, Laplace, Student-t, Chi²,
Log-Normal — PDF plots, sampling, and ML applications.

Run:  python working_example2.py
"""
import math
from pathlib import Path
try:
    import numpy as np
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit("pip install numpy matplotlib")

OUTPUT = Path(__file__).parent / "output"
OUTPUT.mkdir(exist_ok=True)

def demo_ml_distributions():
    print("=== ML-Relevant Distributions ===")
    np.random.seed(42)
    fig, axes = plt.subplots(3, 3, figsize=(12, 9))
    axes = axes.flat
    x = np.linspace(-5, 5, 300)

    # 1. Gaussian
    for mu, sig in [(0, 1), (0, 2), (2, 0.5)]:
        y = np.exp(-0.5*((x-mu)/sig)**2) / (sig * math.sqrt(2*math.pi))
        axes[0].plot(x, y, label=f"mu={mu},sigma={sig}")
    axes[0].set_title("Gaussian N(mu,sigma²)"); axes[0].legend(fontsize=7)

    # 2. Log-Normal
    xs = np.linspace(0.01, 6, 300)
    for mu, sig in [(0, 0.5), (0.5, 0.5), (0, 1)]:
        y = np.exp(-0.5*((np.log(xs)-mu)/sig)**2) / (xs * sig * math.sqrt(2*math.pi))
        axes[1].plot(xs, y, label=f"mu={mu},sigma={sig}")
    axes[1].set_title("Log-Normal"); axes[1].legend(fontsize=7)

    # 3. Laplace (double-exponential) — used for L1 regularisation
    for b in [0.5, 1, 2]:
        y = np.exp(-np.abs(x) / b) / (2*b)
        axes[2].plot(x, y, label=f"b={b}")
    axes[2].set_title("Laplace (-> L1 reg)"); axes[2].legend(fontsize=7)

    # 4. Beta — conjugate prior for Bernoulli
    xb = np.linspace(0.001, 0.999, 300)
    import math as m
    def beta_pdf(x, a, b):
        B = math.lgamma(a) + math.lgamma(b) - math.lgamma(a+b)
        return np.exp((a-1)*np.log(x) + (b-1)*np.log(1-x) - B)
    for a, b in [(2,2), (5,2), (0.5,0.5)]:
        axes[3].plot(xb, beta_pdf(xb,a,b), label=f"alpha={a},beta={b}")
    axes[3].set_title("Beta (prior for p)"); axes[3].legend(fontsize=7)

    # 5. Binomial PMF
    n, p = 20, 0.4
    ks = np.arange(0, 21)
    pmf = np.array([math.comb(n,k)*(p**k)*(1-p)**(n-k) for k in ks])
    axes[4].bar(ks, pmf, color="steelblue", alpha=0.7)
    axes[4].set_title(f"Binomial(n={n},p={p})")

    # 6. Poisson PMF
    lam = 4
    ks = np.arange(0, 16)
    pmf = np.array([math.exp(-lam) * lam**k / math.factorial(k) for k in ks])
    axes[5].bar(ks, pmf, color="coral", alpha=0.7)
    axes[5].set_title(f"Poisson(lambda={lam})")

    # 7. Student-t (used for heavy tails)
    for df in [1, 2, 5, 30]:
        y = (1 + x**2/df)**(-0.5*(df+1)) * math.exp(math.lgamma((df+1)/2) - math.lgamma(df/2)) / math.sqrt(df*math.pi)
        axes[6].plot(x, y, label=f"nu={df}")
    axes[6].set_title("Student-t"); axes[6].legend(fontsize=7)

    # 8. Chi-squared
    xc = np.linspace(0.01, 15, 300)
    for k in [1,2,3,5]:
        log_y = (k/2-1)*np.log(xc) - xc/2 - k/2*math.log(2) - math.lgamma(k/2)
        axes[7].plot(xc, np.exp(log_y), label=f"k={k}")
    axes[7].set_title("Chi-squared"); axes[7].legend(fontsize=7)

    # 9. Dirichlet samples (k=3)
    D = np.random.dirichlet([2,2,2], size=500)
    axes[8].scatter(D[:,0], D[:,1], s=4, alpha=0.4, c="teal")
    axes[8].set_title("Dirichlet(2,2,2) — topic models")

    plt.tight_layout()
    fig.savefig(OUTPUT / "distributions_gallery.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: distributions_gallery.png")

def demo_distribution_moments():
    print("\n=== Moments Summary (simulated) ===")
    np.random.seed(0)
    dists = {
        "Gaussian(0,1)":    np.random.normal(0, 1, 100_000),
        "Laplace(0,1)":     np.random.laplace(0, 1, 100_000),
        "Exponential(1)":   np.random.exponential(1, 100_000),
        "Student-t(5)":     np.random.standard_t(5, 100_000),
    }
    print(f"  {'Distribution':20s}  {'Mean':>6}  {'Var':>6}  {'Skew':>6}  {'Kurt':>6}")
    for name, s in dists.items():
        m = np.mean(s); v = np.var(s); std = np.std(s)
        skew = np.mean(((s-m)/std)**3); kurt = np.mean(((s-m)/std)**4) - 3
        print(f"  {name:20s}  {m:>6.3f}  {v:>6.3f}  {skew:>6.3f}  {kurt:>6.3f}")

if __name__ == "__main__":
    demo_ml_distributions()
    demo_distribution_moments()
