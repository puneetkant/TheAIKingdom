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
    import matplotlib
    matplotlib.use("Agg")
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

    # Poisson(lambda=3)
    lam = 3
    samples_p = np.random.poisson(lam, size=10_000)
    print(f"  Poisson(lambda={lam}):")
    print(f"    E[X] = {np.mean(samples_p):.4f}  Var[X]= {np.var(samples_p):.4f}  (both ~= {lam})")

def demo_continuous():
    print("\n=== Continuous Random Variables ===")
    np.random.seed(0)

    # Gaussian
    mu, sigma = 2.0, 1.5
    samples = np.random.normal(mu, sigma, size=50_000)
    print(f"  Gaussian(mu={mu}, sigma={sigma}):")
    print(f"    E[X] = {np.mean(samples):.4f}  std = {np.std(samples):.4f}")

    # CDF by simulation: P(X <= t)
    for t in [0.0, mu - sigma, mu, mu + sigma]:
        p = np.mean(samples <= t)
        print(f"    P(X <= {t:.1f}) ~= {p:.4f}")

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

def demo_mgf_moments():
    print("\n=== Moment Generating Function (MGF) ===")
    # For X ~ N(mu, sigma^2): M_X(t) = exp(mu*t + sigma^2*t^2/2)
    np.random.seed(0)
    mu, sigma = 1.5, 2.0
    samples = np.random.normal(mu, sigma, size=500_000)
    print(f"  X ~ N(mu={mu}, sigma={sigma})")
    print(f"  {'t':>6}  {'E[e^tX] (MC)':>14}  {'Analytic MGF':>14}")
    for t in [-0.5, 0.0, 0.2, 0.4]:
        mc_val   = float(np.mean(np.exp(t * samples)))
        analytic = math.exp(mu * t + 0.5 * sigma**2 * t**2)
        print(f"  {t:>6.2f}  {mc_val:>14.6f}  {analytic:>14.6f}")
    # E[X^2] = sigma^2 + mu^2  (second moment from MGF second derivative at 0)
    E_X2_theory = sigma**2 + mu**2
    E_X2_sample = float(np.mean(samples**2))
    print(f"  E[X^2] theory={E_X2_theory:.4f},  sample={E_X2_sample:.4f}")


def demo_chebyshev():
    print("\n=== Chebyshev's Inequality Verification ===")
    # P(|X - mu| >= k*sigma) <= 1/k^2
    np.random.seed(5)
    scale = 2.0
    samples = np.random.exponential(scale, size=1_000_000)
    mu    = float(np.mean(samples))
    sigma = float(np.std(samples))
    print(f"  Exponential(scale={scale}): mu={mu:.4f}, sigma={sigma:.4f}")
    print(f"  {'k':>4}  {'Chebyshev bound 1/k^2':>22}  {'Empirical P':>13}")
    for k in [1.5, 2.0, 3.0, 4.0]:
        bound     = 1.0 / k**2
        empirical = float(np.mean(np.abs(samples - mu) >= k * sigma))
        ok = "OK" if empirical <= bound + 1e-9 else "FAIL"
        print(f"  {k:>4.1f}  {bound:>22.6f}  {empirical:>13.6f}  {ok}")


def demo_covariance_matrix():
    print("\n=== 2D Covariance Matrix and Ellipse ===")
    np.random.seed(42)
    mu_vec = np.array([1.0, 2.0])
    cov    = np.array([[4.0, 1.8], [1.8, 1.0]])
    L      = np.linalg.cholesky(cov)
    z       = np.random.randn(2000, 2)
    samples = z @ L.T + mu_vec
    cov_sample = np.cov(samples.T)
    print(f"  True cov:\n{cov}")
    print(f"  Sample cov:\n{cov_sample.round(4)}")
    vals, vecs = np.linalg.eigh(cov)
    print(f"  Eigenvalues (variance along principal axes): {vals.round(4)}")
    # Save scatter + ellipse
    theta = np.linspace(0, 2 * math.pi, 300)
    ellipse_unit = np.column_stack([np.cos(theta), np.sin(theta)])
    ellipse = ellipse_unit @ (vecs * np.sqrt(vals)).T + mu_vec
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(samples[:, 0], samples[:, 1], alpha=0.05, s=5, color="steelblue")
    ax.plot(ellipse[:, 0], ellipse[:, 1], "r", lw=2, label="1-sigma ellipse")
    ax.set_title("2D Covariance Ellipse"); ax.legend(); ax.set_aspect("equal")
    fig.savefig(OUTPUT / "covariance_ellipse.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: covariance_ellipse.png")


def demo_transformation_rv():
    print("\n=== Transformation of RV: Y = X^2 (X ~ Uniform[0,1]) ===")
    # Analytic: F_Y(y) = sqrt(y),  f_Y(y) = 1/(2*sqrt(y))  for y in [0,1]
    np.random.seed(3)
    X = np.random.uniform(0, 1, size=200_000)
    Y = X ** 2
    print(f"  E[Y] theory=1/3={1/3:.4f},  sample={float(np.mean(Y)):.4f}")
    print(f"  Var[Y] theory=4/45={4/45:.4f},  sample={float(np.var(Y)):.4f}")
    print(f"  {'y':>5}  {'F_Y analytic':>14}  {'F_Y sample':>12}")
    for y in [0.1, 0.25, 0.5, 0.75, 0.9]:
        print(f"  {y:>5.2f}  {math.sqrt(y):>14.6f}  {float(np.mean(Y <= y)):>12.6f}")
    y_plot = np.linspace(0.001, 1.0, 300)
    pdf_Y  = 1.0 / (2 * np.sqrt(y_plot))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(Y, bins=80, density=True, alpha=0.6, color="coral")
    axes[0].plot(y_plot, pdf_Y, "k", lw=2, label="f_Y=1/(2*sqrt(y))")
    axes[0].set_title("PDF of Y = X^2"); axes[0].legend()
    axes[1].plot(y_plot, np.sqrt(y_plot), lw=2, color="purple", label="F_Y=sqrt(y)")
    axes[1].set_title("CDF of Y = X^2"); axes[1].legend()
    fig.savefig(OUTPUT / "rv_transformation.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: rv_transformation.png")


def demo_central_limit_theorem():
    print("\n=== Central Limit Theorem ===")
    # Sum of n iid RVs -> Normal, regardless of original distribution
    np.random.seed(99)
    n_vals = [1, 5, 20, 100]
    lam = 2.0   # Exponential(2): skewed, non-Gaussian
    fig, axes = plt.subplots(1, len(n_vals), figsize=(14, 3), sharey=False)
    for ax, n in zip(axes, n_vals):
        # Sample means of n Exponential RVs (10,000 means)
        raw     = np.random.exponential(lam, size=(10_000, n))
        means   = raw.mean(axis=1)
        # Theoretical: mean=lam, std=lam/sqrt(n)
        theo_std = lam / math.sqrt(n)
        x = np.linspace(means.min(), means.max(), 200)
        pdf = np.exp(-0.5 * ((x - lam) / theo_std)**2) / (theo_std * math.sqrt(2 * math.pi))
        ax.hist(means, bins=50, density=True, alpha=0.6, color="steelblue")
        ax.plot(x, pdf, "r", lw=2)
        ax.set_title(f"n={n}")
        ax.set_xlabel("Sample mean")
    axes[0].set_ylabel("Density")
    fig.suptitle("CLT: Exp(2) sample means converge to Normal")
    fig.tight_layout()
    fig.savefig(OUTPUT / "clt_convergence.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Showing CLT for Exp(lambda={lam}) sample means (n=1,5,20,100)")
    print(f"  Saved: clt_convergence.png")
    # Numeric check: skewness decreases as n grows
    print(f"  {'n':>5}  {'sample mean':>12}  {'sample std':>11}  {'theo std':>9}  {'skewness':>9}")
    for n in [1, 5, 20, 100]:
        raw   = np.random.exponential(lam, size=(50_000, n))
        means = raw.mean(axis=1)
        sk    = float(np.mean(((means - means.mean()) / means.std())**3))
        print(f"  {n:>5}  {float(np.mean(means)):>12.4f}  "
              f"{float(np.std(means)):>11.4f}  {lam/math.sqrt(n):>9.4f}  {sk:>9.4f}")


if __name__ == "__main__":
    demo_discrete()
    demo_continuous()
    demo_moments()
    demo_mgf_moments()
    demo_chebyshev()
    demo_covariance_matrix()
    demo_transformation_rv()
    demo_central_limit_theorem()
