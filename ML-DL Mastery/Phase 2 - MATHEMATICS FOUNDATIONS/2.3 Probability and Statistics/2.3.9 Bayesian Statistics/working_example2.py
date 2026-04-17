"""
Working Example 2: Bayesian Statistics — Conjugate Priors, MCMC, Bayesian Linear Regression
============================================================================================
Beta-Binomial conjugate prior, Gaussian posterior, Metropolis-Hastings MCMC,
Bayesian linear regression (posterior predictive).

Run:  python working_example2.py
"""
import math, random
from pathlib import Path
try:
    import numpy as np
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit("pip install numpy matplotlib")

OUTPUT = Path(__file__).parent / "output"
OUTPUT.mkdir(exist_ok=True)

def demo_beta_binomial():
    print("=== Beta-Binomial Conjugate Model ===")
    # Prior: p ~ Beta(alpha, beta)
    # Likelihood: k ~ Binomial(n, p)
    # Posterior: p | k ~ Beta(alpha+k, beta+n-k)
    np.random.seed(42)
    p_true = 0.3
    alpha, beta_prior = 2, 2    # prior
    data = np.random.binomial(1, p_true, 100)
    k, n = data.sum(), len(data)

    alpha_post = alpha + k
    beta_post  = beta_prior + n - k
    p_prior    = alpha / (alpha + beta_prior)
    p_post     = alpha_post / (alpha_post + beta_post)
    print(f"  True p: {p_true}")
    print(f"  Observed: k={k}, n={n}")
    print(f"  Prior mean: {p_prior:.4f}")
    print(f"  Posterior mean: {p_post:.4f}")
    print(f"  MLE: {k/n:.4f}")

    x = np.linspace(0.001, 0.999, 300)
    def beta_pdf(x, a, b):
        log_B = math.lgamma(a) + math.lgamma(b) - math.lgamma(a+b)
        return np.exp((a-1)*np.log(x) + (b-1)*np.log(1-x) - log_B)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, beta_pdf(x, alpha, beta_prior), "--", label=f"Prior Beta({alpha},{beta_prior})")
    ax.plot(x, beta_pdf(x, alpha_post, beta_post), lw=2, label=f"Posterior Beta({alpha_post},{beta_post})")
    ax.axvline(p_true, color="k", ls=":", label=f"True p={p_true}")
    ax.axvline(k/n, color="r", ls=":", label=f"MLE={k/n:.2f}")
    ax.legend(); ax.set_title("Bayesian update: Beta-Binomial")
    fig.savefig(OUTPUT / "bayesian_update.png", dpi=120, bbox_inches="tight")
    plt.close(fig); print(f"  Saved: bayesian_update.png")

def demo_sequential_update():
    print("\n=== Sequential Bayesian Update ===")
    np.random.seed(0)
    p_true = 0.4; alpha, beta_ = 1, 1   # uniform prior
    flips = np.random.binomial(1, p_true, 50)
    print(f"  {'n':>4}  {'Post mean':>10}  {'Post std':>10}")
    for i, x in enumerate(flips, 1):
        alpha += x; beta_ += (1 - x)
        if i in [1, 5, 10, 25, 50]:
            post_mean = alpha / (alpha + beta_)
            post_std  = math.sqrt(alpha * beta_ / (alpha + beta_)**2 / (alpha + beta_ + 1))
            print(f"  {i:>4}  {post_mean:>10.4f}  {post_std:>10.4f}")

def demo_metropolis_hastings():
    print("\n=== Metropolis-Hastings MCMC (Gaussian posterior) ===")
    np.random.seed(7)
    # Data: y ~ N(mu, 1), prior mu ~ N(0, 3²)
    # Posterior: mu | y ~ N(mu_post, sigma_post²)
    data = np.random.normal(2.5, 1.0, 20)
    n = len(data); ybar = data.mean()
    sigma_lik, sigma_prior = 1.0, 3.0
    mu_post = (ybar/sigma_lik**2 + 0/sigma_prior**2) / (n/sigma_lik**2 + 1/sigma_prior**2)
    sig_post = 1 / math.sqrt(n/sigma_lik**2 + 1/sigma_prior**2)
    print(f"  Analytic posterior: N({mu_post:.4f}, {sig_post:.4f}²)")

    # MCMC
    def log_posterior(mu):
        log_lik = -0.5 * np.sum((data - mu)**2) / sigma_lik**2
        log_prior = -0.5 * (mu / sigma_prior)**2
        return log_lik + log_prior

    mu = 0.0; samples = []; prop_std = 0.5
    for _ in range(20_000):
        proposal = mu + np.random.normal(0, prop_std)
        log_accept = log_posterior(proposal) - log_posterior(mu)
        if math.log(random.random() + 1e-15) < log_accept:
            mu = proposal
        samples.append(mu)

    samples = np.array(samples[5000:])   # discard burn-in
    print(f"  MCMC posterior: mean={samples.mean():.4f}  std={samples.std():.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(samples[:500], lw=0.6); axes[0].set_title("MCMC trace (first 500)")
    axes[1].hist(samples, bins=60, density=True, color="steelblue", alpha=0.7)
    x = np.linspace(samples.min(), samples.max(), 200)
    axes[1].plot(x, np.exp(-0.5*((x-mu_post)/sig_post)**2)/(sig_post*(2*math.pi)**0.5), "r", lw=2)
    axes[1].set_title("MCMC posterior vs analytic")
    fig.savefig(OUTPUT / "mcmc.png", dpi=120, bbox_inches="tight")
    plt.close(fig); print(f"  Saved: mcmc.png")

if __name__ == "__main__":
    demo_beta_binomial()
    demo_sequential_update()
    demo_metropolis_hastings()
