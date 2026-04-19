"""
Working Example: Limit Theorems
Covers the Law of Large Numbers, Central Limit Theorem,
convergence in probability/distribution, and applications.
"""
import numpy as np
from scipy import stats
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_limits")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -- 1. Law of Large Numbers (WLLN) --------------------------------------------
def law_of_large_numbers():
    print("=== Law of Large Numbers ===")
    rng  = np.random.default_rng(0)
    N    = 10_000
    lam  = 3.0

    # Poisson, true mean = lambda
    X    = rng.poisson(lam, N)
    ns   = [10, 50, 100, 500, 1000, 5000, N]
    means = [X[:n].mean() for n in ns]

    print(f"  X ~ Poisson(lambda={lam}), true E[X] = {lam}")
    print(f"  {'n':<8} {'sample mean':<14} {'|error|'}")
    for n, m in zip(ns, means):
        print(f"  {n:<8} {m:<14.4f} {abs(m-lam):.4f}")

    # Plot convergence
    fig, ax = plt.subplots(figsize=(8,4))
    cumulative_means = np.cumsum(X) / np.arange(1, N+1)
    ax.plot(cumulative_means, label="Running mean", lw=1.2)
    ax.axhline(lam, color='red', linestyle='--', label=f"True mean={lam}")
    ax.set(xscale='log', xlabel="n (samples, log scale)", ylabel="Mean estimate",
           title="LLN: Running Mean of Poisson(3) samples")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "lln.png")
    plt.savefig(path, dpi=90); plt.close()
    print(f"\n  Saved: {path}")


# -- 2. Central Limit Theorem -------------------------------------------------
def central_limit_theorem():
    print("\n=== Central Limit Theorem ===")
    rng = np.random.default_rng(1)
    M   = 5000   # number of experiments

    dists = {
        "Uniform(0,1)":   (lambda n: rng.uniform(0,1,n),     0.5,    1/12),
        "Exponential(2)": (lambda n: rng.exponential(0.5,n), 0.5,    0.25),
        "Bernoulli(0.3)": (lambda n: rng.binomial(1,0.3,n),  0.3,    0.21),
    }
    ns   = [1, 5, 30, 100]
    fig, axes = plt.subplots(len(dists), len(ns), figsize=(14, 9))

    for row_i, (dist_name, (sampler, mu, var)) in enumerate(dists.items()):
        print(f"\n  {dist_name}: mu={mu}, sigma²={var}, sigma={np.sqrt(var):.4f}")
        sigma = np.sqrt(var)
        for col_i, n in enumerate(ns):
            # Sample mean of n draws, M times
            samples      = np.array([sampler(n).mean() for _ in range(M)])
            z_scaled     = (samples - mu) / (sigma / np.sqrt(n))
            ax = axes[row_i, col_i]
            ax.hist(z_scaled, bins=40, density=True, color='steelblue', alpha=0.7)
            xg = np.linspace(-4, 4, 200)
            ax.plot(xg, stats.norm.pdf(xg), 'r-', lw=1.5)
            ax.set(title=f"n={n}",
                   xlabel="Z" if row_i==len(dists)-1 else "",
                   ylabel=dist_name if col_i==0 else "")
            ax.set_xlim(-4, 4)
            ax.grid(True, alpha=0.2)
            # Kolmogorov-Smirnov test against N(0,1)
            ks_stat, ks_p = stats.kstest(z_scaled, 'norm')
            print(f"    n={n:4d}: Z={z_scaled.mean():.3f} Std={z_scaled.std():.3f}  KS-test p={ks_p:.4f}")

    fig.suptitle("CLT: Standardised Sample Mean Distribution", fontsize=12)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "clt.png")
    plt.savefig(path, dpi=90); plt.close()
    print(f"\n  Saved: {path}")


# -- 3. Convergence in probability --------------------------------------------
def convergence_in_probability():
    print("\n=== Convergence in Probability ===")
    print("  Xn ->ᵖ X if P(|Xn-X|>epsilon) -> 0 as n->inf for all epsilon>0")

    rng = np.random.default_rng(2)
    N_max = 10000
    eps   = 0.1

    # X_n = (1/n) Sigma Bern(0.5) -> E[Bern]=0.5 in probability
    probs = []
    for n in [10, 50, 200, 1000, 5000]:
        # estimate P(|X_n - 0.5| > eps) over 2000 trials
        trials = rng.binomial(n, 0.5, 2000) / n
        p_exceeds = (np.abs(trials - 0.5) > eps).mean()
        probs.append((n, p_exceeds))
        print(f"  n={n:5d}: P(|Xn - 0.5| > {eps}) ~= {p_exceeds:.4f}  (-> 0)")

    # Chebyshev bound: P(|Xn-mu|>epsilon) <= sigma²/(nepsilon²)
    mu, sigma2 = 0.5, 0.25
    print(f"\n  Chebyshev upper bound sigma²/(nepsilon²) = {sigma2}/(n·{eps**2}):")
    for n, _ in probs:
        bound = sigma2 / (n * eps**2)
        print(f"    n={n:5d}: <= {bound:.4f}")


# -- 4. Normal approximation to Binomial --------------------------------------
def normal_approx_binomial():
    print("\n=== Normal Approximation to Binomial ===")
    n, p = 50, 0.4
    mu, sigma = n*p, np.sqrt(n*p*(1-p))

    print(f"  X ~ Bin({n},{p})  mu={mu}  sigma={sigma:.4f}")
    print(f"  {'k':<6} {'Exact P(X=k)':<16} {'Normal approx':<16} {'Cont. corr.'}")
    rv_binom = stats.binom(n, p)
    rv_norm  = stats.norm(mu, sigma)

    for k in [10, 15, 20, 25, 30]:
        exact = rv_binom.pmf(k)
        # Without continuity correction
        norm_approx = rv_norm.pdf(k)
        # With continuity correction: P(k-0.5 < X < k+0.5)
        cont_corr   = rv_norm.cdf(k+0.5) - rv_norm.cdf(k-0.5)
        print(f"  {k:<6} {exact:<16.6f} {norm_approx:<16.6f} {cont_corr:.6f}")


# -- 5. Delta method -----------------------------------------------------------
def delta_method():
    print("\n=== Delta Method: sqrtn(g(X) - g(mu)) ->ᵈ N(0, g'(mu)²sigma²) ===")
    # X ~ Exp(lambda=2) -> X -> 1/lambda, g(x) = log(x)
    rng = np.random.default_rng(3)
    lam = 2.0
    mu  = 1/lam   # E[X]
    var = 1/lam**2   # Var[X]
    g   = np.log
    g_p = lambda x: 1/x   # g'(x)

    n     = 100
    M     = 10000
    means = np.array([rng.exponential(mu, n).mean() for _ in range(M)])
    g_means = g(means)

    delta_var = (g_p(mu))**2 * var   # g'(mu)² sigma²
    print(f"  X ~ Exp(lambda={lam}): mu={mu}, sigma²={var}")
    print(f"  g(x) = ln(x):  g(mu) = ln({mu}) = {g(mu):.4f}")
    print(f"  Delta method: sqrtn·(g(X)-g(mu)) -> N(0, {delta_var:.4f})")
    print(f"\n  n={n}, M={M} simulations:")
    print(f"  E[g(X)] ~= {g_means.mean():.4f}  (theory g(mu) = {g(mu):.4f})")
    scaled = np.sqrt(n) * (g_means - g(mu))
    print(f"  Var[sqrtn(g(X)-g(mu))] ~= {scaled.var():.4f}  (theory {delta_var:.4f})")


if __name__ == "__main__":
    law_of_large_numbers()
    central_limit_theorem()
    convergence_in_probability()
    normal_approx_binomial()
    delta_method()
