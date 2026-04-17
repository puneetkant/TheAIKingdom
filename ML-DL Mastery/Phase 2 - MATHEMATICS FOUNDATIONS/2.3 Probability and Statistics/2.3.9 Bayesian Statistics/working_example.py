"""
Working Example: Bayesian Statistics
Covers Bayes' theorem, prior/posterior, conjugate distributions,
Bayesian inference, MCMC (Metropolis-Hastings), and credible intervals.
"""
import numpy as np
from scipy import stats


# ── 1. Bayes' theorem in action ───────────────────────────────────────────────
def bayes_theorem_classic():
    print("=== Bayes' Theorem: P(H|D) ∝ P(D|H)·P(H) ===")
    # Coin flipping: is the coin fair?
    # H₀: p=0.5 (fair), H₁: p=0.8 (biased)
    prior = {0.5: 0.8, 0.8: 0.2}   # P(H)
    # Observe 7 heads in 10 flips
    k, n = 7, 10
    likelihood = {p: stats.binom.pmf(k, n, p) for p in prior}

    # Unnormalised posterior
    unnorm = {p: likelihood[p] * prior[p] for p in prior}
    total  = sum(unnorm.values())
    posterior = {p: v/total for p, v in unnorm.items()}

    print(f"  Observed: {k} heads in {n} flips")
    for p in [0.5, 0.8]:
        print(f"\n  H: coin has p={p}")
        print(f"    Prior     P(H):    {prior[p]:.4f}")
        print(f"    Likelihood P(D|H): {likelihood[p]:.4f}")
        print(f"    Posterior P(H|D):  {posterior[p]:.4f}")


# ── 2. Beta-Binomial conjugate model ─────────────────────────────────────────
def beta_binomial():
    print("\n=== Beta-Binomial Conjugate Model ===")
    print("  Prior: p ~ Beta(α,β)  →  Posterior: p | data ~ Beta(α+k, β+n-k)")

    alpha0, beta0 = 2.0, 2.0   # prior belief: slightly above 0.5
    rng = np.random.default_rng(0)
    p_true = 0.65

    print(f"  Prior: Beta({alpha0},{beta0})  mean={alpha0/(alpha0+beta0):.4f}")
    print(f"\n  {'Flips':<8} {'Heads':<8} {'Post α':<10} {'Post β':<10} {'Post mean':<12} {'95% CI'}")

    alpha, beta = alpha0, beta0
    total_n, total_k = 0, 0
    for batch in [5, 15, 30, 50, 100]:
        new_flips = rng.binomial(batch, p_true)
        total_n  += batch
        total_k  += new_flips
        alpha    += new_flips
        beta     += batch - new_flips
        post     = stats.beta(alpha, beta)
        lo, hi   = post.ppf(0.025), post.ppf(0.975)
        print(f"  {total_n:<8} {total_k:<8} {alpha:<10.1f} {beta:<10.1f} {post.mean():<12.4f} [{lo:.4f},{hi:.4f}]")

    print(f"\n  True p = {p_true}")


# ── 3. Normal-Normal conjugate model ─────────────────────────────────────────
def normal_normal():
    print("\n=== Normal-Normal Conjugate (unknown μ, known σ) ===")
    # Prior: μ ~ N(μ₀, τ₀²)
    # Likelihood: X | μ ~ N(μ, σ²)
    # Posterior: μ | x ~ N(μₙ, τₙ²)
    mu0, tau0 = 0.0, 10.0     # weak prior
    sigma     = 3.0            # known likelihood std
    mu_true   = 7.0

    rng = np.random.default_rng(1)
    data = rng.normal(mu_true, sigma, 50)

    # Posterior parameters
    n    = len(data)
    xbar = data.mean()
    tau_n_sq = 1 / (1/tau0**2 + n/sigma**2)
    mu_n     = tau_n_sq * (mu0/tau0**2 + n*xbar/sigma**2)

    print(f"  Prior: μ ~ N({mu0},{tau0}²)")
    print(f"  Data: n={n}  x̄={xbar:.4f}  σ_known={sigma}")
    print(f"  Posterior: μ|data ~ N({mu_n:.4f},{tau_n_sq:.4f})")
    lo = mu_n - 1.96*np.sqrt(tau_n_sq)
    hi = mu_n + 1.96*np.sqrt(tau_n_sq)
    print(f"  95% Credible Interval: [{lo:.4f}, {hi:.4f}]")
    print(f"  True μ = {mu_true}  inside CI? {lo <= mu_true <= hi}")

    # Posterior precision = prior precision + data precision
    print(f"\n  Posterior precision = prior prec + data prec")
    print(f"    {1/tau_n_sq:.4f} = {1/tau0**2:.4f} + {n/sigma**2:.4f}")


# ── 4. Posterior predictive distribution ─────────────────────────────────────
def posterior_predictive():
    print("\n=== Posterior Predictive Distribution ===")
    # Beta-Binomial: P(X̃=k | data) = ∫ Binom(k|p)·Beta_post(p) dp
    alpha_post, beta_post = 12, 8   # after seeing 10 heads, 6 tails with Beta(2,2) prior
    n_pred = 10   # predict next 10 flips

    from scipy.special import comb as scomb, betaln

    def predictive_pmf(k, n, a, b):
        """P(X̃=k | n trials, posterior Beta(a,b))"""
        log_p = (np.log(scomb(n, k, exact=True)) +
                 betaln(k+a, n-k+b) -
                 betaln(a, b))
        return np.exp(log_p)

    print(f"  Posterior: Beta({alpha_post},{beta_post})  p_mean={alpha_post/(alpha_post+beta_post):.4f}")
    print(f"  Predicting next {n_pred} flips:")
    total = 0.0
    for k in range(n_pred+1):
        p = predictive_pmf(k, n_pred, alpha_post, beta_post)
        total += p
        print(f"    P(X̃={k:2d}) = {p:.4f}")
    print(f"  Sum = {total:.6f}")


# ── 5. Metropolis-Hastings MCMC ───────────────────────────────────────────────
def metropolis_hastings():
    print("\n=== Metropolis-Hastings MCMC ===")
    # Target: Beta(3,5) distribution sampled via MH
    target_log_prob = lambda x: stats.beta(3, 5).logpdf(x) if 0 < x < 1 else -np.inf

    rng     = np.random.default_rng(2)
    n_iter  = 20_000
    burn_in = 2_000
    theta   = 0.5   # starting point
    proposal_std = 0.15
    accepted = 0
    samples  = []

    for i in range(n_iter):
        proposal = theta + rng.normal(0, proposal_std)
        log_alpha = target_log_prob(proposal) - target_log_prob(theta)
        if np.log(rng.uniform()) < log_alpha:
            theta = proposal
            accepted += 1
        samples.append(theta)

    samples    = np.array(samples[burn_in:])
    true_dist  = stats.beta(3, 5)
    print(f"  Target: Beta(3,5)  mean={true_dist.mean():.4f}  std={true_dist.std():.4f}")
    print(f"  MCMC:  n_iter={n_iter}  burn_in={burn_in}  proposal_std={proposal_std}")
    print(f"  Acceptance rate: {accepted/n_iter:.4f}  (target 20-50%)")
    print(f"  Sample mean: {samples.mean():.4f}  Sample std: {samples.std():.4f}")
    lo, hi = np.percentile(samples, [2.5, 97.5])
    print(f"  95% Credible Interval: [{lo:.4f}, {hi:.4f}]")
    print(f"  True 95% CI: [{true_dist.ppf(0.025):.4f}, {true_dist.ppf(0.975):.4f}]")


# ── 6. Bayesian vs Frequentist comparison ────────────────────────────────────
def bayes_vs_frequentist():
    print("\n=== Bayesian vs Frequentist Comparison ===")
    comparison = [
        ("Question",       "What is P(H₀|data)?",        "What is P(data|H₀)?"),
        ("Parameters",     "Random variables with priors","Fixed unknown constants"),
        ("Inference",      "Posterior distribution",      "Point estimates + CIs"),
        ("Interval",       "Credible interval (direct)",  "Confidence interval (indirect)"),
        ("Prior info",     "Explicitly incorporated",     "Not used"),
        ("Small samples",  "Works with informative prior","May fail or be imprecise"),
        ("Computationally","MCMC needed (expensive)",     "Often analytic formulas"),
    ]
    print(f"  {'Aspect':<22} {'Bayesian':<35} {'Frequentist'}")
    print("  " + "-"*85)
    for aspect, bay, freq in comparison:
        print(f"  {aspect:<22} {bay:<35} {freq}")


if __name__ == "__main__":
    bayes_theorem_classic()
    beta_binomial()
    normal_normal()
    posterior_predictive()
    metropolis_hastings()
    bayes_vs_frequentist()
