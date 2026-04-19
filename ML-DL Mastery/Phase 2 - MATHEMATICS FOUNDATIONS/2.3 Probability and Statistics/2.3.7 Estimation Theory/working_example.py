"""
Working Example: Estimation Theory
Covers point estimation, MLE, MAP, unbiasedness, consistency,
efficiency (Cramér-Rao), confidence intervals, and bootstrap.
"""
import numpy as np
from scipy import stats, optimize


# -- 1. Point estimators -------------------------------------------------------
def point_estimators():
    print("=== Point Estimators ===")
    rng  = np.random.default_rng(0)
    data = rng.normal(loc=5.0, scale=2.0, size=100)

    estimators = {
        "Mean (X)":          data.mean(),
        "Median":             np.median(data),
        "Trimmed mean (5%)":  stats.trim_mean(data, 0.05),
        "Std (MLE, ddof=0)":  data.std(ddof=0),
        "Std (unbiased ddof=1)": data.std(ddof=1),
        "Var (MLE)":          data.var(ddof=0),
        "Var (unbiased)":     data.var(ddof=1),
    }
    print(f"  X ~ N(5,4)  n={len(data)}")
    for name, val in estimators.items():
        print(f"  {name:<28}: {val:.4f}")


# -- 2. Maximum Likelihood Estimation -----------------------------------------
def maximum_likelihood():
    print("\n=== Maximum Likelihood Estimation (MLE) ===")
    rng = np.random.default_rng(1)

    # Example 1: Poisson(lambda) — MLE is sample mean
    lam_true = 4.5
    data_p   = rng.poisson(lam_true, 200)
    lam_mle  = data_p.mean()   # analytic MLE
    print(f"  Poisson: lambda_true={lam_true}  lambda_MLE={lam_mle:.4f}  (sample mean)")

    # Example 2: Normal — analytic MLE
    mu_true, sig_true = 3.0, 1.5
    data_n  = rng.normal(mu_true, sig_true, 500)
    mu_mle  = data_n.mean()
    sig_mle = data_n.std(ddof=0)   # MLE (biased)
    sig_unb = data_n.std(ddof=1)   # Unbiased
    print(f"\n  Normal: mu_true={mu_true}  mu_MLE={mu_mle:.4f}")
    print(f"          sigma_true={sig_true}  sigma_MLE={sig_mle:.4f}  sigma_unbiased={sig_unb:.4f}")

    # Example 3: Numeric MLE via log-likelihood maximisation
    # X ~ Weibull(shape=k, scale=lambda) — no closed form
    k_true, lam_w = 2.0, 3.0
    data_w = rng.weibull(k_true, 300) * lam_w

    def neg_log_lik(params, data):
        k, lam = params
        if k <= 0 or lam <= 0:
            return 1e10
        return -np.sum(stats.weibull_min.logpdf(data, c=k, scale=lam))

    result = optimize.minimize(neg_log_lik, [1.5, 2.5], args=(data_w,), method='Nelder-Mead')
    k_hat, lam_hat = result.x
    print(f"\n  Weibull numeric MLE: k_true={k_true}  k={k_hat:.4f}  lambda_true={lam_w}  lambda={lam_hat:.4f}")


# -- 3. Bias, variance, MSE ----------------------------------------------------
def bias_variance_mse():
    print("\n=== Bias, Variance, MSE ===")
    rng = np.random.default_rng(2)
    mu, sigma = 5.0, 2.0
    n, M = 30, 10_000

    mu_hats     = np.array([rng.normal(mu, sigma, n).mean()    for _ in range(M)])
    sigma_mle   = np.array([rng.normal(mu, sigma, n).std(ddof=0) for _ in range(M)])
    sigma_unb   = np.array([rng.normal(mu, sigma, n).std(ddof=1) for _ in range(M)])

    print(f"  True mu={mu}, sigma={sigma}, n={n}, M={M} simulations")
    for name, est, true_val in [
        ("X (mean)",       mu_hats,   mu),
        ("sigma_MLE (ddof=0)",  sigma_mle, sigma),
        ("sigma_unb (ddof=1)",  sigma_unb, sigma),
    ]:
        bias = est.mean() - true_val
        var  = est.var()
        mse  = np.mean((est - true_val)**2)
        print(f"\n  {name:<22}: E[theta]={est.mean():.4f}  bias={bias:.4f}  var={var:.4f}  MSE={mse:.4f}")
    print(f"\n  Note: MSE = bias² + variance")
    b = sigma_mle.mean() - sigma
    print(f"  sigma_MLE: bias²={b**2:.4f} + var={sigma_mle.var():.4f} = {b**2+sigma_mle.var():.4f} ~= MSE={np.mean((sigma_mle-sigma)**2):.4f}")


# -- 4. Cramér-Rao lower bound ------------------------------------------------
def cramer_rao():
    print("\n=== Cramér-Rao Lower Bound Var(theta) >= 1/I(theta) ===")
    # X ~ Poisson(lambda):  Fisher info I(lambda) = n/lambda  -> CRLB = lambda/n
    lam, n = 3.0, 50
    crlb   = lam / n
    print(f"  Poisson(lambda={lam}), n={n}:")
    print(f"  Fisher info I(lambda) = n/lambda = {n/lam:.4f}")
    print(f"  CRLB = 1/I(lambda) = lambda/n = {crlb:.4f}")
    print(f"  Var(lambda_MLE) = lambda/n = {crlb:.4f}  (MLE is efficient, achieves bound)")

    rng      = np.random.default_rng(5)
    M        = 50_000
    lam_hats = np.array([rng.poisson(lam, n).mean() for _ in range(M)])
    print(f"  Simulated Var(lambda) = {lam_hats.var():.6f}  (CRLB = {crlb:.6f})")


# -- 5. Confidence intervals ---------------------------------------------------
def confidence_intervals():
    print("\n=== Confidence Intervals ===")
    rng  = np.random.default_rng(6)
    mu, sigma = 10.0, 3.0
    n = 40
    data = rng.normal(mu, sigma, n)

    # 95% CI for mean (known sigma)
    z   = stats.norm.ppf(0.975)
    se  = sigma / np.sqrt(n)
    ci_z = (data.mean() - z*se, data.mean() + z*se)

    # 95% CI for mean (unknown sigma — use t-distribution)
    t   = stats.t.ppf(0.975, df=n-1)
    se_t = data.std(ddof=1) / np.sqrt(n)
    ci_t = (data.mean() - t*se_t, data.mean() + t*se_t)

    # scipy shorthand
    ci_scipy = stats.t.interval(0.95, df=n-1, loc=data.mean(), scale=stats.sem(data))

    print(f"  Data: n={n}  x={data.mean():.4f}  s={data.std(ddof=1):.4f}")
    print(f"  95% CI (known sigma):   ({ci_z[0]:.4f}, {ci_z[1]:.4f})")
    print(f"  95% CI (unknown sigma): ({ci_t[0]:.4f}, {ci_t[1]:.4f})")
    print(f"  scipy:              ({ci_scipy[0]:.4f}, {ci_scipy[1]:.4f})")
    print(f"  True mu={mu}  inside known sigma CI? {ci_z[0] <= mu <= ci_z[1]}")

    # Coverage simulation
    N_sims = 10_000
    covered = 0
    for _ in range(N_sims):
        d = rng.normal(mu, sigma, n)
        lo, hi = stats.t.interval(0.95, df=n-1, loc=d.mean(), scale=stats.sem(d))
        if lo <= mu <= hi:
            covered += 1
    print(f"\n  Coverage simulation ({N_sims} datasets): {covered/N_sims:.4f}  (target 0.95)")


# -- 6. Bootstrap -------------------------------------------------------------
def bootstrap():
    print("\n=== Bootstrap Confidence Interval ===")
    rng  = np.random.default_rng(7)
    data = rng.exponential(scale=2.0, size=50)   # true median = 2·ln2 ~= 1.386

    true_median = 2 * np.log(2)
    B = 5000

    boot_medians = np.array([np.median(rng.choice(data, len(data), replace=True))
                              for _ in range(B)])

    ci_lo, ci_hi = np.percentile(boot_medians, [2.5, 97.5])
    print(f"  Data: n=50  sample median={np.median(data):.4f}  true median={true_median:.4f}")
    print(f"  Bootstrap (B={B}):  95% CI = ({ci_lo:.4f}, {ci_hi:.4f})")
    print(f"  True median inside CI? {ci_lo <= true_median <= ci_hi}")


if __name__ == "__main__":
    point_estimators()
    maximum_likelihood()
    bias_variance_mse()
    cramer_rao()
    confidence_intervals()
    bootstrap()
