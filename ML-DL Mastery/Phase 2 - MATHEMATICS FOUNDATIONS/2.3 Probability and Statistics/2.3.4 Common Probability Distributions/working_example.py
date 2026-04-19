"""
Working Example: Common Probability Distributions
Covers Bernoulli, Binomial, Poisson, Geometric, Normal, Exponential,
Gamma, Beta, and Chi-squared distributions with plots.
"""
import numpy as np
from scipy import stats
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_dists")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -- Helper: print distribution summary ---------------------------------------
def dist_summary(name, rv, pmf_or_pdf="pdf"):
    print(f"\n  {name}:")
    print(f"    mean={rv.mean():.4f}  var={rv.var():.4f}  std={rv.std():.4f}")
    print(f"    P(X<=mean) ~= {rv.cdf(rv.mean()):.4f}  (50th pct={rv.ppf(0.5):.4f})")


# -- 1. Discrete distributions ------------------------------------------------
def discrete_distributions():
    print("=== Discrete Distributions ===")
    dists = {
        "Bernoulli(p=0.3)":    stats.bernoulli(p=0.3),
        "Binomial(n=10,p=0.4)":stats.binom(n=10, p=0.4),
        "Poisson(lambda=3)":        stats.poisson(mu=3),
        "Geometric(p=0.25)":   stats.geom(p=0.25),
        "NegBinom(r=5,p=0.4)": stats.nbinom(n=5, p=0.4),
        "HyperGeom(M=20,n=7,N=5)": stats.hypergeom(M=20, n=7, N=5),
    }
    for name, rv in dists.items():
        dist_summary(name, rv)

    # PMF plots for Binomial and Poisson
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (rv, label, color) in zip(axes, [
        (stats.binom(n=10, p=0.4), "Binomial(10,0.4)", "steelblue"),
        (stats.poisson(mu=4),      "Poisson(lambda=4)",      "seagreen"),
    ]):
        xs = np.arange(0, 15)
        ax.bar(xs, rv.pmf(xs), color=color, alpha=0.8, edgecolor='white')
        ax.set(title=label, xlabel="k", ylabel="P(X=k)")
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "discrete_pmfs.png")
    plt.savefig(path, dpi=90); plt.close()
    print(f"\n  Saved: {path}")


# -- 2. Continuous distributions -----------------------------------------------
def continuous_distributions():
    print("\n=== Continuous Distributions ===")
    dists = {
        "Normal(mu=0,sigma=1)":      stats.norm(0, 1),
        "Normal(mu=2,sigma=1.5)":    stats.norm(2, 1.5),
        "Exponential(lambda=2)":      stats.expon(scale=1/2),
        "Uniform(0,1)":          stats.uniform(0, 1),
        "Gamma(alpha=3,beta=2)":        stats.gamma(a=3, scale=2),
        "Beta(alpha=2,beta=5)":         stats.beta(a=2, b=5),
        "LogNormal(mu=0,sigma=0.5)":  stats.lognorm(s=0.5),
        "Student-t(df=5)":       stats.t(df=5),
        "Chi²(df=4)":            stats.chi2(df=4),
        "F(d1=3,d2=10)":         stats.f(dfn=3, dfd=10),
    }
    for name, rv in dists.items():
        dist_summary(name, rv)

    # PDF plots
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    plot_specs = [
        (stats.norm(0,1),          "Normal(0,1)",         np.linspace(-4,4,300)),
        (stats.expon(scale=0.5),   "Exponential(lambda=2)",    np.linspace(0,5,300)),
        (stats.gamma(a=3,scale=2), "Gamma(alpha=3,beta=2)",      np.linspace(0,20,300)),
        (stats.beta(a=2,b=5),      "Beta(alpha=2,beta=5)",       np.linspace(0,1,300)),
        (stats.chi2(df=4),         "Chi²(df=4)",           np.linspace(0,15,300)),
        (stats.t(df=5),            "Student-t(df=5)",      np.linspace(-5,5,300)),
    ]
    for ax, (rv, label, xs) in zip(axes.flat, plot_specs):
        ax.plot(xs, rv.pdf(xs), lw=2)
        ax.fill_between(xs, rv.pdf(xs), alpha=0.2)
        ax.set(title=label, xlabel="x", ylabel="PDF")
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "continuous_pdfs.png")
    plt.savefig(path, dpi=90); plt.close()
    print(f"\n  Saved: {path}")


# -- 3. Distribution properties ------------------------------------------------
def distribution_properties():
    print("\n=== Key Properties ===")

    # Normal: 68-95-99.7 rule
    rv = stats.norm(0, 1)
    print("  Normal 68-95-99.7 rule:")
    for k in [1, 2, 3]:
        p = rv.cdf(k) - rv.cdf(-k)
        print(f"    P(-{k} <= Z <= {k}) = {p:.4f}")

    # Exponential memorylessness: P(X>s+t | X>s) = P(X>t)
    rv_e = stats.expon(scale=2)   # rate lambda=0.5
    s, t = 3, 2
    p1 = (1 - rv_e.cdf(s+t)) / (1 - rv_e.cdf(s))
    p2 = 1 - rv_e.cdf(t)
    print(f"\n  Exponential memorylessness: P(X>{s+t}|X>{s})={p1:.6f}  P(X>{t})={p2:.6f}  match={np.isclose(p1,p2)}")

    # Poisson as limit of Binomial
    n, p_binom = 1000, 0.003
    lam = n * p_binom
    print(f"\n  Poisson approximation: Bin(n={n},p={p_binom}) ~= Poisson(lambda={lam})")
    for k in range(6):
        pb = stats.binom.pmf(k, n, p_binom)
        pp = stats.poisson.pmf(k, lam)
        print(f"    P(X={k}): Binom={pb:.6f}  Poisson={pp:.6f}  diff={abs(pb-pp):.2e}")


# -- 4. Parameter estimation (MLE) --------------------------------------------
def mle_estimation():
    print("\n=== MLE Parameter Estimation ===")
    rng = np.random.default_rng(7)

    # Normal data
    mu_true, sigma_true = 5.0, 2.0
    data = rng.normal(mu_true, sigma_true, 500)
    mu_hat   = data.mean()
    sig_hat  = data.std(ddof=0)    # MLE (biased), ddof=1 is unbiased
    print(f"  Normal MLE: mu_true={mu_true}  mu={mu_hat:.4f}  sigma_true={sigma_true}  sigma_MLE={sig_hat:.4f}")

    # Exponential data
    lam_true = 3.0
    data_exp = rng.exponential(1/lam_true, 500)
    lam_hat  = 1 / data_exp.mean()   # MLE for exponential
    print(f"  Exponential MLE: lambda_true={lam_true}  lambda={lam_hat:.4f}")

    # Scipy fit
    mu_fit, sig_fit = stats.norm.fit(data)
    print(f"  scipy fit Normal: mu={mu_fit:.4f}  sigma={sig_fit:.4f}")


# -- 5. CDF and quantile (ppf) ------------------------------------------------
def cdf_and_quantiles():
    print("\n=== CDF and Quantile Function ===")
    rv = stats.norm(0, 1)
    print("  Standard Normal CDF values:")
    for z in [-2, -1, 0, 1, 1.645, 1.96, 2, 2.576]:
        print(f"    Φ({z:6.3f}) = {rv.cdf(z):.4f}  <- z={z}")
    print("\n  Quantiles (ppf = percent-point function = Φ^-1):")
    for p in [0.025, 0.05, 0.10, 0.50, 0.90, 0.95, 0.975, 0.995]:
        print(f"    Φ^-1({p:.3f}) = {rv.ppf(p):.4f}")


if __name__ == "__main__":
    discrete_distributions()
    continuous_distributions()
    distribution_properties()
    mle_estimation()
    cdf_and_quantiles()
