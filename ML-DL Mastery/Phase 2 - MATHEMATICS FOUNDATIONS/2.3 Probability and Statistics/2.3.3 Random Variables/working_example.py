"""
Working Example: Random Variables
Covers discrete/continuous RVs, PMFs/PDFs, CDFs, expectation,
variance, standard deviation, moments, and moment generating functions.
"""
import numpy as np
from scipy import stats


# ── 1. Discrete random variable ───────────────────────────────────────────────
def discrete_rv():
    print("=== Discrete Random Variable ===")
    # X = number of heads in 3 fair coin flips
    outcomes = {0: 1/8, 1: 3/8, 2: 3/8, 3: 1/8}   # PMF
    values   = np.array(list(outcomes.keys()))
    probs    = np.array(list(outcomes.values()))

    # Verify axioms
    print(f"  PMF: {dict(zip(values, probs))}")
    print(f"  Sum of PMF = {probs.sum():.2f}  (should be 1)")
    print(f"  P(X≥0) = {(probs[values>=0]).sum():.2f}  (should be 1, non-negative)")

    # Statistics
    E_X    = np.sum(values * probs)
    E_X2   = np.sum(values**2 * probs)
    Var_X  = E_X2 - E_X**2
    Std_X  = np.sqrt(Var_X)

    print(f"  E[X] = {E_X:.4f}")
    print(f"  Var[X] = E[X²] - (E[X])² = {E_X2:.4f} - {E_X**2:.4f} = {Var_X:.4f}")
    print(f"  Std[X] = {Std_X:.4f}")

    # CDF
    print(f"\n  CDF F(x):")
    cdf = 0
    for x, p in sorted(zip(values, probs)):
        cdf += p
        print(f"    F({x}) = {cdf:.4f}")


# ── 2. Continuous random variable ─────────────────────────────────────────────
def continuous_rv():
    print("\n=== Continuous Random Variable ===")
    # X ~ Uniform(a,b)
    a, b = 2, 5
    rv   = stats.uniform(loc=a, scale=b-a)

    print(f"  X ~ Uniform({a},{b})")
    print(f"  PDF f(x) = 1/(b-a) = {1/(b-a):.4f}  for x ∈ [{a},{b}]")
    print(f"  E[X]   = (a+b)/2 = {(a+b)/2:.4f}  scipy: {rv.mean():.4f}")
    print(f"  Var[X] = (b-a)²/12 = {(b-a)**2/12:.4f}  scipy: {rv.var():.4f}")

    # P(2.5 ≤ X ≤ 4)
    p = rv.cdf(4) - rv.cdf(2.5)
    print(f"  P(2.5 ≤ X ≤ 4) = {p:.4f}  (geometric: {(4-2.5)/(b-a):.4f})")

    # X ~ Normal(μ,σ)
    mu, sigma = 3, 1.5
    rv_n = stats.norm(loc=mu, scale=sigma)
    print(f"\n  X ~ N({mu},{sigma}²)")
    print(f"  P(μ-σ ≤ X ≤ μ+σ) = {rv_n.cdf(mu+sigma)-rv_n.cdf(mu-sigma):.4f}  (≈68%)")
    print(f"  P(μ-2σ ≤ X ≤ μ+2σ)= {rv_n.cdf(mu+2*sigma)-rv_n.cdf(mu-2*sigma):.4f}  (≈95%)")
    print(f"  99th percentile   = {rv_n.ppf(0.99):.4f}")


# ── 3. Expectation properties ─────────────────────────────────────────────────
def expectation_properties():
    print("\n=== Expectation Properties ===")
    rng = np.random.default_rng(0)
    X = rng.standard_normal(100_000)
    Y = 2*X + 3    # Y = 2X + 3

    print(f"  X ~ N(0,1):  E[X] ≈ {X.mean():.4f}  (true 0)")
    print(f"  Y = 2X + 3:  E[Y] ≈ {Y.mean():.4f}  (true 3)")
    print(f"  Linearity:   2·E[X]+3 = {2*X.mean()+3:.4f}")
    print(f"  Var[Y] = 4·Var[X]: {Y.var():.4f} vs {4*X.var():.4f}")

    # Jensen's inequality: E[f(X)] ≥ f(E[X]) for convex f
    Z = rng.exponential(scale=2, size=100_000)
    print(f"\n  Jensen's (f=exp, convex): E[exp(Z)] = {np.exp(Z).mean():.4f}  ≥  exp(E[Z]) = {np.exp(Z.mean()):.4f}")


# ── 4. Variance and moments ───────────────────────────────────────────────────
def moments():
    print("\n=== Moments ===")
    mu, sigma = 5, 2
    rv = stats.norm(mu, sigma)

    def moment(k, center=0):
        """E[(X-center)^k] via numerical integration"""
        from scipy.integrate import quad
        result, _ = quad(lambda x: (x-center)**k * rv.pdf(x), mu-8*sigma, mu+8*sigma)
        return result

    print(f"  X ~ N({mu},{sigma}²)")
    print(f"  {'Moment':<30} {'Value':<12} {'Theory'}")
    print(f"  {'Raw 1st  E[X]':<30} {moment(1):<12.4f} {mu:.4f}")
    print(f"  {'Raw 2nd  E[X²]':<30} {moment(2):<12.4f} {mu**2+sigma**2:.4f}")
    print(f"  {'Central 2nd  Var[X]':<30} {moment(2, mu):<12.4f} {sigma**2:.4f}")
    print(f"  {'Central 3rd  skewness×σ³':<30} {moment(3, mu):<12.4f} {0:.4f}  (symmetric)")
    print(f"  {'Central 4th  kurtosis term':<30} {moment(4, mu):<12.4f} {3*sigma**4:.4f}  (3σ⁴)")

    excess_kurtosis = moment(4, mu) / sigma**4 - 3
    print(f"\n  Excess kurtosis = {excess_kurtosis:.4f}  (Normal=0 by def)")


# ── 5. Transformations of RVs ─────────────────────────────────────────────────
def rv_transformations():
    print("\n=== Transformations of Random Variables ===")
    rng = np.random.default_rng(1)

    # Z ~ N(0,1) → X = σZ + μ ~ N(μ,σ²)
    Z  = rng.standard_normal(200_000)
    mu, sigma = 10, 3
    X  = sigma * Z + mu
    print(f"  Z~N(0,1) → X=σZ+μ:  E[X]={X.mean():.4f} (true {mu})  Std[X]={X.std():.4f} (true {sigma})")

    # U ~ Uniform(0,1) → inverse CDF → exponential
    U  = rng.uniform(0, 1, 200_000)
    lam = 2.0
    E  = -np.log(1 - U) / lam   # inverse CDF of Exp(λ)
    print(f"  U~Uniform → Exp(λ={lam}): E[X]={E.mean():.4f} (true {1/lam})")

    # Log-normal: Y = exp(X), X ~ N(μ,σ²)
    mu_n, sigma_n = 0, 1
    LN = np.exp(rng.normal(mu_n, sigma_n, 200_000))
    E_LN  = np.exp(mu_n + sigma_n**2/2)   # analytical
    print(f"  Log-Normal:  E[Y]={LN.mean():.4f} (theory {E_LN:.4f})")


# ── 6. Standardisation and the empirical rule ─────────────────────────────────
def standardisation():
    print("\n=== Standardisation Z = (X-μ)/σ ===")
    rng = np.random.default_rng(42)
    X   = rng.normal(loc=170, scale=10, size=10_000)   # heights in cm

    mu_hat    = X.mean()
    sigma_hat = X.std()
    Z         = (X - mu_hat) / sigma_hat

    print(f"  X ~ N(170,10²)  (sample of {len(X):,})")
    print(f"  μ̂={mu_hat:.2f}  σ̂={sigma_hat:.2f}")
    print(f"  Z: mean={Z.mean():.4f}  std={Z.std():.4f}")
    print(f"\n  Empirical rule for Z (should match 68/95/99.7%):")
    for k in [1, 2, 3]:
        pct = np.mean(np.abs(Z) <= k) * 100
        print(f"    |Z| ≤ {k}: {pct:.2f}%")


if __name__ == "__main__":
    discrete_rv()
    continuous_rv()
    expectation_properties()
    moments()
    rv_transformations()
    standardisation()
