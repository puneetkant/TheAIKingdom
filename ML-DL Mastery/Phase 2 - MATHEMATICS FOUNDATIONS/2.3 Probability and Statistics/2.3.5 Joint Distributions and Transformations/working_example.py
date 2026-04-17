"""
Working Example: Joint Distributions and Transformations
Covers joint PMF/PDF, marginals, conditional distributions, covariance,
correlation, independence, and variable transformations.
"""
import numpy as np
from scipy import stats, integrate


# ── 1. Joint discrete distribution ───────────────────────────────────────────
def joint_discrete():
    print("=== Joint Discrete Distribution ===")
    # Joint PMF: X = row index, Y = col index (each in {0,1,2})
    # P(X=i, Y=j) stored in a 3×3 table
    joint = np.array([
        [0.10, 0.08, 0.02],   # X=0
        [0.06, 0.14, 0.10],   # X=1
        [0.04, 0.18, 0.28],   # X=2
    ])
    assert np.isclose(joint.sum(), 1.0), "PMF must sum to 1"
    print(f"  Joint PMF table (X rows, Y cols):")
    print(f"  {joint}")

    # Marginals
    p_X = joint.sum(axis=1)   # sum over Y
    p_Y = joint.sum(axis=0)   # sum over X
    print(f"\n  Marginal P(X) = {p_X}")
    print(f"  Marginal P(Y) = {p_Y}")

    # Conditional P(Y|X=1)
    cond_Y_gX1 = joint[1, :] / p_X[1]
    print(f"\n  P(Y|X=1) = {np.round(cond_Y_gX1, 4)}")

    # Independence check: P(X,Y) == P(X)·P(Y) ?
    independent = np.allclose(joint, np.outer(p_X, p_Y))
    print(f"\n  X and Y independent? {independent}")

    # Covariance
    xs = np.array([0, 1, 2])
    EX  = np.sum(xs * p_X)
    EY  = np.sum(xs * p_Y)
    EXY = sum(i*j*joint[i,j] for i in range(3) for j in range(3))
    cov = EXY - EX * EY
    print(f"\n  E[X]={EX:.4f}  E[Y]={EY:.4f}  E[XY]={EXY:.4f}  Cov(X,Y)={cov:.4f}")


# ── 2. Joint continuous distribution ─────────────────────────────────────────
def joint_continuous():
    print("\n=== Joint Continuous Distribution ===")
    # f(x,y) = 6xy(x+y) on [0,1]×[0,1]  — verify it integrates to 1
    f = lambda x, y: 6 * x * y * (x + y)

    total, _ = integrate.dblquad(f, 0, 1, 0, 1)
    print(f"  f(x,y) = 6xy(x+y) on [0,1]²")
    print(f"  ∫∫ f(x,y) dx dy = {total:.6f}  (should be 1)")

    # Marginal of X: f_X(x) = ∫₀¹ f(x,y) dy
    f_X = lambda x: integrate.quad(lambda y: f(x, y), 0, 1)[0]
    xs  = np.linspace(0, 1, 10)
    print(f"\n  Marginal f_X(x) at x=0.5: {f_X(0.5):.4f}")

    # E[X]
    EX, _ = integrate.dblquad(lambda x, y: x * f(x, y), 0, 1, 0, 1)
    EY, _ = integrate.dblquad(lambda x, y: y * f(x, y), 0, 1, 0, 1)
    EXY,_ = integrate.dblquad(lambda x, y: x*y*f(x,y), 0, 1, 0, 1)
    EX2,_ = integrate.dblquad(lambda x, y: x**2*f(x,y), 0, 1, 0, 1)
    EY2,_ = integrate.dblquad(lambda x, y: y**2*f(x,y), 0, 1, 0, 1)
    cov   = EXY - EX*EY
    varX  = EX2 - EX**2
    varY  = EY2 - EY**2
    corr  = cov / np.sqrt(varX * varY)
    print(f"  E[X]={EX:.4f}  E[Y]={EY:.4f}  Cov(X,Y)={cov:.4f}  ρ={corr:.4f}")


# ── 3. Multivariate Normal ────────────────────────────────────────────────────
def multivariate_normal():
    print("\n=== Multivariate Normal Distribution ===")
    mu  = np.array([1., 2.])
    Cov = np.array([[2., 1.],
                    [1., 1.5]])
    mvn = stats.multivariate_normal(mean=mu, cov=Cov)

    print(f"  μ = {mu}")
    print(f"  Σ =\n{Cov}")
    print(f"  Correlation ρ = Σ₁₂/√(Σ₁₁Σ₂₂) = {Cov[0,1]/np.sqrt(Cov[0,0]*Cov[1,1]):.4f}")

    # Marginal distributions
    print(f"\n  Marginal X₁ ~ N({mu[0]},{Cov[0,0]})")
    print(f"  Marginal X₂ ~ N({mu[1]},{Cov[1,1]})")

    # Conditional distribution X₁ | X₂ = x₂
    x2 = 2.5
    mu_cond  = mu[0] + Cov[0,1]/Cov[1,1] * (x2 - mu[1])
    var_cond = Cov[0,0] - Cov[0,1]**2/Cov[1,1]
    print(f"\n  Conditional X₁ | X₂={x2}: μ_cond={mu_cond:.4f}  σ²_cond={var_cond:.4f}")

    # Sample and verify
    rng = np.random.default_rng(0)
    samples = mvn.rvs(size=100_000, random_state=0)
    print(f"\n  MC sample (N=100k): mean={np.round(samples.mean(0),4)}  cov≈\n{np.round(np.cov(samples.T),4)}")


# ── 4. Covariance and correlation ─────────────────────────────────────────────
def covariance_correlation():
    print("\n=== Covariance and Correlation ===")
    rng  = np.random.default_rng(5)
    n    = 1000
    X    = rng.standard_normal(n)
    Y1   = 2*X + rng.standard_normal(n)   # positively correlated
    Y2   = -X  + rng.standard_normal(n)   # negatively correlated
    Y3   = rng.standard_normal(n)         # independent

    for label, Y in [("Y=2X+ε (pos)", Y1), ("Y=-X+ε (neg)", Y2), ("Y=ε (indep)", Y3)]:
        cov  = np.cov(X, Y)[0,1]
        corr = np.corrcoef(X, Y)[0,1]
        print(f"  {label:<20}: Cov={cov:.4f}  ρ={corr:.4f}")

    print("\n  Properties:")
    print(f"    Cov(X,X) = Var(X) = {np.var(X, ddof=1):.4f}")
    a, b = 2.0, 3.0
    print(f"    Cov(aX, bY1) = ab·Cov(X,Y1): {np.cov(a*X,b*Y1)[0,1]:.4f} vs {a*b*np.cov(X,Y1)[0,1]:.4f}")


# ── 5. Transformations of random variables ────────────────────────────────────
def rv_transformations():
    print("\n=== Transformations of RVs ===")
    rng = np.random.default_rng(9)
    n   = 500_000

    # X ~ N(0,1)  →  Y = X² ~ χ²(1)
    X = rng.standard_normal(n)
    Y = X**2
    chi2 = stats.chi2(df=1)
    print(f"  Y = X² where X~N(0,1):")
    print(f"    E[Y] empirical={Y.mean():.4f}  theory (chi²(1))={chi2.mean():.4f}")

    # Sum of squares: Z = Σ Xᵢ² ~ χ²(k)
    k = 5
    Z = rng.standard_normal((n, k))**2
    Z_sum = Z.sum(axis=1)
    chi2k = stats.chi2(df=k)
    print(f"\n  Z = Σᵢ₌₁⁵ Xᵢ² ~ χ²(5):")
    print(f"    E[Z] empirical={Z_sum.mean():.4f}  theory={chi2k.mean():.4f}")
    print(f"    Var[Z] empirical={Z_sum.var():.4f}  theory={chi2k.var():.4f}")

    # Box-Muller: U1,U2 uniform → two N(0,1) samples
    U1 = rng.uniform(0, 1, n)
    U2 = rng.uniform(0, 1, n)
    Z1 = np.sqrt(-2*np.log(U1)) * np.cos(2*np.pi*U2)
    Z2 = np.sqrt(-2*np.log(U1)) * np.sin(2*np.pi*U2)
    print(f"\n  Box-Muller transform: Z1 mean={Z1.mean():.4f} std={Z1.std():.4f}  Z2 mean={Z2.mean():.4f} std={Z2.std():.4f}")


if __name__ == "__main__":
    joint_discrete()
    joint_continuous()
    multivariate_normal()
    covariance_correlation()
    rv_transformations()
