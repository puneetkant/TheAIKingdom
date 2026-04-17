"""
Working Example: Hypothesis Testing
Covers null/alternative hypotheses, p-values, Type I/II errors,
z-tests, t-tests, chi-squared tests, ANOVA, and multiple testing.
"""
import numpy as np
from scipy import stats


# ── 1. Framework overview ─────────────────────────────────────────────────────
def framework():
    print("=== Hypothesis Testing Framework ===")
    print("  H₀: null hypothesis (claim to be tested)")
    print("  H₁: alternative hypothesis")
    print("  α  = P(Type I error)  = P(reject H₀ | H₀ true)  — significance level")
    print("  β  = P(Type II error) = P(fail to reject H₀ | H₁ true)")
    print("  Power = 1 - β = P(reject H₀ | H₁ true)")
    print("  p-value = P(observing test stat ≥ observed | H₀) — reject if p < α")


# ── 2. One-sample z-test ──────────────────────────────────────────────────────
def z_test():
    print("\n=== One-Sample z-Test (known σ) ===")
    rng  = np.random.default_rng(0)
    mu0  = 100   # H₀: μ = 100
    mu1  = 103   # true mean
    sigma = 15
    n    = 50
    alpha = 0.05
    data = rng.normal(mu1, sigma, n)

    z_stat = (data.mean() - mu0) / (sigma / np.sqrt(n))
    p_val  = 2 * (1 - stats.norm.cdf(abs(z_stat)))   # two-tailed
    z_crit = stats.norm.ppf(1 - alpha/2)

    print(f"  H₀: μ={mu0}  True μ={mu1}  σ={sigma}  n={n}")
    print(f"  x̄={data.mean():.4f}  z={z_stat:.4f}  z_crit=±{z_crit:.4f}")
    print(f"  p-value={p_val:.4f}  α={alpha}")
    print(f"  Decision: {'REJECT H₀' if p_val < alpha else 'FAIL TO REJECT H₀'}")


# ── 3. One-sample and two-sample t-tests ─────────────────────────────────────
def t_tests():
    print("\n=== t-Tests ===")
    rng = np.random.default_rng(1)

    # One-sample: is μ = 5?
    data = rng.normal(loc=5.8, scale=2, size=30)
    t, p = stats.ttest_1samp(data, popmean=5)
    print(f"  One-sample t-test (H₀: μ=5):")
    print(f"    t={t:.4f}  p={p:.4f}  {'reject' if p<0.05 else 'fail'} H₀ at α=0.05")

    # Two-sample independent (equal variances assumed)
    A = rng.normal(10, 2, 40)
    B = rng.normal(11.5, 2, 35)
    t2, p2 = stats.ttest_ind(A, B, equal_var=True)
    print(f"\n  Two-sample t-test (H₀: μ_A=μ_B):")
    print(f"    x̄_A={A.mean():.4f}  x̄_B={B.mean():.4f}")
    print(f"    t={t2:.4f}  p={p2:.4f}  {'reject' if p2<0.05 else 'fail'} H₀")

    # Welch's t-test (unequal variances — preferred by default)
    C = rng.normal(10, 2, 40)
    D = rng.normal(10.5, 4, 25)   # different variance
    t3, p3 = stats.ttest_ind(C, D, equal_var=False)
    print(f"\n  Welch's t-test (H₀: μ_C=μ_D, unequal var):")
    print(f"    t={t3:.4f}  p={p3:.4f}  {'reject' if p3<0.05 else 'fail'} H₀")

    # Paired t-test
    before = rng.normal(70, 5, 20)
    after  = before - rng.normal(3, 2, 20)   # treatment effect ≈ 3
    t4, p4 = stats.ttest_rel(before, after)
    print(f"\n  Paired t-test (before/after, H₀: Δ=0):")
    print(f"    mean diff={np.mean(before-after):.4f}  t={t4:.4f}  p={p4:.4f}  {'reject' if p4<0.05 else 'fail'} H₀")


# ── 4. Chi-squared tests ──────────────────────────────────────────────────────
def chi_squared_tests():
    print("\n=== Chi-Squared Tests ===")
    rng = np.random.default_rng(2)

    # Goodness-of-fit: is the die fair?
    observed = np.array([18, 22, 15, 20, 14, 11])   # 100 rolls
    expected = np.full(6, 100/6)
    chi2, p  = stats.chisquare(observed, f_exp=expected)
    print(f"  Goodness-of-fit (fair die):")
    print(f"    observed: {observed}  total={observed.sum()}")
    print(f"    χ²={chi2:.4f}  p={p:.4f}  {'reject' if p<0.05 else 'fail'} H₀ at α=0.05")

    # Independence test (contingency table)
    # Gender × Preference
    table = np.array([[45, 30],   # Male: prefer A, B
                       [25, 50]]) # Female: prefer A, B
    chi2_ind, p_ind, dof, expected_table = stats.chi2_contingency(table)
    print(f"\n  Independence test (gender × preference):")
    print(f"    contingency table:\n{table}")
    print(f"    χ²={chi2_ind:.4f}  p={p_ind:.4f}  df={dof}  {'reject' if p_ind<0.05 else 'fail'} independence")

    # Normality: Shapiro-Wilk
    normal_data = rng.normal(0, 1, 50)
    unif_data   = rng.uniform(0, 1, 50)
    for label, d in [("Normal(0,1)", normal_data), ("Uniform(0,1)", unif_data)]:
        stat, p_sw = stats.shapiro(d)
        print(f"\n  Shapiro-Wilk normality ({label}): W={stat:.4f}  p={p_sw:.4f}  {'normal' if p_sw>=0.05 else 'NOT normal'}")


# ── 5. ANOVA ──────────────────────────────────────────────────────────────────
def anova():
    print("\n=== One-Way ANOVA (H₀: all group means equal) ===")
    rng = np.random.default_rng(3)
    G1  = rng.normal(10, 2, 30)
    G2  = rng.normal(12, 2, 30)   # different mean
    G3  = rng.normal(10.5, 2, 30)
    G4  = rng.normal(11, 2, 30)

    F, p = stats.f_oneway(G1, G2, G3, G4)
    print(f"  Groups: G1~N(10,4) G2~N(12,4) G3~N(10.5,4) G4~N(11,4)")
    print(f"  F={F:.4f}  p={p:.4f}  {'reject H₀ (≥1 means differ)' if p<0.05 else 'fail to reject H₀'}")

    # Post-hoc pairwise (manual Bonferroni corrected)
    groups = {"G1":G1,"G2":G2,"G3":G3,"G4":G4}
    pairs  = [("G1","G2"),("G1","G3"),("G1","G4"),("G2","G3"),("G2","G4"),("G3","G4")]
    alpha  = 0.05
    alpha_bonf = alpha / len(pairs)
    print(f"\n  Post-hoc (Bonferroni α_adj={alpha_bonf:.4f}):")
    for a, b in pairs:
        _, p_pair = stats.ttest_ind(groups[a], groups[b])
        sig = "*" if p_pair < alpha_bonf else ""
        print(f"    {a} vs {b}: p={p_pair:.4f} {sig}")


# ── 6. Power and sample size ──────────────────────────────────────────────────
def power_analysis():
    print("\n=== Power Analysis ===")
    # Effect size d = (μ1-μ0)/σ
    mu0, mu1, sigma = 0, 0.5, 1.0
    alpha, beta = 0.05, 0.20
    d = abs(mu1-mu0) / sigma

    # Required n (two-sided, two-sample)
    z_alpha = stats.norm.ppf(1-alpha/2)
    z_beta  = stats.norm.ppf(1-beta)
    n_req   = 2 * ((z_alpha + z_beta) / d)**2
    print(f"  Effect size d={d:.2f}  α={alpha}  β={beta}  power={1-beta:.0%}")
    print(f"  Required n per group ≈ {int(np.ceil(n_req))}")

    # Power curve vs n
    print(f"\n  Power at various n (two-sample z-test):")
    print(f"  {'n':<6} {'power':<10} {'β'}")
    for n_val in [20, 40, 64, 80, 100, 150]:
        se   = np.sqrt(2/n_val) * sigma
        z_eff = (mu1-mu0)/se - z_alpha
        power = stats.norm.cdf(z_eff)
        print(f"  {n_val:<6} {power:<10.4f} {1-power:.4f}")


# ── 7. Multiple testing correction ───────────────────────────────────────────
def multiple_testing():
    print("\n=== Multiple Testing Correction ===")
    rng = np.random.default_rng(4)
    # 100 tests, 10 truly significant (signal), 90 null
    n = 30
    p_vals = (
        [stats.ttest_1samp(rng.normal(0.5, 1, n), 0)[1] for _ in range(10)]  +  # signal
        [stats.ttest_1samp(rng.normal(0.0, 1, n), 0)[1] for _ in range(90)]     # null
    )
    p_vals = np.array(p_vals)

    alpha = 0.05
    # Uncorrected
    reject_raw = (p_vals < alpha).sum()

    # Bonferroni
    reject_bonf = (p_vals < alpha/len(p_vals)).sum()

    # Benjamini-Hochberg (FDR)
    m   = len(p_vals)
    idx = np.argsort(p_vals)
    bh_thresholds = (np.arange(1, m+1) / m) * alpha
    reject_bh = (p_vals[idx] <= bh_thresholds).sum()

    print(f"  100 tests (10 signal, 90 null)  α={alpha}")
    print(f"  {'Method':<20} {'Rejections'}")
    print(f"  {'Uncorrected':<20} {reject_raw}")
    print(f"  {'Bonferroni':<20} {reject_bonf}  (α_adj={alpha/m:.4f})")
    print(f"  {'Benjamini-Hochberg':<20} {reject_bh}  (controls FDR)")


if __name__ == "__main__":
    framework()
    z_test()
    t_tests()
    chi_squared_tests()
    anova()
    power_analysis()
    multiple_testing()
