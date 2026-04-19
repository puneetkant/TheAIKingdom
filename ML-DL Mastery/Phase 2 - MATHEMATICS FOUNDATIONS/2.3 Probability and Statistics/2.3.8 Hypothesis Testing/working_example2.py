"""
Working Example 2: Hypothesis Testing — t-test, Chi², A/B Test, p-values, Power
=================================================================================
One-sample t-test, two-sample t-test, chi-squared test of independence,
A/B test simulation, Type I/II errors, power analysis.

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

def t_statistic_pvalue(data, mu0):
    """One-sample t-test vs null H0: mu = mu0."""
    n = len(data)
    t = (np.mean(data) - mu0) / (np.std(data, ddof=1) / math.sqrt(n))
    # p-value via t-distribution CDF approximation (two-sided)
    # Use Student-t quantile via normal approximation for df>30
    df = n - 1
    # Approximation: p ~= 2 * P(Z > |t|) for large df
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))
    return t, p

def demo_one_sample_t():
    print("=== One-Sample t-test ===")
    np.random.seed(42)
    data = np.random.normal(5.3, 1.5, 30)   # true mean 5.3
    for mu0 in [5.0, 5.3, 6.0]:
        t, p = t_statistic_pvalue(data, mu0)
        reject = "REJECT H0" if p < 0.05 else "FAIL TO REJECT"
        print(f"  H0: mu={mu0}: t={t:.3f}  p={p:.4f}  -> {reject}")

def demo_two_sample_t():
    print("\n=== Two-Sample t-test (A/B Test) ===")
    np.random.seed(1)
    ctrl = np.random.normal(10.0, 2.0, 50)    # control
    trt1 = np.random.normal(10.5, 2.0, 50)    # treatment 1 (small effect)
    trt2 = np.random.normal(12.0, 2.0, 50)    # treatment 2 (large effect)

    def two_sample_t(a, b):
        n1, n2 = len(a), len(b)
        sp = math.sqrt(((n1-1)*np.var(a,ddof=1) + (n2-1)*np.var(b,ddof=1)) / (n1+n2-2))
        t = (np.mean(a) - np.mean(b)) / (sp * math.sqrt(1/n1 + 1/n2))
        p = 2 * (1 - 0.5*(1 + math.erf(abs(t)/math.sqrt(2))))
        return t, p

    for name, trt in [("trt1 (Delta=0.5)", trt1), ("trt2 (Delta=2.0)", trt2)]:
        t, p = two_sample_t(ctrl, trt)
        print(f"  {name}: t={t:.3f}  p={p:.4f}  {'SIGNIFICANT' if p<0.05 else 'NOT SIG'}")

def demo_type1_type2():
    print("\n=== Type I / Type II Errors ===")
    np.random.seed(0)
    alpha = 0.05; n = 30
    # Type I: H0 true (mu=0), test at alpha=0.05 -> should reject ~5%
    type1 = sum(
        abs(np.mean(np.random.normal(0,1,n))) > 1.96/math.sqrt(n)
        for _ in range(10_000)
    ) / 10_000
    # Type II: H1 true (mu=0.5), fail to reject
    type2 = sum(
        abs(np.mean(np.random.normal(0.5,1,n))) <= 1.96/math.sqrt(n)
        for _ in range(10_000)
    ) / 10_000
    print(f"  Type I error (false positive): {type1:.4f}  (alpha={alpha})")
    print(f"  Type II error (false negative): {type2:.4f}  (power={1-type2:.4f})")

def demo_chi_squared():
    print("\n=== Chi-Squared Test of Independence ===")
    # Observed: click rates by version (A vs B, clicked vs not)
    obs = np.array([[120, 880], [160, 840]])   # [A_click, A_noclick], [B_click, B_noclick]
    n = obs.sum()
    row_sums = obs.sum(axis=1, keepdims=True)
    col_sums = obs.sum(axis=0, keepdims=True)
    expected = row_sums @ col_sums / n
    chi2 = np.sum((obs - expected)**2 / expected)
    print(f"  Observed:\n{obs}")
    print(f"  Expected:\n{expected.round(2)}")
    print(f"  chi² = {chi2:.4f}  (df=1, critical value at p=0.05: 3.841)")
    print(f"  Result: {'SIGNIFICANT (p<0.05)' if chi2 > 3.841 else 'NOT SIGNIFICANT'}")

if __name__ == "__main__":
    demo_one_sample_t()
    demo_two_sample_t()
    demo_type1_type2()
    demo_chi_squared()
