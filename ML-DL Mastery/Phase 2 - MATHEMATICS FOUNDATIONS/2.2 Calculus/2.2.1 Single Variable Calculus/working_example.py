"""
Working Example: Single Variable Calculus
Covers limits, derivatives, integrals, the fundamental theorem,
Taylor series, and numerical differentiation/integration.
"""
import numpy as np
from scipy import integrate, misc


# ── 1. Limits ─────────────────────────────────────────────────────────────────
def limits():
    print("=== Limits ===")
    # Numeric approach: squeeze from both sides
    def limit_approach(f, x0, label, eps_vals=None):
        if eps_vals is None:
            eps_vals = [1e-1, 1e-3, 1e-6, 1e-9]
        print(f"\n  lim_{label}:")
        for eps in eps_vals:
            try:
                fl = f(x0 - eps)
                fr = f(x0 + eps)
                print(f"    ε={eps:.0e}:  f(x₀-ε)={fl:.8f}  f(x₀+ε)={fr:.8f}")
            except (ZeroDivisionError, ValueError):
                pass

    # lim_{x→0} sin(x)/x = 1
    limit_approach(lambda x: np.sin(x)/x, 0, "x→0 sin(x)/x")
    # lim_{x→0} (e^x - 1)/x = 1
    limit_approach(lambda x: (np.exp(x)-1)/x, 0, "x→0 (eˣ-1)/x")
    # lim_{x→∞} (1 + 1/x)^x = e
    print("\n  lim_{x→∞} (1+1/x)^x:")
    for x in [10, 100, 1e4, 1e7]:
        val = (1 + 1/x)**x
        print(f"    x={x:.0e}: {val:.8f}  (e = {np.e:.8f})")


# ── 2. Derivatives ────────────────────────────────────────────────────────────
def derivatives():
    print("\n=== Derivatives ===")
    # Analytical derivatives
    funcs = {
        "x³":         (lambda x: x**3,        lambda x: 3*x**2),
        "sin(x)":     (lambda x: np.sin(x),   lambda x: np.cos(x)),
        "eˣ":         (lambda x: np.exp(x),   lambda x: np.exp(x)),
        "ln(x)":      (lambda x: np.log(x),   lambda x: 1/x),
        "x·sin(x)":   (lambda x: x*np.sin(x), lambda x: np.sin(x)+x*np.cos(x)),
    }
    x = np.pi / 4
    print(f"  Evaluated at x = π/4 ≈ {x:.4f}")
    print(f"  {'f(x)':<15} {'f\'(x) exact':<15} {'f\'(x) numeric':<15} {'error'}")
    print("  " + "-"*60)
    for name, (f, df) in funcs.items():
        exact   = df(x)
        numeric = misc.derivative(f, x, dx=1e-6)
        err     = abs(exact - numeric)
        print(f"  {name:<15} {exact:<15.6f} {numeric:<15.6f} {err:.2e}")

    # Chain rule demo: d/dx sin(x²) = 2x cos(x²)
    print("\n  Chain rule: d/dx sin(x²) = 2x·cos(x²)")
    f   = lambda x: np.sin(x**2)
    df  = lambda x: 2*x * np.cos(x**2)
    x0  = 1.5
    print(f"    exact={df(x0):.6f}  numeric={misc.derivative(f, x0, dx=1e-7):.6f}")


# ── 3. Numerical differentiation (finite differences) ────────────────────────
def finite_differences():
    print("\n=== Finite Differences ===")
    f  = lambda x: np.exp(x) * np.sin(x)
    df = lambda x: np.exp(x) * (np.sin(x) + np.cos(x))
    x0 = 1.0
    true_val = df(x0)

    print(f"  f(x) = eˣ sin(x)   at x={x0}")
    print(f"  {'h':<12} {'forward':<16} {'central':<16} {'err_fwd':<12} {'err_ctr'}")
    for h in [1e-1, 1e-3, 1e-5, 1e-7, 1e-9]:
        fwd = (f(x0+h) - f(x0)) / h
        ctr = (f(x0+h) - f(x0-h)) / (2*h)
        print(f"  {h:<12.0e} {fwd:<16.8f} {ctr:<16.8f} {abs(fwd-true_val):<12.2e} {abs(ctr-true_val):.2e}")


# ── 4. Integration ────────────────────────────────────────────────────────────
def integration():
    print("\n=== Integration ===")
    # Exact antiderivative comparison
    cases = [
        ("x²",  lambda x: x**2,  lambda a,b: b**3/3 - a**3/3,  0, 1),
        ("sin", lambda x: np.sin(x), lambda a,b: -np.cos(b)+np.cos(a), 0, np.pi),
        ("eˣ",  lambda x: np.exp(x), lambda a,b: np.exp(b)-np.exp(a), 0, 1),
    ]
    print(f"  {'f(x)':<8} {'exact':<14} {'scipy.quad':<14} {'error'}")
    for name, f, antideriv, a, b in cases:
        exact    = antideriv(a, b)
        numeric, _ = integrate.quad(f, a, b)
        print(f"  {name:<8} {exact:<14.8f} {numeric:<14.8f} {abs(exact-numeric):.2e}")

    # Numerical integration methods on [0, π]
    print("\n  Integrating sin(x) over [0,π] using different methods:")
    n   = 20
    xs  = np.linspace(0, np.pi, n)
    ys  = np.sin(xs)
    exact_val = 2.0

    trap   = np.trapz(ys, xs)
    simp   = integrate.simpson(ys, x=xs)
    gauss, _ = integrate.quad(np.sin, 0, np.pi)
    print(f"    Trapezoid rule (n={n}): {trap:.8f}  err={abs(trap-exact_val):.2e}")
    print(f"    Simpson's rule  (n={n}): {simp:.8f}  err={abs(simp-exact_val):.2e}")
    print(f"    Gauss quadrature      : {gauss:.8f}  err={abs(gauss-exact_val):.2e}")


# ── 5. Taylor series ──────────────────────────────────────────────────────────
def taylor_series():
    print("\n=== Taylor Series ===")
    def taylor_sin(x, n_terms):
        """sin(x) ≈ Σ (-1)^k x^(2k+1) / (2k+1)!"""
        result = 0.0
        for k in range(n_terms):
            result += ((-1)**k) * x**(2*k+1) / np.math.factorial(2*k+1)
        return result

    def taylor_exp(x, n_terms):
        """eˣ ≈ Σ x^k / k!"""
        result = 0.0
        for k in range(n_terms):
            result += x**k / np.math.factorial(k)
        return result

    x = 1.2
    print(f"  sin({x}) = {np.sin(x):.8f}  |  eˣ at x={x}: {np.exp(x):.8f}")
    print(f"\n  {'terms':<8} {'sin approx':<14} {'err_sin':<12} {'exp approx':<14} {'err_exp'}")
    for n in [1, 2, 3, 5, 8, 12]:
        s = taylor_sin(x, n)
        e = taylor_exp(x, n)
        print(f"  {n:<8} {s:<14.8f} {abs(s-np.sin(x)):<12.2e} {e:<14.8f} {abs(e-np.exp(x)):.2e}")


# ── 6. Mean Value Theorem demo ────────────────────────────────────────────────
def mean_value_theorem():
    print("\n=== Mean Value Theorem ===")
    # There exists c in (a,b) such that f'(c) = (f(b)-f(a))/(b-a)
    f  = lambda x: x**3 - 2*x
    df = lambda x: 3*x**2 - 2
    a, b = 0.5, 2.5
    mvt_slope = (f(b) - f(a)) / (b - a)
    # Solve f'(c) = mvt_slope: 3c²-2 = mvt_slope → c = sqrt((mvt_slope+2)/3)
    c = np.sqrt((mvt_slope + 2) / 3)
    print(f"  f(x) = x³ - 2x,  [a,b] = [{a},{b}]")
    print(f"  Average slope (f(b)-f(a))/(b-a) = {mvt_slope:.4f}")
    print(f"  c = {c:.4f}  (in ({a},{b})? {a < c < b})")
    print(f"  f'(c) = {df(c):.4f}  == {mvt_slope:.4f}? {np.isclose(df(c), mvt_slope)}")


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    limits()
    derivatives()
    finite_differences()
    integration()
    taylor_series()
    mean_value_theorem()
