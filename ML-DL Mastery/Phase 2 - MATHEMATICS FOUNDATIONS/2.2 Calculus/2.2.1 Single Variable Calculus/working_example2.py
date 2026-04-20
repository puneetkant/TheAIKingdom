"""
Working Example 2: Single Variable Calculus — Derivatives, Numerical Differentiation, Taylor Series
====================================================================================================
Demonstrates: numerical derivatives (finite difference), chain rule verification,
Taylor series approximation, gradient descent for 1D function optimisation.

Run:  python working_example2.py
"""
from pathlib import Path
try:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit("pip install numpy matplotlib")

OUTPUT = Path(__file__).parent / "output"
OUTPUT.mkdir(exist_ok=True)

def demo_numerical_derivative():
    print("=== Numerical Derivatives ===")
    f = lambda x: np.sin(x) + 0.5 * x**2
    df_true = lambda x: np.cos(x) + x

    x0 = 1.0
    for h in [1e-1, 1e-4, 1e-7, 1e-10]:
        # Central difference: more accurate than forward
        df_num = (f(x0 + h) - f(x0 - h)) / (2 * h)
        df_exact = df_true(x0)
        print(f"  h={h:.0e}: f'(1) ~= {df_num:.8f}  err={abs(df_num - df_exact):.2e}")

def demo_chain_rule():
    print("\n=== Chain Rule Verification ===")
    # f(x) = sigmoid(x^2), f'(x) = sigmoid(x^2) * (1-sigmoid(x^2)) * 2x
    def sigmoid(u): return 1 / (1 + np.exp(-u))
    f   = lambda x: sigmoid(x**2)
    df  = lambda x: sigmoid(x**2) * (1 - sigmoid(x**2)) * 2 * x

    x = np.array([0.5, 1.0, 2.0])
    h = 1e-7
    df_num = (f(x + h) - f(x - h)) / (2 * h)
    print(f"  Analytic:  {df(x).round(6)}")
    print(f"  Numerical: {df_num.round(6)}")
    print(f"  Max error: {np.max(np.abs(df(x) - df_num)):.2e}")

def demo_taylor_series():
    print("\n=== Taylor Series Approximation ===")
    # sin(x) ~= x - x^3/6 + x^5/120 - x^7/5040
    x = np.linspace(-np.pi, np.pi, 300)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, np.sin(x), "k", lw=2, label="sin(x)")
    for n, c, terms in [(1, "blue", [1]), (3, "green", [1, -1/6]),
                        (5, "orange", [1, -1/6, 1/120]),
                        (7, "red", [1, -1/6, 1/120, -1/5040])]:
        powers = [1, 3, 5, 7][:len(terms)]
        approx = sum(c * x**p for c, p in zip(terms, powers))
        ax.plot(x, approx, "--", c=c, label=f"T_{n}")
    ax.set_ylim(-2, 2); ax.legend(); ax.set_title("Taylor series of sin(x)")
    fig.savefig(OUTPUT / "taylor_sin.png", dpi=120, bbox_inches="tight")
    plt.close(fig); print(f"  Saved: taylor_sin.png")

def demo_gradient_descent_1d():
    print("\n=== Gradient Descent (1D) ===")
    # Minimise f(x) = (x-3)^2 + 2
    f   = lambda x: (x - 3)**2 + 2
    df  = lambda x: 2 * (x - 3)

    x = 0.0; lr = 0.1; path = [x]
    for _ in range(50):
        x = x - lr * df(x)
        path.append(x)

    print(f"  Minimum at x ~= {path[-1]:.6f}  (true: 3.0)")
    print(f"  f(x*) ~= {f(path[-1]):.6f}  (true: 2.0)")

    xs = np.linspace(-1, 6, 200)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(xs, f(xs)); axes[0].scatter(path[::5], [f(p) for p in path[::5]], c="r")
    axes[0].set_title("f(x) with GD steps")
    axes[1].semilogy([abs(p - 3) for p in path]); axes[1].set_title("Error vs iteration")
    fig.savefig(OUTPUT / "gd_1d.png", dpi=120, bbox_inches="tight")
    plt.close(fig); print(f"  Saved: gd_1d.png")

def demo_numerical_derivatives():
    print("\n=== Numerical Derivatives: Forward/Backward/Central Differences ===")
    f  = lambda x: x**3 - 2*x + 1
    df = lambda x: 3*x**2 - 2          # analytical derivative

    x0 = 2.0
    df_exact = df(x0)
    print(f"  f(x) = x^3 - 2x + 1  at x={x0}")
    print(f"  Analytical f'({x0}) = {df_exact:.6f}")
    print(f"  {'h':>10s}  {'forward':>12s}  {'backward':>12s}  {'central':>12s}  {'fwd_err':>10s}  {'ctr_err':>10s}")
    hs = [1e-1, 1e-2, 1e-4, 1e-6, 1e-8, 1e-10]
    fwd_errs, ctr_errs = [], []
    for h in hs:
        fwd = (f(x0 + h) - f(x0)) / h
        bwd = (f(x0) - f(x0 - h)) / h
        ctr = (f(x0 + h) - f(x0 - h)) / (2 * h)
        fe, ce = abs(fwd - df_exact), abs(ctr - df_exact)
        fwd_errs.append(fe); ctr_errs.append(ce)
        print(f"  {h:>10.0e}  {fwd:>12.6f}  {bwd:>12.6f}  {ctr:>12.6f}  {fe:>10.2e}  {ce:>10.2e}")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.loglog(hs, fwd_errs, "o-", label="Forward diff error")
    ax.loglog(hs, ctr_errs, "s-", label="Central diff error")
    ax.set_xlabel("Step size h"); ax.set_ylabel("Absolute error")
    ax.set_title("Derivative error vs step size h  (f=x^3-2x+1 at x=2)")
    ax.legend(); ax.invert_xaxis()
    fig.savefig(OUTPUT / "derivative_error.png", dpi=120, bbox_inches="tight")
    plt.close(fig); print("  Saved: derivative_error.png")


def demo_integration():
    print("\n=== Numerical Integration: Trapezoidal and Simpson's Rule ===")
    # integral of sin(x) from 0 to pi = 2 (exact)
    exact = 2.0
    a, b = 0.0, np.pi
    print(f"  Integral of sin(x) from 0 to pi  (exact = {exact})")
    print(f"  {'n panels':>10s}  {'trapezoidal':>14s}  {'simpsons':>12s}  {'trap_err':>10s}  {'simp_err':>10s}")
    panel_counts = [4, 8, 16, 32, 64, 128, 256]
    for n in panel_counts:
        x = np.linspace(a, b, n + 1)
        y = np.sin(x)
        h = (b - a) / n
        trap = 0.5 * h * (y[0] + 2 * np.sum(y[1:-1]) + y[-1])
        simp = (h / 3) * (y[0] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-2:2]) + y[-1])
        te, se = abs(trap - exact), abs(simp - exact)
        print(f"  {n:>10d}  {trap:>14.8f}  {simp:>12.8f}  {te:>10.2e}  {se:>10.2e}")
    print("  Convergence: trapezoidal O(h^2), Simpson's O(h^4)")


def demo_taylor_series_extended():
    print("\n=== Taylor Series: exp(x) and sin(x) around x=0 ===")
    from math import factorial
    x = np.linspace(-3, 3, 400)

    def exp_taylor(x_arr, terms):
        return sum(x_arr**k / factorial(k) for k in range(terms))

    def sin_taylor(x_arr, terms):
        return sum(((-1)**k) * x_arr**(2*k+1) / factorial(2*k+1) for k in range(terms))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(x, np.exp(x), "k", lw=2.5, label="exp(x) true")
    for n, c in zip([1, 2, 3, 5], ["blue", "green", "orange", "red"]):
        axes[0].plot(x, exp_taylor(x, n), "--", color=c, label=f"{n} terms")
    axes[0].set_ylim(-2, 10); axes[0].legend(fontsize=8)
    axes[0].set_title("Taylor series of exp(x)")

    axes[1].plot(x, np.sin(x), "k", lw=2.5, label="sin(x) true")
    for n, c in zip([1, 2, 3, 5], ["blue", "green", "orange", "red"]):
        axes[1].plot(x, sin_taylor(x, n), "--", color=c, label=f"{n} terms")
    axes[1].set_ylim(-3, 3); axes[1].legend(fontsize=8)
    axes[1].set_title("Taylor series of sin(x)")

    fig.tight_layout()
    fig.savefig(OUTPUT / "taylor_series.png", dpi=120, bbox_inches="tight")
    plt.close(fig); print("  Saved: taylor_series.png")


if __name__ == "__main__":
    demo_numerical_derivative()
    demo_chain_rule()
    demo_taylor_series()
    demo_gradient_descent_1d()
    demo_numerical_derivatives()
    demo_integration()
    demo_taylor_series_extended()
