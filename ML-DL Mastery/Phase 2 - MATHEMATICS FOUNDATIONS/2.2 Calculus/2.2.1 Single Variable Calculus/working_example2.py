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

if __name__ == "__main__":
    demo_numerical_derivative()
    demo_chain_rule()
    demo_taylor_series()
    demo_gradient_descent_1d()
