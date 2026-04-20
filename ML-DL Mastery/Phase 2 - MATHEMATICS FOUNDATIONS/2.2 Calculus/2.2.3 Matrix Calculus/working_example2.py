"""
Working Example 2: Matrix Calculus — Backpropagation & Gradient Derivations
=============================================================================
Numerically verify matrix calculus identities used in deep learning:
d/dw (Xw-y)²  = X^T(Xw-y),  softmax gradient, MSE + L2 gradient.

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

DATA   = Path(__file__).parent / "data"
OUTPUT = Path(__file__).parent / "output"
DATA.mkdir(exist_ok=True); OUTPUT.mkdir(exist_ok=True)

def numerical_gradient_matrix(f, W, h=1e-6):
    """Element-wise gradient of scalar f w.r.t. matrix W."""
    G = np.zeros_like(W, dtype=float)
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            Wp, Wm = W.copy(), W.copy()
            Wp[i,j] += h; Wm[i,j] -= h
            G[i,j] = (f(Wp) - f(Wm)) / (2*h)
    return G

def demo_linear_regression_gradient():
    print("=== d/dw ||Xw - y||² = 2 X^T(Xw-y) ===")
    np.random.seed(42)
    n, d = 20, 5
    X = np.random.randn(n, d)
    y = np.random.randn(n)
    w = np.random.randn(d)

    # Scalar loss
    f_w = lambda v: np.sum((X @ v - y)**2)
    # Analytic gradient
    g_ana = 2 * X.T @ (X @ w - y)
    # Numerical gradient
    h = 1e-7
    g_num = np.array([(f_w(w + h*np.eye(d)[i]) - f_w(w - h*np.eye(d)[i])) / (2*h) for i in range(d)])
    print(f"  Analytic:  {g_ana.round(4)}")
    print(f"  Numerical: {g_num.round(4)}")
    print(f"  Max error: {np.max(np.abs(g_ana - g_num)):.2e}")

def demo_softmax_gradient():
    print("\n=== Softmax Jacobian dS/dz ===")
    def softmax(z):
        e = np.exp(z - z.max())
        return e / e.sum()

    z = np.array([2.0, 1.0, 0.1])
    s = softmax(z)
    # Analytic Jacobian: diag(s) - s s^T
    J_ana = np.diag(s) - np.outer(s, s)
    # Numerical Jacobian (central difference)
    h = 1e-7
    J_num = np.array([(softmax(z + h*np.eye(3)[i]) - softmax(z - h*np.eye(3)[i])) / (2*h) for i in range(3)]).T
    print(f"  s(z):      {s.round(4)}")
    print(f"  Max |J_ana - J_num|: {np.max(np.abs(J_ana - J_num)):.2e}")
    print(f"  Row sums (should be 0 — probabilities sum to 1):\n  {J_ana.sum(axis=1).round(8)}")

def demo_ridge_regression_gradient():
    print("\n=== Ridge regression: d/dw [||Xw-y||² + lambda||w||²] ===")
    np.random.seed(1)
    n, d = 50, 8
    X = np.random.randn(n, d)
    y = X @ np.ones(d) + 0.1 * np.random.randn(n)
    lam = 0.1
    w = np.zeros(d)

    f = lambda v: np.sum((X @ v - y)**2) + lam * np.dot(v, v)
    g_ana = 2 * X.T @ (X @ w - y) + 2 * lam * w
    h = 1e-7
    g_num = np.array([(f(w + h*np.eye(d)[i]) - f(w - h*np.eye(d)[i])) / (2*h) for i in range(d)])
    print(f"  Analytic:  {g_ana.round(4)}")
    print(f"  Numerical: {g_num.round(4)}")
    print(f"  Max error: {np.max(np.abs(g_ana - g_num)):.2e}")

    # GD to optimal
    w = np.random.randn(d)
    losses = []
    for _ in range(500):
        grad = 2 * X.T @ (X @ w - y) + 2 * lam * w
        w -= 0.01 * grad
        losses.append(f(w))

    w_opt = np.linalg.solve(2*(X.T@X + lam*np.eye(d)), 2*X.T@y)
    print(f"\n  GD w: {w.round(4)}")
    print(f"  Analytic w*: {w_opt.round(4)}")
    print(f"  Error: {np.linalg.norm(w - w_opt):.4e}")

    fig, ax = plt.subplots(figsize=(6,4))
    ax.semilogy(losses)
    ax.set_xlabel("Iteration"); ax.set_ylabel("Loss"); ax.set_title("Ridge GD convergence")
    fig.savefig(OUTPUT / "ridge_gd.png", dpi=120, bbox_inches="tight")
    plt.close(fig); print(f"  Saved: ridge_gd.png")

def demo_gradient_of_quadratic():
    print("\n=== Gradient of f(w) = 0.5*w'Aw - b'w ===")
    np.random.seed(5)
    d = 4
    M = np.random.randn(d, d)
    A = M @ M.T + np.eye(d)           # symmetric positive definite
    b = np.random.randn(d)
    w = np.random.randn(d)

    # Analytic gradient: df/dw = Aw - b
    g_analytic = A @ w - b
    f = lambda v: 0.5 * v @ A @ v - b @ v
    h = 1e-6
    g_numeric = np.array([
        (f(w + h*np.eye(d)[i]) - f(w - h*np.eye(d)[i])) / (2*h)
        for i in range(d)
    ])
    print(f"  d={d}, A = random PD {d}x{d}")
    print(f"  Analytic  gradient: {g_analytic.round(6)}")
    print(f"  Numerical gradient: {g_numeric.round(6)}")
    print(f"  Max error: {np.max(np.abs(g_analytic - g_numeric)):.2e}")
    w_star = np.linalg.solve(A, b)
    print(f"  Minimum at w* = A^-1 b: {w_star.round(6)}, f(w*) = {f(w_star):.6f}")


def demo_jacobian_numeric():
    print("\n=== Jacobian of f: R^3 -> R^2 (numerical vs analytic) ===")
    # f(x1,x2,x3) = [x1^2 + sin(x2),  x2*x3 + exp(x1)]
    F = lambda v: np.array([
        v[0]**2 + np.sin(v[1]),
        v[1]*v[2] + np.exp(v[0])
    ])
    x0 = np.array([1.0, 2.0, 0.5])
    h = 1e-6; m, n = 2, 3
    J_num = np.zeros((m, n))
    for j in range(n):
        xp, xm = x0.copy(), x0.copy()
        xp[j] += h; xm[j] -= h
        J_num[:, j] = (F(xp) - F(xm)) / (2 * h)

    # Analytic: df1/dx1=2x1, df1/dx2=cos(x2), df1/dx3=0
    #           df2/dx1=e^x1, df2/dx2=x3,      df2/dx3=x2
    J_ana = np.array([
        [2*x0[0],       np.cos(x0[1]),  0     ],
        [np.exp(x0[0]), x0[2],          x0[1] ]
    ])
    print(f"  x0 = {x0}")
    print(f"  Numerical Jacobian (2x3):\n{J_num.round(6)}")
    print(f"  Analytic  Jacobian (2x3):\n{J_ana.round(6)}")
    print(f"  Max error: {np.max(np.abs(J_num - J_ana)):.2e}")


def demo_least_squares_gradient():
    print("\n=== MSE Gradient: dL/dw = 2X'(Xw-y) ===")
    np.random.seed(12)
    n, d = 30, 4
    X = np.random.randn(n, d)
    w_true = np.array([1.0, -2.0, 0.5, 3.0])
    y = X @ w_true + 0.1 * np.random.randn(n)
    w = np.zeros(d)

    mse = lambda v: np.mean((X @ v - y)**2)
    g_analytic = 2 * X.T @ (X @ w - y) / n
    h = 1e-6
    g_numeric = np.array([
        (mse(w + h*np.eye(d)[i]) - mse(w - h*np.eye(d)[i])) / (2*h)
        for i in range(d)
    ])
    print(f"  n={n}, d={d}, w initialised to zeros")
    print(f"  MSE at w=0: {mse(w):.4f}")
    print(f"  Analytic  gradient: {g_analytic.round(4)}")
    print(f"  Numerical gradient: {g_numeric.round(4)}")
    print(f"  Max error: {np.max(np.abs(g_analytic - g_numeric)):.2e}")
    lr = 0.01
    w_new = w - lr * g_analytic
    improved = mse(w_new) < mse(w)
    print(f"  MSE after one GD step (lr={lr}): {mse(w_new):.4f}  (should decrease)")
    print(f"  Loss decreased: {improved}")


if __name__ == "__main__":
    demo_linear_regression_gradient()
    demo_softmax_gradient()
    demo_ridge_regression_gradient()
    demo_gradient_of_quadratic()
    demo_jacobian_numeric()
    demo_least_squares_gradient()
