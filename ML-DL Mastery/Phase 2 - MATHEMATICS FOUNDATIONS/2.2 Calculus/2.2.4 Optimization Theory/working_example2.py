"""
Working Example 2: Optimization Theory — GD, SGD, Momentum, Adam, Convexity
=============================================================================
Implements and compares optimizers from scratch: gradient descent, SGD mini-batch,
momentum, RMSProp, Adam; KKT conditions; convexity checks.

Run:  python working_example2.py
"""
from pathlib import Path
try:
    import numpy as np
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit("pip install numpy matplotlib")

DATA   = Path(__file__).parent / "data"
OUTPUT = Path(__file__).parent / "output"
DATA.mkdir(exist_ok=True); OUTPUT.mkdir(exist_ok=True)

def demo_convexity():
    print("=== Convexity Checks ===")
    # A differentiable function is convex iff Hessian is PSD everywhere
    # f(x,y) = x^2 + xy + 2y^2  -> H = [[2,1],[1,4]]
    H = np.array([[2., 1.], [1., 4.]])
    eigs = np.linalg.eigvalsh(H)
    print(f"  f(x,y) = x²+xy+2y²  H eigenvalues: {eigs}  -> {'CONVEX' if all(eigs>=0) else 'NOT CONVEX'}")

    # Saddle: f(x,y) = x^2 - y^2  -> H = [[2,0],[0,-2]]
    H2 = np.array([[2., 0.], [0., -2.]])
    eigs2 = np.linalg.eigvalsh(H2)
    print(f"  f(x,y) = x²-y²  H eigenvalues: {eigs2}  -> SADDLE POINT")

def optimise_all():
    print("\n=== Optimizer Comparison on Ridge Regression ===")
    np.random.seed(0)
    n, d = 100, 10
    X = np.random.randn(n, d)
    y = X @ np.random.randn(d) + 0.1 * np.random.randn(n)
    lam = 0.01

    def loss(w):
        return np.mean((X @ w - y)**2) + lam * np.dot(w, w)

    def full_grad(w):
        return 2 * X.T @ (X @ w - y) / n + 2 * lam * w

    def sgd_grad(w, idx):
        Xi, yi = X[idx], y[idx]
        return 2 * Xi.T @ (Xi @ w - yi) / len(idx) + 2 * lam * w

    w_opt = np.linalg.solve(X.T@X/n + lam*np.eye(d), X.T@y/n)
    n_iters = 200; batch = 16
    results = {}

    # --- Full-batch GD ---
    w = np.zeros(d); hist = []
    for _ in range(n_iters):
        w -= 0.1 * full_grad(w); hist.append(loss(w))
    results["GD"] = hist

    # --- SGD mini-batch ---
    w = np.zeros(d); hist = []
    for t in range(n_iters):
        idx = np.random.choice(n, batch, replace=False)
        w -= 0.05 * sgd_grad(w, idx); hist.append(loss(w))
    results["SGD"] = hist

    # --- Momentum ---
    w = np.zeros(d); v = np.zeros(d); hist = []
    for _ in range(n_iters):
        v = 0.9 * v - 0.05 * full_grad(w); w += v; hist.append(loss(w))
    results["Momentum"] = hist

    # --- Adam ---
    w = np.zeros(d); m = np.zeros(d); vv = np.zeros(d); hist = []
    lr, b1, b2, eps = 0.05, 0.9, 0.999, 1e-8
    for t in range(1, n_iters+1):
        idx = np.random.choice(n, batch, replace=False)
        g = sgd_grad(w, idx)
        m = b1*m + (1-b1)*g; vv = b2*vv + (1-b2)*g**2
        m_hat = m/(1-b1**t); v_hat = vv/(1-b2**t)
        w -= lr * m_hat / (np.sqrt(v_hat) + eps); hist.append(loss(w))
    results["Adam"] = hist

    opt_loss = loss(w_opt)
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, h in results.items():
        ax.semilogy([max(l - opt_loss + 1e-10, 1e-10) for l in h], label=name)
    ax.axhline(0, color="k", ls="--", lw=0.7, label="optimal")
    ax.set_xlabel("Iteration"); ax.set_ylabel("Loss - optimal (log)")
    ax.set_title("Optimizer Comparison"); ax.legend()
    fig.savefig(OUTPUT / "optimizer_comparison.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    for name, h in results.items():
        print(f"  {name:10s} final loss: {h[-1]:.6f}")
    print(f"  Optimal loss: {opt_loss:.6f}")
    print(f"  Saved: optimizer_comparison.png")

def demo_newton():
    print("\n=== Newton's Method (1D) ===")
    # Minimise f(x) = x^4 - 4x^2 + x  -> one of two local minima
    f   = lambda x: x**4 - 4*x**2 + x
    df  = lambda x: 4*x**3 - 8*x + 1
    d2f = lambda x: 12*x**2 - 8

    x = 2.0; path = [x]
    for _ in range(10):
        step = df(x) / d2f(x)
        x -= step
        path.append(x)
    print(f"  Newton converged to x = {x:.6f}  f(x) = {f(x):.6f}")

    xs = np.linspace(-2.5, 2.5, 300)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(xs, f(xs), label="f(x)"); ax.scatter(path, [f(p) for p in path], c="r", zorder=5)
    ax.set_ylim(-10, 10); ax.legend(); ax.set_title("Newton's Method")
    fig.savefig(OUTPUT / "newton.png", dpi=120, bbox_inches="tight")
    plt.close(fig); print(f"  Saved: newton.png")

if __name__ == "__main__":
    demo_convexity()
    optimise_all()
    demo_newton()
