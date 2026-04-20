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
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit("pip install numpy matplotlib")
try:
    from scipy.optimize import minimize as sp_minimize
except ImportError:
    raise SystemExit("pip install scipy")

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

def demo_constrained_optimization():
    print("\n=== Constrained Optimization: min x^2+y^2 s.t. x+y=1 ===")
    # Lagrangian: L = x^2+y^2 - lam*(x+y-1)
    # dL/dx = 2x - lam = 0 -> x = lam/2
    # dL/dy = 2y - lam = 0 -> y = lam/2
    # Constraint: x+y=1 -> lam=1 -> x*=y*=0.5
    print("  Lagrangian analysis:")
    print("  L = x^2+y^2 - lam*(x+y-1)")
    print("  dL/dx=0: 2x - lam = 0 -> x = lam/2")
    print("  dL/dy=0: 2y - lam = 0 -> y = lam/2")
    print("  Constraint x+y=1 -> lam=1 -> x*=y*=0.5")
    x_star = np.array([0.5, 0.5])
    print(f"  Analytical solution: x*={x_star}, f(x*)={np.sum(x_star**2):.4f}")

    f_obj  = lambda v: v[0]**2 + v[1]**2
    grad_f = lambda v: np.array([2*v[0], 2*v[1]])
    constraint = {"type": "eq", "fun": lambda v: v[0] + v[1] - 1}
    res = sp_minimize(f_obj, x0=[0.0, 0.0], jac=grad_f,
                      constraints=constraint, method="SLSQP")
    print(f"  scipy solution:      x*={res.x.round(6)}, f(x*)={res.fun:.6f}  success={res.success}")
    print(f"  Error vs analytical: {np.linalg.norm(res.x - x_star):.2e}")


def demo_convexity_check():
    print("\n=== Convexity Check via Jensen's Inequality for f(x) = x^2 ===")
    np.random.seed(42)
    f = lambda x: x**2
    n_trials = 10000
    a   = np.random.uniform(-5, 5, n_trials)
    b   = np.random.uniform(-5, 5, n_trials)
    lam = np.random.uniform(0, 1, n_trials)
    lhs = f(lam * a + (1 - lam) * b)       # f(lambda*a + (1-lambda)*b)
    rhs = lam * f(a) + (1 - lam) * f(b)    # lambda*f(a) + (1-lambda)*f(b)
    violations = int(np.sum(lhs > rhs + 1e-12))
    print(f"  f(x) = x^2,  {n_trials} random (a, b, lambda) triples")
    print(f"  Jensen's: f(lam*a+(1-lam)*b) <= lam*f(a)+(1-lam)*f(b)")
    print(f"  Max (LHS - RHS): {np.max(lhs - rhs):.2e}  (should be <= 0 for convex)")
    print(f"  Violations (LHS > RHS + 1e-12): {violations}  (should be 0)")
    print(f"  f(x)=x^2 is {'CONVEX' if violations == 0 else 'NOT convex'}")


def demo_lbfgs_vs_sgd():
    print("\n=== L-BFGS-B vs Gradient Descent on Rosenbrock ===")
    f    = lambda v: (1 - v[0])**2 + 100 * (v[1] - v[0]**2)**2
    grad = lambda v: np.array([
        -2*(1 - v[0]) - 400*v[0]*(v[1] - v[0]**2),
         200*(v[1] - v[0]**2)
    ])
    x0 = np.array([-1.0, 1.0])

    lbfgs_path = [x0.copy()]
    def _cb(xk):
        lbfgs_path.append(xk.copy())
    res = sp_minimize(f, x0, jac=grad, method="L-BFGS-B", callback=_cb,
                      options={"maxiter": 500, "ftol": 1e-15, "gtol": 1e-10})
    lbfgs_path = np.array(lbfgs_path)
    print(f"  L-BFGS-B: {res.nit} iters, x*={res.x.round(6)}, f={res.fun:.2e}, success={res.success}")

    x = x0.copy(); lr = 1e-3; gd_path = [x.copy()]
    for _ in range(5000):
        x = x - lr * grad(x)
        gd_path.append(x.copy())
    gd_path = np.array(gd_path)
    print(f"  GD (lr=1e-3, 5000 steps): x*={x.round(6)}, f={f(x):.2e}")

    xr = np.linspace(-2, 1.5, 300); yr = np.linspace(-1, 2, 300)
    XX, YY = np.meshgrid(xr, yr)
    ZZ = (1 - XX)**2 + 100*(YY - XX**2)**2
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, path, title in zip(
        axes,
        [lbfgs_path, gd_path],
        ["L-BFGS-B path", "GD path (lr=1e-3)"]
    ):
        ax.contour(XX, YY, np.log1p(ZZ), 30, cmap="viridis")
        ax.plot(path[:, 0], path[:, 1], "r-", lw=1, alpha=0.7, label=title)
        ax.scatter(1, 1, color="g", s=100, zorder=5, label="min (1,1)")
        ax.scatter(*x0, color="b", s=80, zorder=5, label="start")
        ax.legend(fontsize=8); ax.set_title(title)
    fig.tight_layout()
    fig.savefig(OUTPUT / "optimization_paths.png", dpi=120, bbox_inches="tight")
    plt.close(fig); print("  Saved: optimization_paths.png")


if __name__ == "__main__":
    demo_convexity()
    optimise_all()
    demo_newton()
    demo_constrained_optimization()
    demo_convexity_check()
    demo_lbfgs_vs_sgd()
