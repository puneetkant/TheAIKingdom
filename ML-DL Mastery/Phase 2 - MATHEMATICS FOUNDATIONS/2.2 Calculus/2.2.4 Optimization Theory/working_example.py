"""
Working Example: Optimization Theory
Covers convexity, gradient descent variants, momentum, Adam,
Lagrange multipliers, KKT conditions, and learning rate schedules.
"""
import numpy as np
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_optim")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── 1. Convexity ───────────────────────────────────────────────────────────────
def convexity():
    print("=== Convexity ===")
    # Jensen's inequality: f(E[x]) ≤ E[f(x)] for convex f
    f = lambda x: x**2    # convex
    g = lambda x: -x**2   # concave

    xs = np.array([1., 3.])
    t  = 0.5
    xm = t*xs[0] + (1-t)*xs[1]   # midpoint

    print(f"  f(x)=x² (convex):  f(midpoint)={f(xm):.2f}  ≤  t·f(x1)+(1-t)·f(x2)={t*f(xs[0])+(1-t)*f(xs[1]):.2f}")
    print(f"  g(x)=-x² (concave): g(midpoint)={g(xm):.2f}  ≥  t·g(x1)+(1-t)·g(x2)={t*g(xs[0])+(1-t)*g(xs[1]):.2f}")

    # Second-order condition: f convex ↔ f''(x) ≥ 0
    print("\n  Second-order convexity checks via Hessian eigenvalues:")
    funcs = {
        "x²+y²":   (lambda v: v[0]**2 + v[1]**2,  np.diag([2., 2.])),
        "x²-y²":   (lambda v: v[0]**2 - v[1]**2,  np.diag([2.,-2.])),
        "x²+xy+y²":(lambda v: v[0]**2+v[0]*v[1]+v[1]**2, np.array([[2.,1.],[1.,2.]])),
    }
    for name, (_, H) in funcs.items():
        eigs = np.linalg.eigvalsh(H)
        cvx  = "convex" if np.all(eigs>=0) else ("concave" if np.all(eigs<=0) else "non-convex")
        print(f"    {name:<18}: H eigs={np.round(eigs,2)}  → {cvx}")


# ── 2. Gradient Descent variants ──────────────────────────────────────────────
def gradient_descent(f, grad_f, x0, lr, n_iter, name="GD"):
    x = x0.copy().astype(float)
    history = [f(x)]
    for _ in range(n_iter):
        x -= lr * grad_f(x)
        history.append(f(x))
    return x, history


def momentum_gd(f, grad_f, x0, lr, n_iter, beta=0.9):
    x = x0.copy().astype(float)
    v = np.zeros_like(x)
    history = [f(x)]
    for _ in range(n_iter):
        v = beta*v + (1-beta)*grad_f(x)
        x -= lr * v
        history.append(f(x))
    return x, history


def adam(f, grad_f, x0, lr=0.01, n_iter=200, beta1=0.9, beta2=0.999, eps=1e-8):
    x  = x0.copy().astype(float)
    m  = np.zeros_like(x)
    v  = np.zeros_like(x)
    history = [f(x)]
    for t in range(1, n_iter+1):
        g = grad_f(x)
        m = beta1*m + (1-beta1)*g
        v = beta2*v + (1-beta2)*g**2
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        x -= lr * m_hat / (np.sqrt(v_hat) + eps)
        history.append(f(x))
    return x, history


def optimizer_comparison():
    print("\n=== Optimizer Comparison on Rosenbrock Function ===")
    # f(x,y) = (1-x)² + 100(y-x²)²  minimum at (1,1)
    f    = lambda v: (1-v[0])**2 + 100*(v[1]-v[0]**2)**2
    grad = lambda v: np.array([
        -2*(1-v[0]) - 400*v[0]*(v[1]-v[0]**2),
         200*(v[1]-v[0]**2)])

    x0    = np.array([-0.5, 0.5])
    n_iter = 2000

    results = {
        "GD (lr=1e-3)":       gradient_descent(f, grad, x0, 1e-3, n_iter),
        "Momentum (lr=1e-3)": momentum_gd(f, grad, x0, 1e-3, n_iter),
        "Adam (lr=1e-2)":     adam(f, grad, x0, lr=1e-2, n_iter=n_iter),
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, (x_final, hist) in results.items():
        ax.semilogy(hist, label=label)
        print(f"  {label:<25}: x*={np.round(x_final,4)}  f*={f(x_final):.6f}")
    ax.set(xlabel="Iteration", ylabel="Loss (log)", title="Optimizers on Rosenbrock")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "optimizer_comparison.png")
    plt.savefig(path, dpi=90)
    plt.close()
    print(f"  Saved: {path}")


# ── 3. Learning-rate schedules ────────────────────────────────────────────────
def lr_schedules():
    print("\n=== Learning-Rate Schedules ===")
    n = 100
    iters = np.arange(n)
    lr0   = 0.1

    schedules = {
        "constant":      np.full(n, lr0),
        "step decay":    lr0 * (0.5 ** (iters // 20)),
        "exponential":   lr0 * np.exp(-0.05 * iters),
        "cosine anneal": lr0 * 0.5 * (1 + np.cos(np.pi * iters / n)),
        "1/t decay":     lr0 / (1 + 0.1 * iters),
    }

    fig, ax = plt.subplots(figsize=(8, 4))
    for name, lrs in schedules.items():
        ax.plot(iters, lrs, label=name)
        print(f"  {name:<16}: lr[0]={lrs[0]:.4f}  lr[50]={lrs[50]:.4f}  lr[99]={lrs[99]:.4f}")
    ax.set(xlabel="Epoch", ylabel="Learning rate", title="LR Schedules")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "lr_schedules.png")
    plt.savefig(path, dpi=90)
    plt.close()
    print(f"  Saved: {path}")


# ── 4. Lagrange multipliers ───────────────────────────────────────────────────
def lagrange_multipliers():
    print("\n=== Lagrange Multipliers ===")
    print("  Maximize f(x,y) = xy  subject to g(x,y) = x+y-1 = 0")
    print("  Stationarity:  ∇f = λ∇g  →  (y, x) = λ(1, 1)  →  y=x")
    print("  Constraint:    x+y=1  →  2x=1  →  x=y=0.5")
    print("  Maximum value: f(0.5,0.5) = 0.25")
    # Verify numerically with scipy
    from scipy.optimize import minimize
    res = minimize(lambda v: -v[0]*v[1],
                   [0.3, 0.7],
                   method='SLSQP',
                   constraints={'type':'eq', 'fun': lambda v: v[0]+v[1]-1})
    print(f"  scipy result:  x*={np.round(res.x,4)}  f*={-res.fun:.4f}")


# ── 5. KKT conditions (inequality constraints) ────────────────────────────────
def kkt_conditions():
    print("\n=== KKT Conditions (Inequality Constraints) ===")
    print("  min f(x) = x² + y² subject to x+y ≥ 1")
    print("  Reformulate: min x²+y² s.t. 1-x-y ≤ 0")
    print("  KKT conditions:")
    print("    ∇f + λ∇g = 0  →  2x=λ, 2y=λ  →  x=y")
    print("    λ ≥ 0  (dual feasibility)")
    print("    λ·g = 0  (complementary slackness)")
    print("    At minimum: 2x=1 → x=y=0.5, λ=1, f*=0.5")
    from scipy.optimize import minimize
    res = minimize(lambda v: v[0]**2 + v[1]**2,
                   [0.0, 0.0],
                   method='SLSQP',
                   constraints={'type':'ineq', 'fun': lambda v: v[0]+v[1]-1})
    print(f"\n  scipy result:  x*={np.round(res.x,4)}  f*={res.fun:.4f}")


# ── 6. Convex vs non-convex landscape ────────────────────────────────────────
def loss_landscape():
    print("\n=== Loss Landscape Visualisation ===")
    x = np.linspace(-3, 3, 300)
    y = np.linspace(-3, 3, 300)
    X, Y = np.meshgrid(x, y)

    Z_convex    = X**2 + Y**2                              # bowl
    Z_nonconvex = np.sin(3*X) * np.cos(3*Y) + 0.2*(X**2 + Y**2)   # many local minima

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, Z, title in zip(axes,
                              [Z_convex, Z_nonconvex],
                              ["Convex: x²+y²", "Non-convex: sin(3x)cos(3y)+0.2(x²+y²)"]):
        c = ax.contourf(X, Y, Z, levels=30, cmap="RdYlBu_r")
        fig.colorbar(c, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("x"); ax.set_ylabel("y")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "loss_landscape.png")
    plt.savefig(path, dpi=90)
    plt.close()
    print(f"  Saved: {path}")


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    convexity()
    optimizer_comparison()
    lr_schedules()
    lagrange_multipliers()
    kkt_conditions()
    loss_landscape()
