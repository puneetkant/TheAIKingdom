"""
Working Example: Optimizers (Detailed)
Covers SGD, Momentum, NAG, Adagrad, RMSProp, Adam, AdamW, AMSGrad,
Lion, from-scratch implementations, and convergence comparison.
"""
import numpy as np
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_optimizers")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Test loss surface ─────────────────────────────────────────────────────────
def rosenbrock(x, y):
    """Non-convex Rosenbrock function; minimum at (1,1)."""
    return (1-x)**2 + 100*(y-x**2)**2

def rosenbrock_grad(x, y):
    dL_dx = -2*(1-x) - 400*x*(y-x**2)
    dL_dy = 200*(y-x**2)
    return np.array([dL_dx, dL_dy])


# ── Optimizer implementations ─────────────────────────────────────────────────
def sgd_step(theta, grad, state, lr=0.001, **_):
    theta = theta - lr * grad
    return theta, state

def momentum_step(theta, grad, state, lr=0.001, beta=0.9, **_):
    v = beta * state.get("v", np.zeros_like(theta)) - lr * grad
    state["v"] = v
    return theta + v, state

def nag_step(theta, grad_fn, state, lr=0.001, beta=0.9, **_):
    v     = state.get("v", np.zeros_like(theta))
    theta_look = theta + beta * v
    grad  = grad_fn(*theta_look)
    v     = beta*v - lr*grad
    state["v"] = v
    return theta + v, state

def adagrad_step(theta, grad, state, lr=0.01, eps=1e-8, **_):
    G  = state.get("G", np.zeros_like(theta)) + grad**2
    state["G"] = G
    return theta - lr * grad / (np.sqrt(G) + eps), state

def rmsprop_step(theta, grad, state, lr=0.001, beta=0.9, eps=1e-8, **_):
    v  = beta * state.get("v", np.zeros_like(theta)) + (1-beta)*grad**2
    state["v"] = v
    return theta - lr * grad / (np.sqrt(v) + eps), state

def adam_step(theta, grad, state, lr=0.001, b1=0.9, b2=0.999, eps=1e-8, **_):
    t  = state.get("t", 0) + 1
    m  = b1 * state.get("m", np.zeros_like(theta)) + (1-b1)*grad
    v  = b2 * state.get("v", np.zeros_like(theta)) + (1-b2)*grad**2
    m_hat = m / (1 - b1**t)
    v_hat = v / (1 - b2**t)
    state.update({"t": t, "m": m, "v": v})
    return theta - lr * m_hat / (np.sqrt(v_hat) + eps), state

def adamw_step(theta, grad, state, lr=0.001, b1=0.9, b2=0.999, eps=1e-8, wd=0.01, **_):
    t  = state.get("t", 0) + 1
    m  = b1 * state.get("m", np.zeros_like(theta)) + (1-b1)*grad
    v  = b2 * state.get("v", np.zeros_like(theta)) + (1-b2)*grad**2
    m_hat = m / (1 - b1**t)
    v_hat = v / (1 - b2**t)
    state.update({"t": t, "m": m, "v": v})
    # Decoupled weight decay
    return theta * (1 - lr*wd) - lr * m_hat / (np.sqrt(v_hat) + eps), state


# ── 1. SGD ────────────────────────────────────────────────────────────────────
def sgd_demo():
    print("=== Stochastic Gradient Descent (SGD) ===")
    print("  θ ← θ - η · ∇L(θ)")
    print("  Batch GD: use all data  |  SGD: 1 sample  |  Mini-batch: batch_size")
    print()
    print("  Batch GD:   stable convergence, expensive per step")
    print("  SGD:        noisy updates, fast, can escape local minima")
    print("  Mini-batch: best of both worlds (typical: 32–512)")


# ── 2. Momentum ───────────────────────────────────────────────────────────────
def momentum_demo():
    print("\n=== Momentum ===")
    print("  v ← βv - η·∇L    θ ← θ + v")
    print("  Accelerates in consistent directions; dampens oscillations")
    print("  β=0.9: velocity decays as 1/(1-β)=10 steps memory")
    print()
    print("  Effective step size ≈ η/(1-β)  e.g.  0.01/(1-0.9) = 0.1")


# ── 3. Adagrad, RMSProp ───────────────────────────────────────────────────────
def adaptive_lr_demo():
    print("\n=== Adaptive Learning Rate Methods ===")
    print("  Adagrad: accumulates squared gradients G += g²")
    print("    θ ← θ - η/√(G+ε) · g  (LR shrinks monotonically → stops learning)")
    print()
    print("  RMSProp: exponential moving average of G:")
    print("    v ← β·v + (1-β)·g²   θ ← θ - η/√(v+ε) · g")
    print("    Fixes Adagrad's decaying LR problem")


# ── 4. Adam ───────────────────────────────────────────────────────────────────
def adam_demo():
    print("\n=== Adam (Adaptive Moment Estimation) ===")
    print("  Combines Momentum + RMSProp:")
    print("  m ← β1·m + (1-β1)·g      (1st moment / momentum)")
    print("  v ← β2·v + (1-β2)·g²     (2nd moment / adaptive LR)")
    print("  m̂ = m/(1-β1^t)  v̂ = v/(1-β2^t)   (bias correction)")
    print("  θ ← θ - η·m̂/√(v̂+ε)")
    print()
    print("  Defaults: β1=0.9  β2=0.999  ε=1e-8  η=0.001")
    print("  AdamW: decoupled weight decay θ ← θ·(1-η·λ) - η·m̂/√(v̂+ε)")


# ── 5. Convergence comparison on Rosenbrock ───────────────────────────────────
def convergence_comparison():
    print("\n=== Convergence Comparison (Rosenbrock function) ===")
    print("  f(x,y) = (1-x)² + 100(y-x²)²   minimum at (1,1)  f=0")

    optimizers = {
        "SGD (lr=0.001)":      (lambda th, g, s: sgd_step(th, g, s, lr=0.001),  True),
        "Momentum (lr=0.001)": (lambda th, g, s: momentum_step(th, g, s, lr=0.001), True),
        "Adagrad (lr=0.1)":   (lambda th, g, s: adagrad_step(th, g, s, lr=0.1),  True),
        "RMSProp (lr=0.01)":  (lambda th, g, s: rmsprop_step(th, g, s, lr=0.01),  True),
        "Adam (lr=0.01)":     (lambda th, g, s: adam_step(th, g, s, lr=0.01),    True),
        "AdamW (lr=0.01)":    (lambda th, g, s: adamw_step(th, g, s, lr=0.01),   True),
    }

    n_steps = 2000
    theta0  = np.array([-1.5, 0.5])
    histories = {}

    for name, (step_fn, _) in optimizers.items():
        theta = theta0.copy()
        state = {}
        losses = []
        for _ in range(n_steps):
            g    = rosenbrock_grad(*theta)
            theta, state = step_fn(theta, g, state)
            losses.append(rosenbrock(*theta))
        final_loss = losses[-1]
        dist = np.sqrt((theta[0]-1)**2 + (theta[1]-1)**2)
        print(f"  {name:<30}: final_loss={final_loss:.4f}  dist_to_opt={dist:.4f}")
        histories[name] = losses

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, losses in histories.items():
        ax.semilogy(losses, lw=2, label=name)
    ax.set(xlabel="Step", ylabel="Loss (log scale)", title="Optimizer Convergence (Rosenbrock)")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "optimizer_convergence.png")
    plt.savefig(path, dpi=90); plt.close()
    print(f"\n  Convergence plot saved: {path}")


# ── 6. Optimizer selection guide ─────────────────────────────────────────────
def optimizer_guide():
    print("\n=== Optimizer Selection Guide ===")
    print(f"  {'Optimizer':<20} {'Best for':<40} {'Notes'}")
    rows = [
        ("SGD (no momentum)", "Convex problems",               "Slow; rarely used alone"),
        ("SGD + Momentum",    "CNNs, ResNets (with schedule)", "Often best final acc with tuning"),
        ("Adagrad",           "Sparse gradients (NLP)",        "LR shrinks → stops learning"),
        ("RMSProp",           "Non-stationary problems, RNNs", "Good default for RNNs"),
        ("Adam",              "General default",               "Fast convergence, often suboptimal"),
        ("AdamW",             "Transformers, large models",    "Decoupled weight decay"),
        ("AMSGrad",           "Adam with convergence guarantee","Slower than Adam in practice"),
        ("Lion",              "Large language models",         "Momentum + sign update"),
    ]
    for r in rows:
        print(f"  {r[0]:<20} {r[1]:<40} {r[2]}")


if __name__ == "__main__":
    sgd_demo()
    momentum_demo()
    adaptive_lr_demo()
    adam_demo()
    convergence_comparison()
    optimizer_guide()
