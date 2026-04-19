"""
Working Example 2: Multivariable Calculus — Gradients, Jacobian, Hessian
=========================================================================
Partial derivatives, numerical Jacobian/Hessian, gradient descent on 2D surfaces,
saddle points vs minima, contour plots.

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

def numerical_gradient(f, x, h=1e-6):
    """Central difference gradient for f: R^n -> R."""
    grad = np.zeros_like(x, dtype=float)
    for i in range(len(x)):
        xp, xm = x.copy(), x.copy()
        xp[i] += h; xm[i] -= h
        grad[i] = (f(xp) - f(xm)) / (2 * h)
    return grad

def numerical_jacobian(F, x, h=1e-6):
    """Jacobian of F: R^n -> R^m  (m x n matrix)."""
    m = len(F(x))
    J = np.zeros((m, len(x)))
    for i in range(len(x)):
        xp, xm = x.copy(), x.copy()
        xp[i] += h; xm[i] -= h
        J[:, i] = (F(xp) - F(xm)) / (2 * h)
    return J

def numerical_hessian(f, x, h=1e-5):
    """Approximate Hessian via finite differences."""
    n = len(x)
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            xi, xj = x.copy(), x.copy()
            xij = x.copy()
            xi[i] += h; xj[j] += h; xij[i] += h; xij[j] += h
            H[i,j] = (f(xij) - f(xi) - f(xj) + f(x)) / h**2
    return H

def demo_gradient():
    print("=== Gradient of f(x,y) = sin(x)*e^(-y²) ===")
    f  = lambda v: np.sin(v[0]) * np.exp(-v[1]**2)
    df_x = lambda v: np.cos(v[0]) * np.exp(-v[1]**2)
    df_y = lambda v: -2 * v[1] * np.sin(v[0]) * np.exp(-v[1]**2)

    x0 = np.array([1.0, 0.5])
    g_num  = numerical_gradient(f, x0)
    g_true = np.array([df_x(x0), df_y(x0)])
    print(f"  Numerical gradient:  {g_num.round(6)}")
    print(f"  Analytic gradient:   {g_true.round(6)}")
    print(f"  Max error: {np.max(np.abs(g_num - g_true)):.2e}")

def demo_jacobian():
    print("\n=== Jacobian of F: R²->R³ ===")
    # F(x,y) = [x², sin(x+y), e^x]
    F = lambda v: np.array([v[0]**2, np.sin(v[0]+v[1]), np.exp(v[0])])
    x0 = np.array([1.0, 2.0])
    J = numerical_jacobian(F, x0)
    print(f"  J(1,2) ~=\n{J.round(4)}")
    # Analytic: [[2x, 0], [cos(x+y), cos(x+y)], [e^x, 0]]
    J_true = np.array([[2*x0[0], 0],
                       [np.cos(x0.sum()), np.cos(x0.sum())],
                       [np.exp(x0[0]), 0]])
    print(f"  Analytic:\n{J_true.round(4)}")
    print(f"  Max error: {np.max(np.abs(J - J_true)):.2e}")

def demo_hessian():
    print("\n=== Hessian of f(x,y) = x²+2y²+xy ===")
    f = lambda v: v[0]**2 + 2*v[1]**2 + v[0]*v[1]
    x0 = np.array([1.0, 2.0])
    H = numerical_hessian(f, x0)
    H_true = np.array([[2., 1.], [1., 4.]])
    print(f"  Numerical H:\n{H.round(4)}")
    print(f"  Analytic H:\n{H_true}")
    eigs = np.linalg.eigvalsh(H_true)
    print(f"  Eigenvalues: {eigs}  -> {'min' if all(eigs>0) else 'saddle'}")

def demo_contour_gd():
    print("\n=== Gradient Descent on 2D Surface ===")
    # Rosenbrock banana: f(x,y)=(1-x)²+100(y-x²)²
    f = lambda v: (1-v[0])**2 + 100*(v[1]-v[0]**2)**2
    x = np.array([-1.0, 1.0]); lr = 1e-3; history = [x.copy()]
    for _ in range(3000):
        g = numerical_gradient(f, x)
        x = x - lr * g
        history.append(x.copy())
    print(f"  GD minimum: {x.round(4)}  (true: [1,1])")
    print(f"  f(x*): {f(x):.6f}")

    path = np.array(history)
    xr = np.linspace(-2, 2, 300); yr = np.linspace(-1, 3, 300)
    XX, YY = np.meshgrid(xr, yr)
    ZZ = (1-XX)**2 + 100*(YY-XX**2)**2

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.contour(XX, YY, np.log1p(ZZ), 30, cmap="viridis")
    ax.plot(path[:,0], path[:,1], "r-", lw=0.7, alpha=0.5)
    ax.scatter(*x, color="r", zorder=5, label="GD solution")
    ax.scatter(1, 1, color="g", zorder=5, label="true min (1,1)")
    ax.legend(); ax.set_title("Gradient Descent on Rosenbrock")
    fig.savefig(OUTPUT / "rosenbrock_gd.png", dpi=120, bbox_inches="tight")
    plt.close(fig); print(f"  Saved: rosenbrock_gd.png")

if __name__ == "__main__":
    demo_gradient()
    demo_jacobian()
    demo_hessian()
    demo_contour_gd()
