"""
Working Example 2: Inner Product Spaces — Kernels and Similarity Metrics
========================================================================
Demonstrates inner products, Hilbert spaces, kernel trick, Cauchy-Schwarz,
and how different inner products correspond to different geometries in ML.

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

def demo_inner_products():
    print("=== Inner Products ===")
    x = np.array([1., 2., 3.])
    y = np.array([4., 5., 6.])

    # Standard (Euclidean) inner product
    print(f"  <x,y>_E = {np.dot(x, y):.4f}")

    # Weighted inner product <x,y>_W = x^T W y  (W diagonal)
    w = np.diag([2., 0.5, 1.])
    print(f"  <x,y>_W = {x @ w @ y:.4f}")

    # Angle between vectors
    cos_theta = np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
    theta = np.degrees(np.arccos(np.clip(cos_theta, -1, 1)))
    print(f"  Angle = {theta:.2f}°  (cos = {cos_theta:.4f})")

    # Cauchy-Schwarz: |<x,y>| <= ||x||·||y||
    lhs = abs(np.dot(x, y))
    rhs = np.linalg.norm(x) * np.linalg.norm(y)
    print(f"  Cauchy-Schwarz: |<x,y>|={lhs:.4f} <= ||x||||y||={rhs:.4f}  [OK]={lhs<=rhs+1e-9}")

def demo_kernel_trick():
    print("\n=== Kernel Trick ===")
    # Linear kernel
    def k_linear(x, y): return np.dot(x, y)

    # Polynomial kernel: k(x,y) = (x·y + c)^d
    def k_poly(x, y, c=1, d=2): return (np.dot(x, y) + c) ** d

    # RBF (Gaussian) kernel: k(x,y) = exp(-gamma||x-y||²)
    def k_rbf(x, y, gamma=0.5):
        diff = x - y
        return float(np.exp(-gamma * np.dot(diff, diff)))

    x = np.array([1., 2.])
    y = np.array([3., 1.])

    print(f"  k_linear(x,y) = {k_linear(x,y):.4f}")
    print(f"  k_poly(x,y)   = {k_poly(x,y):.4f}  (d=2, c=1)")
    print(f"  k_rbf(x,y)    = {k_rbf(x,y):.6f}")

    # Kernel matrix (Gram matrix) on 5 points
    X = np.random.randn(5, 2)
    K = np.array([[k_rbf(X[i], X[j]) for j in range(5)] for i in range(5)])
    print(f"\n  RBF Gram matrix (5×5):\n{K.round(4)}")
    print(f"  Symmetric: {np.allclose(K, K.T)}")
    print(f"  PSD (all eigenvalues >= 0): {all(np.linalg.eigvalsh(K) >= -1e-10)}")

def demo_orthogonal_projection():
    print("\n=== Orthogonal Projection (Hilbert space) ===")
    # Project a function (signal) onto a subspace (Fourier basis)
    t = np.linspace(0, 2*np.pi, 100)
    signal = np.sin(t) + 0.5 * np.cos(2*t) + 0.1 * np.random.randn(100)
    # Basis: [1, sin(t), cos(t), sin(2t), cos(2t)]
    basis = np.column_stack([np.ones_like(t), np.sin(t), np.cos(t), np.sin(2*t), np.cos(2*t)])
    # L2 inner product ~ dot product / n
    # Projection: coefficients c = (B^T B)^{-1} B^T f
    c = np.linalg.solve(basis.T @ basis, basis.T @ signal)
    signal_hat = basis @ c
    residual = signal - signal_hat
    print(f"  Fourier coefficients: {c.round(4)}")
    print(f"  Residual norm: {np.linalg.norm(residual):.4f}")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t, signal, alpha=0.5, label="noisy signal")
    ax.plot(t, signal_hat, lw=2, label="projection onto Fourier basis")
    ax.legend(); ax.set_title("Hilbert space projection")
    fig.savefig(OUTPUT / "inner_product_projection.png", dpi=120, bbox_inches="tight")
    plt.close(fig); print(f"  Saved: inner_product_projection.png")

if __name__ == "__main__":
    demo_inner_products()
    demo_kernel_trick()
    demo_orthogonal_projection()
