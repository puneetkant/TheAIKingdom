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

def demo_gram_schmidt():
    print("\n=== Gram-Schmidt Orthogonalisation ===")
    np.random.seed(10)
    # Start with 3 linearly independent vectors
    V = np.random.randn(4, 3)  # columns are vectors in R^4
    Q = np.zeros_like(V, dtype=float)
    for i in range(V.shape[1]):
        q = V[:, i].astype(float)
        for j in range(i):
            q -= np.dot(q, Q[:, j]) * Q[:, j]
        Q[:, i] = q / np.linalg.norm(q)
    print(f"  Q^T Q (should be I_3x3):\n{(Q.T @ Q).round(10)}")
    print(f"  Orthonormal: {np.allclose(Q.T @ Q, np.eye(3))}")


def demo_kernel_comparison():
    print("\n=== Kernel Comparison on 2D Grid ===")
    np.random.seed(0)
    # Show how different kernels change the similarity landscape
    x0 = np.array([0.0, 0.0])
    grid = np.array([[i/5, j/5] for i in range(-5, 6) for j in range(-5, 6)])

    def k_linear(x, y): return np.dot(x, y)
    def k_rbf(x, y, g=1.0): return np.exp(-g * np.dot(x-y, x-y))
    def k_poly(x, y, d=2): return (np.dot(x, y) + 1) ** d

    for name, kfn in [("Linear", k_linear), ("RBF", k_rbf), ("Poly d=2", k_poly)]:
        sims = np.array([kfn(x0, g) for g in grid])
        print(f"  {name}: min={sims.min():.3f}  max={sims.max():.3f}  mean={sims.mean():.3f}")

    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    for ax, (name, kfn) in zip(axes, [("Linear", k_linear), ("RBF", k_rbf), ("Poly d=2", k_poly)]):
        sims = np.array([kfn(x0, g) for g in grid]).reshape(11, 11)
        im = ax.imshow(sims, cmap="RdYlGn", origin="lower")
        ax.set_title(f"{name} Kernel (center=origin)")
        plt.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.savefig(OUTPUT / "kernel_comparison.png", dpi=100, bbox_inches="tight")
    plt.close(fig); print("  Saved: kernel_comparison.png")


if __name__ == "__main__":
    demo_inner_products()
    demo_kernel_trick()
    demo_orthogonal_projection()
    demo_gram_schmidt()
    demo_kernel_comparison()
