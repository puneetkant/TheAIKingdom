"""
Working Example 2: Numerical Linear Algebra — Stability, Sparse Matrices, Iterative Solvers
=============================================================================================
Covers: condition numbers, numerical stability of QR vs normal equations,
sparse matrix representation (CSR), conjugate gradient iteration, Jacobi method.

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

def demo_condition_and_stability():
    print("=== Condition Number and Numerical Stability ===")
    np.random.seed(42)
    n = 100
    # Well-conditioned system
    A_good = np.random.randn(n, 10)
    # Ill-conditioned: Hilbert-like matrix
    A_bad = np.array([[1./(i+j+1) for j in range(10)] for i in range(n)])
    for name, A in [("random (well)", A_good), ("Hilbert-like (ill)", A_bad)]:
        _, s, _ = np.linalg.svd(A, full_matrices=False)
        cond = s[0] / (s[-1] + 1e-15)
        b = A @ np.ones(10) + 1e-8 * np.random.randn(n)
        # Normal equations
        x_ne = np.linalg.solve(A.T @ A, A.T @ b)
        # QR
        Q, R = np.linalg.qr(A)
        x_qr = np.linalg.solve(R, Q.T @ b)
        err_ne = np.linalg.norm(x_ne - np.ones(10))
        err_qr = np.linalg.norm(x_qr - np.ones(10))
        print(f"\n  {name}: cond={cond:.3e}")
        print(f"    Normal eqn error: {err_ne:.3e}")
        print(f"    QR error:         {err_qr:.3e}")

def demo_sparse():
    print("\n=== Sparse Matrices (COO / CSR via dict) ===")
    # Simulate a sparse adjacency matrix
    n = 8
    edges = [(0,1),(0,2),(1,3),(2,3),(3,4),(4,5),(5,6),(6,7),(7,0)]
    # CSR-style: store non-zeros
    rows_idx, cols_idx, vals = zip(*[(i,j,1.0) for i,j in edges])
    data = {"rows": list(rows_idx), "cols": list(cols_idx), "vals": list(vals), "shape": (n,n)}
    print(f"  Sparse graph: {len(edges)} edges, {n}×{n} matrix")
    print(f"  Density: {len(edges)/n**2:.3f}")
    # Convert to dense for demo
    A = np.zeros((n, n))
    for r,c,v in zip(data["rows"], data["cols"], data["vals"]):
        A[r,c] = v
    print(f"  Dense form:\n{A.astype(int)}")

def demo_jacobi(tol=1e-6, max_iter=1000):
    print("\n=== Jacobi Iterative Solver ===")
    # Diagonally dominant system (guaranteed convergence)
    A = np.array([[10.,-1., 2.],
                  [-1.,11.,-1.],
                  [ 2.,-1.,10.]])
    b = np.array([6., 25., -11.])
    x_true = np.linalg.solve(A, b)

    D = np.diag(np.diag(A))
    R = A - D
    x = np.zeros_like(b)
    errors = []
    for k in range(max_iter):
        x_new = np.linalg.solve(D, b - R @ x)
        err = np.linalg.norm(x_new - x)
        errors.append(err)
        x = x_new
        if err < tol:
            print(f"  Converged in {k+1} iterations")
            break

    print(f"  Solution:   {x.round(6)}")
    print(f"  True soln:  {x_true.round(6)}")
    print(f"  Final error: {np.linalg.norm(x - x_true):.3e}")

    fig, ax = plt.subplots(figsize=(6,4))
    ax.semilogy(errors, c="steelblue")
    ax.set_xlabel("Iteration"); ax.set_ylabel("||Deltax||")
    ax.set_title("Jacobi convergence")
    fig.savefig(OUTPUT / "jacobi_convergence.png", dpi=120, bbox_inches="tight")
    plt.close(fig); print(f"  Saved: jacobi_convergence.png")

if __name__ == "__main__":
    demo_condition_and_stability()
    demo_sparse()
    demo_jacobi()
