"""
Working Example 2: Matrix Decompositions — LU, QR, Cholesky
============================================================
Demonstrates LU (scipy), QR (numpy), Cholesky decompositions with
ML applications: QR for least squares, Cholesky for covariance sampling.

Run:  python working_example2.py
"""
from pathlib import Path
try:
    import numpy as np
    from numpy.linalg import qr, cholesky, solve
except ImportError:
    raise SystemExit("pip install numpy")

DATA = Path(__file__).parent / "data"
DATA.mkdir(exist_ok=True)

def demo_qr():
    print("=== QR Decomposition ===")
    A = np.array([[1.,2.,3.],[4.,5.,6.],[7.,8.,10.],[1.,0.,2.]])
    Q, R = qr(A)
    print(f"  A: {A.shape}  Q: {Q.shape}  R: {R.shape}")
    print(f"  Q orthonormal check (Q^T Q ≈ I): max off-diag={np.max(np.abs(Q.T@Q - np.eye(Q.shape[1]))):.2e}")
    print(f"  Reconstruction A ≈ QR: err={np.linalg.norm(A - Q@R):.2e}")
    # QR for least squares: min ‖Ax-b‖  →  Rx = Q^T b
    b = np.array([14., 32., 50., 3.])
    x = solve(R, Q.T @ b)
    print(f"  QR least-squares x = {x.round(4)}")

def demo_cholesky():
    print("\n=== Cholesky Decomposition (SPD matrices) ===")
    # Positive definite covariance matrix
    import urllib.request, csv
    import random; random.seed(99)
    rows = []
    for _ in range(100):
        rows.append([random.gauss(0,1), random.gauss(0,1), random.gauss(0,1)])
    X = np.array(rows)
    cov = X.T @ X / len(X)       # symmetric positive semi-definite

    # Regularise to make positive definite
    cov += 1e-6 * np.eye(3)
    L = cholesky(cov)
    print(f"  Cov:\n{cov.round(4)}")
    print(f"  L (lower triangular):\n{L.round(4)}")
    print(f"  L @ L.T ≈ Cov: err={np.linalg.norm(cov - L@L.T):.2e}")

    # Sample from multivariate normal: x = L @ z, z ~ N(0,I)
    z = np.random.randn(5, 3)
    samples = z @ L.T
    print(f"  Samples from N(0, Cov):\n{samples.round(4)}")

def demo_lu_scipy():
    print("\n=== LU Decomposition (scipy) ===")
    try:
        from scipy.linalg import lu
        A = np.array([[2.,1.,-1.],[-3.,-1.,2.],[-2.,1.,2.]])
        P, L, U = lu(A)
        print(f"  L:\n{L.round(4)}")
        print(f"  U:\n{U.round(4)}")
        print(f"  PLU ≈ A: err={np.linalg.norm(P@L@U - A):.2e}")
    except ImportError:
        # Fallback: NumPy doesn't have LU directly, use SVD
        A = np.array([[2.,1.,-1.],[-3.,-1.,2.],[-2.,1.,2.]])
        print("  (scipy not available — using np.linalg.svd)")
        U, s, Vt = np.linalg.svd(A)
        print(f"  Singular values: {s.round(4)}")

if __name__ == "__main__":
    demo_qr()
    demo_cholesky()
    demo_lu_scipy()
