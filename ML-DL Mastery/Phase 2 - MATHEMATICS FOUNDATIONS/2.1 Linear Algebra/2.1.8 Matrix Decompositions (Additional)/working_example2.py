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
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit("pip install numpy matplotlib")

DATA = Path(__file__).parent / "data"
DATA.mkdir(exist_ok=True)

def demo_qr():
    print("=== QR Decomposition ===")
    A = np.array([[1.,2.,3.],[4.,5.,6.],[7.,8.,10.],[1.,0.,2.]])
    Q, R = qr(A)
    print(f"  A: {A.shape}  Q: {Q.shape}  R: {R.shape}")
    print(f"  Q orthonormal check (Q^T Q ~= I): max off-diag={np.max(np.abs(Q.T@Q - np.eye(Q.shape[1]))):.2e}")
    print(f"  Reconstruction A ~= QR: err={np.linalg.norm(A - Q@R):.2e}")
    # QR for least squares: min ||Ax-b||  ->  Rx = Q^T b
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
    print(f"  L @ L.T ~= Cov: err={np.linalg.norm(cov - L@L.T):.2e}")

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
        print(f"  PLU ~= A: err={np.linalg.norm(P@L@U - A):.2e}")
    except ImportError:
        # Fallback: NumPy doesn't have LU directly, use SVD
        A = np.array([[2.,1.,-1.],[-3.,-1.,2.],[-2.,1.,2.]])
        print("  (scipy not available — using np.linalg.svd)")
        U, s, Vt = np.linalg.svd(A)
        print(f"  Singular values: {s.round(4)}")

def demo_schur_decomposition():
    print("\n=== Schur Decomposition (A = Z T Z^H) ===")
    try:
        from scipy.linalg import schur
        A = np.array([[4., 3., 2.], [0., 2., 1.], [1., 1., 3.]], dtype=float)
        T, Z = schur(A)
        print(f"  A =\n{A}")
        print(f"  Schur T (quasi-upper-triangular):\n{T.round(4)}")
        err_unitary = np.linalg.norm(Z.T @ Z - np.eye(3))
        err_recon   = np.linalg.norm(Z @ T @ Z.T - A)
        print(f"  Z^T Z ~= I: err={err_unitary:.2e}")
        print(f"  Z T Z^T ~= A: err={err_recon:.2e}")
        print(f"  Eigenvalues from diag(T): {np.diag(T).round(4)}")
        print(f"  Direct eigenvalues:       {np.sort(np.linalg.eigvals(A).real).round(4)}")
    except ImportError:
        print("  scipy not available -- skipping Schur demo")


def demo_polar_decomposition():
    print("\n=== Polar Decomposition (A = U P) ===")
    # A = U P  where U is orthogonal and P is symmetric PSD
    # Via SVD: A = V S W^T  =>  U = V W^T,  P = W S W^T
    np.random.seed(7)
    A = np.random.randn(3, 3)
    V, s, Wt = np.linalg.svd(A)
    U_pol = V @ Wt                       # orthogonal factor
    P_pol = Wt.T @ np.diag(s) @ Wt      # symmetric PSD factor
    err = np.linalg.norm(U_pol @ P_pol - A)
    print(f"  A =\n{A.round(4)}")
    print(f"  U det = {np.linalg.det(U_pol):.4f}  (should be +/-1)")
    print(f"  P eigenvalues (all >= 0): {np.linalg.eigvalsh(P_pol).round(4)}")
    print(f"  U @ P ~= A: err={err:.2e}")
    # Save singular value bar chart
    out = Path(__file__).parent / "output"
    out.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(range(len(s)), s, color="steelblue")
    ax.set_title("Polar Decomposition: Singular Values of A")
    ax.set_xlabel("Index"); ax.set_ylabel("Singular Value")
    fig.savefig(out / "polar_singular_values.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: polar_singular_values.png")


def demo_svd_low_rank_approx():
    print("\n=== SVD Low-Rank Approximation ===")
    # Approximate A by keeping only top-k singular values/vectors
    np.random.seed(0)
    # Create a rank-3 matrix with noise
    m, n, r = 8, 6, 3
    U0 = np.linalg.qr(np.random.randn(m, r))[0]
    V0 = np.linalg.qr(np.random.randn(n, r))[0]
    S0 = np.diag([10., 5., 2.])
    A  = U0 @ S0 @ V0.T + 0.1 * np.random.randn(m, n)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    print(f"  A: {A.shape},  singular values: {s.round(4)}")
    print(f"  {'k':>3}  {'||A - A_k||_F':>16}  {'Var explained':>15}")
    total_var = np.sum(s**2)
    for k in range(1, len(s) + 1):
        A_k   = (U[:, :k] * s[:k]) @ Vt[:k, :]
        err   = np.linalg.norm(A - A_k, 'fro')
        var_k = np.sum(s[:k]**2) / total_var
        print(f"  {k:>3}  {err:>16.6f}  {var_k:>15.4%}")


def demo_eigendecomposition():
    print("\n=== Eigendecomposition A = V D V^{-1} ===")
    # Only works for diagonalizable matrices; use symmetric for guaranteed real eigs
    np.random.seed(1)
    X = np.random.randn(4, 4)
    A = X.T @ X   # symmetric PSD -> real eigenvalues, orthogonal eigenvectors
    eigvals, eigvecs = np.linalg.eigh(A)   # eigh for symmetric
    D = np.diag(eigvals)
    V = eigvecs
    A_reconstructed = V @ D @ V.T
    print(f"  A (symmetric 4x4) eigenvalues: {eigvals.round(4)}")
    print(f"  V orthogonal check: err={np.linalg.norm(V.T @ V - np.eye(4)):.2e}")
    print(f"  V D V^T ~= A: err={np.linalg.norm(A - A_reconstructed):.2e}")
    # Matrix powers via eigendecomposition: A^k = V D^k V^T
    for power in [2, 3, 5]:
        A_pow_eig    = V @ np.diag(eigvals**power) @ V.T
        A_pow_direct = np.linalg.matrix_power(A, power)
        err = np.linalg.norm(A_pow_eig - A_pow_direct)
        print(f"  A^{power} via eig vs direct: err={err:.2e}")


def demo_iterative_refinement():
    print("\n=== QR vs Direct Solve and Iterative Refinement ===")
    np.random.seed(0)
    n = 5
    # Build ill-conditioned matrix via SVD
    U_mat = qr(np.random.randn(n, n))[0]
    V_mat = qr(np.random.randn(n, n))[0]
    sv    = np.array([1e3, 1e1, 1e0, 1e-1, 1e-3])
    A_sys = U_mat @ np.diag(sv) @ V_mat.T
    x_true = np.ones(n)
    b = A_sys @ x_true
    cond_A = np.linalg.cond(A_sys)
    print(f"  A: {n}x{n}, cond={cond_A:.4e}")
    # Direct solve
    x_dir = solve(A_sys, b)
    print(f"  Direct solve  ||err|| = {np.linalg.norm(x_dir - x_true):.4e}")
    # QR solve
    Q_s, R_s = qr(A_sys)
    x_qr = solve(R_s, Q_s.T @ b)
    print(f"  QR solve      ||err|| = {np.linalg.norm(x_qr - x_true):.4e}")
    # Iterative refinement on top of direct solve
    x_ref = x_dir.copy()
    print(f"\n  Iterative refinement (residual correction):")
    for step in range(6):
        r     = b - A_sys @ x_ref
        delta = solve(A_sys, r)
        x_ref = x_ref + delta
        err   = np.linalg.norm(x_ref - x_true)
        print(f"    step {step+1}: ||x - x_true|| = {err:.4e}")
    # Verify det via SVD: det(A) = prod(sv) * det(U) * det(V)
    det_svd    = np.prod(sv) * np.linalg.det(U_mat) * np.linalg.det(V_mat)
    det_direct = np.linalg.det(A_sys)
    print(f"\n  det(A) via SVD factors = {det_svd:.6f}")
    print(f"  det(A) direct          = {det_direct:.6f}")
    print(f"  Match: {np.isclose(det_svd, det_direct)}")


if __name__ == "__main__":
    demo_qr()
    demo_cholesky()
    demo_lu_scipy()
    demo_schur_decomposition()
    demo_polar_decomposition()
    demo_svd_low_rank_approx()
    demo_eigendecomposition()
    demo_iterative_refinement()
