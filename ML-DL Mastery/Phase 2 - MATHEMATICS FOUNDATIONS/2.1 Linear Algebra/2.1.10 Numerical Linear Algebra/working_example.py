"""
Working Example: Numerical Linear Algebra
Covers condition numbers, floating-point stability, iterative solvers,
sparse matrices, and practical numerical considerations.
"""
import numpy as np
from scipy import linalg, sparse
from scipy.sparse.linalg import spsolve, cg


# ── 1. Condition number and numerical stability ───────────────────────────────
def condition_numbers():
    print("=== Condition Numbers κ(A) = ||A|| · ||A⁻¹|| ===")

    # Well-conditioned
    A_good = np.array([[2., 1.], [1., 2.]])
    # Ill-conditioned (Hilbert matrix)
    n = 8
    A_bad = np.array([[1.0 / (i + j + 1) for j in range(n)] for i in range(n)])

    for name, A in [("Well-conditioned 2×2", A_good),
                    (f"Hilbert matrix {n}×{n}", A_bad)]:
        kappa = np.linalg.cond(A)
        print(f"\n  {name}:")
        print(f"    κ(A) = {kappa:.3e}")
        print(f"    ~{int(np.log10(kappa))} digits of precision lost")
        # Compare solve accuracy
        x_true = np.ones(A.shape[0])
        b      = A @ x_true
        x_comp = np.linalg.solve(A, b)
        err    = np.linalg.norm(x_comp - x_true)
        print(f"    ||x_computed - x_true|| = {err:.3e}")


# ── 2. Floating-point pitfalls ────────────────────────────────────────────────
def floating_point_issues():
    print("\n=== Floating-Point Issues ===")

    # Catastrophic cancellation
    x = 1.0
    eps = 1e-10
    a = (x + eps)**2 - x**2
    b = 2*x*eps + eps**2       # algebraically equivalent
    print(f"  Cancellation: (x+ε)²-x² = {a:.6e}   2xε+ε² = {b:.6e}  differ by {abs(a-b):.2e}")

    # Machine epsilon
    eps_machine = np.finfo(float).eps
    print(f"\n  Machine epsilon : {eps_machine:.2e}")
    print(f"  1 + eps = 1?    {1.0 + eps_machine == 1.0}  (eps just barely representable)")
    print(f"  1 + eps/2 = 1?  {1.0 + eps_machine/2 == 1.0}")

    # Associativity breakdown
    a, b, c = 1e15, -1e15, 1.23
    print(f"\n  (a+b)+c = {(a+b)+c}   a+(b+c) = {a+(b+c)}  (should both be 1.23)")


# ── 3. Iterative solvers ──────────────────────────────────────────────────────
def iterative_solvers():
    print("\n=== Iterative Solvers (Conjugate Gradient) ===")
    # Build large sparse positive definite system
    n   = 100
    A_sp = sparse.diags([-1, 4, -1], [-1, 0, 1], shape=(n, n), format='csr')
    b    = np.ones(n)

    # Direct solve
    x_direct = spsolve(A_sp, b)

    # Conjugate gradient (works for SPD)
    x_cg, info = cg(A_sp, b, tol=1e-8)
    print(f"  System: {n}×{n} tridiagonal SPD sparse matrix")
    print(f"  CG convergence info: {info}  (0 = success)")
    print(f"  ||x_cg - x_direct|| = {np.linalg.norm(x_cg - x_direct):.2e}")
    print(f"  ||r|| = ||Ax-b||    = {np.linalg.norm(A_sp @ x_cg - b):.2e}")


# ── 4. Jacobi and Gauss-Seidel iteration ─────────────────────────────────────
def jacobi_method(A, b, max_iter=100, tol=1e-8):
    n = len(b)
    x = np.zeros(n)
    D_inv = 1.0 / np.diag(A)
    R = A - np.diag(np.diag(A))
    for k in range(max_iter):
        x_new = D_inv * (b - R @ x)
        if np.linalg.norm(x_new - x) < tol:
            return x_new, k + 1
        x = x_new
    return x, max_iter


def iterative_classic():
    print("\n=== Classic Iterative Methods (Jacobi) ===")
    # Diagonally dominant system (ensures convergence)
    A = np.array([[4., 1., 0., 0.],
                  [1., 4., 1., 0.],
                  [0., 1., 4., 1.],
                  [0., 0., 1., 4.]])
    b = np.array([1., 2., 3., 4.])

    x_jacobi, iters = jacobi_method(A, b)
    x_direct        = np.linalg.solve(A, b)
    print(f"  Jacobi converged in {iters} iterations")
    print(f"  x (Jacobi) = {np.round(x_jacobi, 6)}")
    print(f"  x (direct) = {np.round(x_direct, 6)}")
    print(f"  ||diff||   = {np.linalg.norm(x_jacobi - x_direct):.2e}")

    # Convergence condition: spectral radius of D⁻¹R < 1
    D = np.diag(np.diag(A))
    R = A - D
    iteration_matrix = np.linalg.inv(D) @ R
    rho = max(abs(np.linalg.eigvals(iteration_matrix)))
    print(f"  Spectral radius ρ(D⁻¹R) = {rho:.4f}  < 1? {rho < 1}  (convergence condition)")


# ── 5. Sparse matrix storage ──────────────────────────────────────────────────
def sparse_matrices():
    print("\n=== Sparse Matrices ===")
    n = 1000
    # Dense storage
    dense_bytes = n * n * 8   # float64

    # Sparse tri-diagonal
    A_sp = sparse.diags([-1, 4, -1], [-1, 0, 1], shape=(n, n), format='csr')
    nnz  = A_sp.nnz
    sparse_bytes = nnz * (8 + 4) + (n + 1) * 4  # data + col_ind + row_ptr

    print(f"  {n}×{n} tridiagonal matrix:")
    print(f"    dense  storage: {dense_bytes / 1e6:.1f} MB")
    print(f"    sparse storage: {sparse_bytes / 1e3:.1f} KB  ({nnz} non-zeros)")
    print(f"    reduction: {dense_bytes / sparse_bytes:.0f}×")

    formats = ['csr', 'csc', 'coo', 'lil', 'dok']
    print(f"\n  Common sparse formats:")
    print(f"    {'CSR':<6}: compressed row — fast row ops, matrix-vector multiply")
    print(f"    {'CSC':<6}: compressed col — fast col ops, column slicing")
    print(f"    {'COO':<6}: coordinate — easy construction")
    print(f"    {'LIL':<6}: list of lists — incremental construction")
    print(f"    {'DOK':<6}: dict of keys — random access during build")


# ── 6. Numerical rank and pivoting ────────────────────────────────────────────
def numerical_rank():
    print("\n=== Numerical Rank & Pivoting ===")
    A = np.array([[1., 2., 3.],
                  [4., 5., 6.],
                  [7., 8., 9.]], dtype=float)   # rank-2 matrix

    # Rank via SVD (robust)
    U, s, Vt = np.linalg.svd(A)
    tol  = max(A.shape) * np.finfo(float).eps * s[0]
    rank = (s > tol).sum()
    print(f"  A (rank-deficient):")
    print(f"    singular values: {np.round(s, 6)}")
    print(f"    numerical rank = {rank}  (tol={tol:.2e})")

    # LU with pivoting for rank detection
    P, L, U_lu = linalg.lu(A)
    diag_U = np.abs(np.diag(U_lu))
    print(f"\n  LU diagonal |U_ii|: {np.round(diag_U, 6)}")
    print(f"  Small pivots indicate near-singular rows")


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    condition_numbers()
    floating_point_issues()
    iterative_solvers()
    iterative_classic()
    sparse_matrices()
    numerical_rank()
