"""
Working Example: Systems of Linear Equations
Covers Gaussian elimination, back-substitution, LU decomposition,
least squares, and applied examples like linear regression.
"""
import numpy as np
from scipy import linalg


# ── 1. Manual Gaussian elimination ───────────────────────────────────────────
def gaussian_elimination(A, b):
    """Solve Ax = b via Gaussian elimination with partial pivoting."""
    n = len(b)
    M = np.column_stack([A.astype(float), b.astype(float)])   # augmented matrix

    for col in range(n):
        # Partial pivot
        pivot_row = col + np.argmax(np.abs(M[col:, col]))
        M[[col, pivot_row]] = M[[pivot_row, col]]

        for row in range(col + 1, n):
            factor = M[row, col] / M[col, col]
            M[row, col:] -= factor * M[col, col:]

    # Back-substitution
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (M[i, -1] - np.dot(M[i, i+1:n], x[i+1:])) / M[i, i]
    return x


def demo_gaussian():
    print("=== Gaussian Elimination ===")
    # 3×3 system
    A = np.array([[2, 1, -1],
                  [-3, -1, 2],
                  [-2, 1, 2]], dtype=float)
    b = np.array([8, -11, -3], dtype=float)
    x = gaussian_elimination(A, b)
    print(f"  A:\n{A}")
    print(f"  b: {b}")
    print(f"  solution x = {x}")
    print(f"  residual ||Ax - b|| = {np.linalg.norm(A @ x - b):.2e}")
    print(f"  numpy verify       = {np.linalg.solve(A, b)}")


# ── 2. np.linalg.solve ────────────────────────────────────────────────────────
def demo_numpy_solve():
    print("\n=== np.linalg.solve ===")
    systems = [
        # (A, b, label)
        (np.array([[3, 1], [1, 2]]), np.array([9, 8]), "2×2"),
        (np.array([[1, 2, 1],
                   [3, 8, 1],
                   [0, 4, 1]]), np.array([2, 12, 2]), "3×3"),
    ]
    for A, b, label in systems:
        x = np.linalg.solve(A.astype(float), b.astype(float))
        residual = np.linalg.norm(A @ x - b)
        print(f"  {label}: x = {x}  residual={residual:.2e}")


# ── 3. LU decomposition ───────────────────────────────────────────────────────
def demo_lu():
    print("\n=== LU Decomposition ===")
    A = np.array([[2, 1, 1],
                  [4, 3, 3],
                  [8, 7, 9]], dtype=float)
    P, L, U = linalg.lu(A)
    print(f"  A:\n{A}")
    print(f"  P (permutation):\n{P.astype(int)}")
    print(f"  L (lower):\n{np.round(L, 4)}")
    print(f"  U (upper):\n{np.round(U, 4)}")
    print(f"  P @ L @ U ≈ A: {np.allclose(P @ L @ U, A)}")

    # Solve using LU
    lu_piv = linalg.lu_factor(A)
    b = np.array([1, 2, 3], dtype=float)
    x = linalg.lu_solve(lu_piv, b)
    print(f"  solve Ax=b via LU: x={x}  check: {np.round(A@x,4)}")


# ── 4. Least squares (overdetermined systems) ─────────────────────────────────
def demo_least_squares():
    print("\n=== Least Squares (Overdetermined System) ===")
    # Fit y = a + b*x to noisy data
    rng = np.random.default_rng(0)
    x   = np.linspace(0, 10, 30)
    y   = 2.5 + 1.8 * x + rng.normal(0, 1, 30)

    # Design matrix A for [a, b]
    A = np.column_stack([np.ones_like(x), x])
    # Normal equations: (A^T A) coeff = A^T y
    coeff_norm = np.linalg.solve(A.T @ A, A.T @ y)
    # numpy lstsq (more numerically stable)
    coeff_lstsq, residuals, rank, sv = np.linalg.lstsq(A, y, rcond=None)

    print(f"  true coeffs     : a=2.5, b=1.8")
    print(f"  normal equations: a={coeff_norm[0]:.4f}, b={coeff_norm[1]:.4f}")
    print(f"  lstsq           : a={coeff_lstsq[0]:.4f}, b={coeff_lstsq[1]:.4f}")
    print(f"  matrix rank     : {rank}")


# ── 5. Underdetermined system (infinite solutions) ────────────────────────────
def demo_underdetermined():
    print("\n=== Underdetermined System (infinite solutions) ===")
    A = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
    b = np.array([14, 32], dtype=float)

    # Minimum-norm solution via pseudo-inverse
    x_min_norm = np.linalg.pinv(A) @ b
    print(f"  A (2×3, more unknowns than equations):\n{A}")
    print(f"  b: {b}")
    print(f"  min-norm solution x = {x_min_norm}")
    print(f"  ||x||₂             = {np.linalg.norm(x_min_norm):.4f}")
    print(f"  check Ax           = {np.round(A @ x_min_norm, 6)}")


# ── 6. Checking existence and uniqueness ──────────────────────────────────────
def demo_solution_analysis():
    print("\n=== Solution Existence & Uniqueness ===")
    cases = [
        (np.array([[1,2],[3,4.]]), np.array([5,6.]),    "unique solution"),
        (np.array([[1,2],[2,4.]]), np.array([5,10.]),   "infinite solutions"),
        (np.array([[1,2],[2,4.]]), np.array([5,11.]),   "no solution"),
    ]
    for A, b, label in cases:
        rank_A  = np.linalg.matrix_rank(A)
        Ab      = np.column_stack([A, b])
        rank_Ab = np.linalg.matrix_rank(Ab)
        n       = A.shape[1]
        print(f"  {label:<24}: rank(A)={rank_A}, rank([A|b])={rank_Ab}, n={n}")
        if rank_A == rank_Ab == n:
            print(f"    → Unique solution: {np.round(np.linalg.solve(A, b), 4)}")
        elif rank_A == rank_Ab < n:
            print(f"    → Infinitely many solutions (underdetermined)")
        else:
            print(f"    → No solution (inconsistent)")


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    demo_gaussian()
    demo_numpy_solve()
    demo_lu()
    demo_least_squares()
    demo_underdetermined()
    demo_solution_analysis()
