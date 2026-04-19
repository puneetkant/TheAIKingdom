"""
Working Example: Determinants
Covers definition, cofactor expansion, properties, computation,
geometric interpretation, and Cramer's rule.
"""
import numpy as np
from itertools import permutations
import math


# -- Manual determinant via Leibniz formula ------------------------------------
def det_leibniz(M):
    """O(n!) — educational only, not for large matrices."""
    n = len(M)
    total = 0
    for perm in permutations(range(n)):
        # Count inversions to get sign
        inv = sum(perm[i] > perm[j] for i in range(n) for j in range(i+1, n))
        sign = (-1) ** inv
        product = math.prod(M[i][perm[i]] for i in range(n))
        total += sign * product
    return total


# -- Cofactor expansion (recursive) -------------------------------------------
def minor(M, row, col):
    return [[M[i][j] for j in range(len(M[0])) if j != col]
            for i in range(len(M)) if i != row]


def det_cofactor(M):
    n = len(M)
    if n == 1: return M[0][0]
    if n == 2: return M[0][0]*M[1][1] - M[0][1]*M[1][0]
    return sum((-1)**c * M[0][c] * det_cofactor(minor(M, 0, c)) for c in range(n))


def demo_det_computation():
    print("=== Determinant Computation ===")
    cases = [
        [[3, 8], [4, 6]],
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],          # singular -> det=0
        [[2, -1, 0], [3, 5, -1], [1, -2, 4]],
    ]
    for M in cases:
        A      = np.array(M, dtype=float)
        d_lei  = round(det_leibniz(M), 6)
        d_cof  = round(det_cofactor(M), 6)
        d_np   = round(np.linalg.det(A), 6)
        print(f"  M={M}")
        print(f"    Leibniz={d_lei}  Cofactor={d_cof}  NumPy={d_np}\n")


# -- Properties of determinants ------------------------------------------------
def det_properties():
    print("=== Determinant Properties ===")
    rng = np.random.default_rng(42)
    A = rng.integers(1, 5, (3, 3)).astype(float)
    B = rng.integers(1, 5, (3, 3)).astype(float)

    dA = np.linalg.det(A)
    dB = np.linalg.det(B)

    props = [
        ("det(A@B) = det(A)*det(B)",
         np.linalg.det(A @ B),       dA * dB),
        ("det(A.T) = det(A)",
         np.linalg.det(A.T),         dA),
        ("det(2A) = 2³ det(A) [n=3]",
         np.linalg.det(2 * A),       8 * dA),
        ("det(A^-1) = 1/det(A)",
         np.linalg.det(np.linalg.inv(A)), 1 / dA),
        ("swap rows -> negate det",
         np.linalg.det(A[[1,0,2]]),  -dA),
    ]
    for label, computed, expected in props:
        ok = "[OK]" if abs(computed - expected) < 1e-8 else "[X]"
        print(f"  [{ok}] {label}")
        print(f"       computed={computed:.4f}  expected={expected:.4f}")


# -- Geometric interpretation -------------------------------------------------
def geometric_interpretation():
    print("\n=== Geometric Interpretation ===")
    # In 2D: |det| = area of parallelogram spanned by column vectors
    v1 = np.array([3, 1], dtype=float)
    v2 = np.array([1, 4], dtype=float)
    A  = np.column_stack([v1, v2])
    area = abs(np.linalg.det(A))
    print(f"  v1={v1}, v2={v2}")
    print(f"  |det(A)| = {area:.4f}  (area of parallelogram)")

    # In 3D: |det| = volume of parallelepiped
    a = np.array([1, 0, 0], dtype=float)
    b = np.array([0, 2, 0], dtype=float)
    c = np.array([0, 0, 3], dtype=float)
    M3 = np.column_stack([a, b, c])
    vol = abs(np.linalg.det(M3))
    print(f"  a={a}, b={b}, c={c}")
    print(f"  |det| = {vol:.4f}  (volume of box = 1×2×3 = 6)")

    # Scale factor interpretation
    A = 2 * np.eye(3)   # scales all axes by 2
    print(f"\n  scaling matrix 2I: det = {np.linalg.det(A):.4f}  (volume scale factor = 2³ = 8)")


# -- Cramer's Rule ------------------------------------------------------------
def cramers_rule(A, b):
    """Solve Ax = b using Cramer's Rule (for small n)."""
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    det_A = np.linalg.det(A)
    if abs(det_A) < 1e-12:
        raise ValueError("Matrix is singular — no unique solution")
    n = len(b)
    x = np.zeros(n)
    for i in range(n):
        Ai = A.copy()
        Ai[:, i] = b
        x[i] = np.linalg.det(Ai) / det_A
    return x


def demo_cramers_rule():
    print("\n=== Cramer's Rule ===")
    A = [[2, 1, -1], [-3, -1, 2], [-2, 1, 2]]
    b = [8, -11, -3]
    x = cramers_rule(A, b)
    print(f"  A = {A}")
    print(f"  b = {b}")
    print(f"  x = {x}")
    print(f"  verify Ax = {np.round(np.array(A) @ x, 6)}")


# -- Singularity detection -----------------------------------------------------
def singularity_demo():
    print("\n=== Singularity & Ill-Conditioning ===")
    cases = [
        ("identity",        np.eye(4)),
        ("all-ones",        np.ones((4,4))),
        ("Hilbert 4×4",     np.array([[1/(i+j+1) for j in range(4)] for i in range(4)])),
        ("nearly singular", np.array([[1,2],[2,4.0000001]])),
    ]
    for label, A in cases:
        det  = np.linalg.det(A)
        cond = np.linalg.cond(A)
        rank = np.linalg.matrix_rank(A)
        print(f"  {label:<20}: det={det:.4e}  cond={cond:.4e}  rank={rank}")


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    demo_det_computation()
    det_properties()
    geometric_interpretation()
    demo_cramers_rule()
    singularity_demo()
