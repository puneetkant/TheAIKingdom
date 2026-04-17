"""
Working Example: Matrices
Covers matrix creation, operations, transpose, inverse, trace,
rank, norms, special matrices, and ML applications.
"""
import numpy as np


# ── Creation ──────────────────────────────────────────────────────────────────
def creation():
    print("=== Matrix Creation ===")
    A = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]], dtype=float)
    print(f"  A:\n{A}")
    print(f"  shape={A.shape}  rank={np.linalg.matrix_rank(A)}  dtype={A.dtype}")
    print(f"\n  zeros(3,3):\n{np.zeros((3,3))}")
    print(f"\n  identity(4):\n{np.eye(4).astype(int)}")
    print(f"\n  diag([1,2,3,4]):\n{np.diag([1,2,3,4])}")
    print(f"\n  random(2,3):\n{np.round(np.random.default_rng(0).standard_normal((2,3)),3)}")


# ── Basic arithmetic ──────────────────────────────────────────────────────────
def arithmetic():
    print("\n=== Matrix Arithmetic ===")
    A = np.array([[1, 2], [3, 4]], dtype=float)
    B = np.array([[5, 6], [7, 8]], dtype=float)

    print(f"  A:\n{A}")
    print(f"  B:\n{B}")
    print(f"  A + B:\n{A + B}")
    print(f"  A - B:\n{A - B}")
    print(f"  2 * A:\n{2 * A}")
    print(f"  A * B (element-wise):\n{A * B}")   # Hadamard
    print(f"  A @ B (matmul):\n{A @ B}")
    print(f"  A.T (transpose):\n{A.T}")
    print(f"\n  trace(A) = {np.trace(A)}")
    print(f"  frobenius norm = {np.linalg.norm(A, 'fro'):.4f}")


# ── Matrix powers and inverse ─────────────────────────────────────────────────
def power_and_inverse():
    print("\n=== Inverse & Powers ===")
    A = np.array([[2, 1], [5, 3]], dtype=float)
    print(f"  A:\n{A}")
    print(f"  det(A) = {np.linalg.det(A):.4f}")
    Ainv = np.linalg.inv(A)
    print(f"  A⁻¹:\n{Ainv}")
    print(f"  A @ A⁻¹ ≈ I:\n{np.round(A @ Ainv, 10)}")

    # Matrix power via repeated multiplication
    A2 = A @ A
    A3 = A2 @ A
    print(f"  A² :\n{A2}")
    print(f"  A³ :\n{A3}")
    # Using numpy
    from numpy.linalg import matrix_power
    print(f"  matrix_power(A,3):\n{matrix_power(A, 3)}")

    # Pseudo-inverse for non-square / singular
    B = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
    Bplus = np.linalg.pinv(B)
    print(f"\n  B (2×3):\n{B}")
    print(f"  pinv(B) (3×2):\n{np.round(Bplus, 4)}")
    print(f"  B @ pinv(B) ≈ I₂:\n{np.round(B @ Bplus, 4)}")


# ── Rank ──────────────────────────────────────────────────────────────────────
def rank():
    print("\n=== Matrix Rank ===")
    cases = {
        "full rank 3×3":   np.eye(3),
        "rank-2 (row dep)": np.array([[1,2,3],[2,4,6],[0,1,0.]], dtype=float),
        "zero matrix":     np.zeros((3,3)),
        "3×5 random":      np.random.default_rng(7).standard_normal((3,5)),
    }
    for label, M in cases.items():
        print(f"  {label:<22}: rank={np.linalg.matrix_rank(M)}, shape={M.shape}")


# ── Special matrices ──────────────────────────────────────────────────────────
def special_matrices():
    print("\n=== Special Matrices ===")

    # Symmetric
    A = np.array([[4, 2, 1], [2, 5, 3], [1, 3, 6]], dtype=float)
    print(f"  symmetric A == A.T: {np.allclose(A, A.T)}")

    # Orthogonal: Q^T Q = I
    theta = np.pi / 4
    Q = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    print(f"  rotation matrix Q^T Q ≈ I: {np.allclose(Q.T @ Q, np.eye(2))}")

    # Upper / lower triangular
    L = np.tril(np.ones((4,4)))
    U = np.triu(np.ones((4,4)))
    print(f"  lower triangular L:\n{L.astype(int)}")
    print(f"  upper triangular U:\n{U.astype(int)}")

    # Diagonal dominance
    D = np.array([[10,1,1],[1,10,1],[1,1,10]], dtype=float)
    is_diag_dom = all(abs(D[i,i]) >= sum(abs(D[i,j]) for j in range(3) if j!=i)
                     for i in range(3))
    print(f"  diagonally dominant: {is_diag_dom}")


# ── Norms ─────────────────────────────────────────────────────────────────────
def matrix_norms():
    print("\n=== Matrix Norms ===")
    A = np.array([[1, 2], [3, 4]], dtype=float)
    norms = {
        "Frobenius":    np.linalg.norm(A, "fro"),
        "1-norm (max col sum)": np.linalg.norm(A, 1),
        "∞-norm (max row sum)": np.linalg.norm(A, np.inf),
        "2-norm (spectral)":    np.linalg.norm(A, 2),
        "nuclear":              np.sum(np.linalg.svd(A, compute_uv=False)),
    }
    for name, val in norms.items():
        print(f"  {name:<30}: {val:.4f}")


# ── ML application: covariance matrix ────────────────────────────────────────
def covariance_matrix():
    print("\n=== Application: Covariance Matrix ===")
    rng  = np.random.default_rng(42)
    data = rng.multivariate_normal(mean=[0, 0, 0],
                                   cov=[[1, 0.8, 0.3],
                                        [0.8, 1, 0.1],
                                        [0.3, 0.1, 1]],
                                   size=500)
    X  = data - data.mean(axis=0)
    Cov = (X.T @ X) / (len(X) - 1)
    print(f"  data shape : {data.shape}")
    print(f"  Cov matrix :\n{np.round(Cov, 3)}")
    print(f"  symmetric  : {np.allclose(Cov, Cov.T)}")
    print(f"  pos definite: {np.all(np.linalg.eigvalsh(Cov) > 0)}")


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    creation()
    arithmetic()
    power_and_inverse()
    rank()
    special_matrices()
    matrix_norms()
    covariance_matrix()
