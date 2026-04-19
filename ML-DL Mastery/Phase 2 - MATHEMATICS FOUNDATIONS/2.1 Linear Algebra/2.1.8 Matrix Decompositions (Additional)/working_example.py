"""
Working Example: Matrix Decompositions (Additional)
Covers QR, Cholesky, Schur, and LDL decompositions
with their applications (least squares, positive-definite systems, etc.).
"""
import numpy as np
from scipy import linalg


# -- 1. QR Decomposition -------------------------------------------------------
def qr_decomposition():
    print("=== QR Decomposition (A = QR) ===")
    A = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 10],
                  [1, 0, 2]], dtype=float)
    Q, R = np.linalg.qr(A)
    print(f"  A ({A.shape}):\n{A}")
    print(f"  Q ({Q.shape}) — orthonormal columns:")
    print(np.round(Q, 4))
    print(f"  R ({R.shape}) — upper triangular:")
    print(np.round(R, 4))
    print(f"  Q orthogonal: QᵀQ=I? {np.allclose(Q.T @ Q, np.eye(Q.shape[1]))}")
    print(f"  QR = A?        {np.allclose(Q @ R, A)}")

    # Least squares using QR
    print("\n  QR-based least squares Ax ~= b:")
    b = np.array([1, 2, 3, 4], dtype=float)
    # x = R^-1 Qᵀ b  (only for full-rank A)
    x_qr   = np.linalg.solve(R[:A.shape[1]], Q.T @ b)
    x_lstsq = np.linalg.lstsq(A, b, rcond=None)[0]
    print(f"  x (QR)    = {np.round(x_qr, 4)}")
    print(f"  x (lstsq) = {np.round(x_lstsq, 4)}")
    print(f"  match: {np.allclose(x_qr, x_lstsq, atol=1e-8)}")


# -- 2. Cholesky Decomposition -------------------------------------------------
def cholesky_decomposition():
    print("\n=== Cholesky Decomposition (A = LLᵀ, A symmetric positive definite) ===")
    # Create a positive definite matrix: A = Bᵀ B + epsilon I
    rng = np.random.default_rng(0)
    B   = rng.standard_normal((4, 4))
    A   = B.T @ B + 4 * np.eye(4)   # positive definite

    L   = np.linalg.cholesky(A)   # lower triangular
    print(f"  A ({A.shape}):\n{np.round(A, 3)}")
    print(f"  L (Cholesky factor):\n{np.round(L, 4)}")
    print(f"  L Lᵀ = A? {np.allclose(L @ L.T, A)}")

    # Solve Ax = b via Cholesky (twice as fast as LU for SPD matrices)
    b = rng.standard_normal(4)
    y = np.linalg.solve(L, b)          # Ly = b
    x = np.linalg.solve(L.T, y)        # Lᵀx = y
    print(f"\n  Solving Ax=b via Cholesky:")
    print(f"  x = {np.round(x, 4)}")
    print(f"  ||Ax-b|| = {np.linalg.norm(A @ x - b):.2e}")

    # Check positive definiteness
    eigvals = np.linalg.eigvalsh(A)
    print(f"\n  eigenvalues of A: {np.round(eigvals, 4)}")
    print(f"  positive definite: {np.all(eigvals > 0)}")


# -- 3. Schur Decomposition ----------------------------------------------------
def schur_decomposition():
    print("\n=== Schur Decomposition (A = Q T Qᵀ) ===")
    A = np.array([[4, 1, 0],
                  [2, 3, 1],
                  [0, 1, 2]], dtype=float)
    T, Z = linalg.schur(A)   # Z unitary, T upper triangular (or quasi-upper)
    print(f"  A:\n{A}")
    print(f"  T (quasi-triangular):\n{np.round(T, 4)}")
    print(f"  diagonal of T = eigenvalues: {np.round(np.diag(T), 4)}")
    print(f"  scipy eig                  : {np.round(np.sort(np.linalg.eigvals(A)), 4)}")
    print(f"  Z Q Qᵀ = I: {np.allclose(Z @ Z.T, np.eye(3))}")
    print(f"  Z T Zᵀ = A: {np.allclose(Z @ T @ Z.T, A)}")


# -- 4. LDLᵀ Decomposition ----------------------------------------------------
def ldl_decomposition():
    print("\n=== LDLᵀ Decomposition ===")
    A = np.array([[4, 2, 1],
                  [2, 5, 3],
                  [1, 3, 6]], dtype=float)
    # scipy LDL
    lu, d, perm = linalg.ldl(A)
    print(f"  A:\n{A}")
    print(f"  L (unit lower triangular):\n{np.round(lu, 4)}")
    print(f"  D (block diagonal):\n{np.round(d, 4)}")
    # Reconstruct
    recon = lu @ d @ lu.T
    print(f"  L D Lᵀ = A? {np.allclose(recon[np.ix_(perm,perm)], A, atol=1e-10)}")


# -- 5. Comparison: when to use which -----------------------------------------
def decomposition_guide():
    print("\n=== Decomposition Selection Guide ===")
    guide = [
        ("LU",       "General square A",              "Solve Ax=b, compute det"),
        ("QR",       "mxn (m>=n), any A",              "Least squares, QR algorithm"),
        ("Cholesky", "Symmetric positive definite A",  "SPD systems, sampling gaussians"),
        ("SVD",      "Any A",                          "Pseudoinverse, PCA, rank reveal"),
        ("Eig",      "Square A",                       "Diagonalisation, PCA, PageRank"),
        ("Schur",    "Square A",                       "Computing matrix functions"),
        ("LDLᵀ",     "Symmetric A (not nec PD)",       "Indefinite systems"),
    ]
    print(f"  {'Decomp':<12} {'When to use':<35} {'Applications'}")
    print("  " + "-"*75)
    for name, when, apps in guide:
        print(f"  {name:<12} {when:<35} {apps}")


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    qr_decomposition()
    cholesky_decomposition()
    schur_decomposition()
    ldl_decomposition()
    decomposition_guide()
