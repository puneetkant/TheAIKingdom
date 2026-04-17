"""
Working Example: Eigenvalues and Eigenvectors
Covers definition, computation, diagonalisation, spectral theorem,
PCA preview, and PageRank / power iteration.
"""
import numpy as np


# ── Definition and verification ───────────────────────────────────────────────
def definition():
    print("=== Eigenvalue Definition: Av = λv ===")
    A  = np.array([[4, 1], [2, 3]], dtype=float)
    vals, vecs = np.linalg.eig(A)
    print(f"  A:\n{A}")
    print(f"  eigenvalues  λ = {vals}")
    print(f"  eigenvectors (columns):\n{np.round(vecs, 4)}")

    # Verify Av = λv for each pair
    for i, (lam, v) in enumerate(zip(vals, vecs.T)):
        Av   = A @ v
        lv   = lam * v
        match = np.allclose(Av, lv)
        print(f"  Pair {i}: λ={lam:.4f}  v={np.round(v,4)}  Av≈λv: {match}")


# ── Symmetric matrix (real eigenvalues, orthogonal eigenvectors) ──────────────
def symmetric_matrix():
    print("\n=== Symmetric Matrix (Spectral Theorem) ===")
    A = np.array([[4, 2, 0],
                  [2, 3, 1],
                  [0, 1, 5]], dtype=float)
    vals, vecs = np.linalg.eigh(A)   # eigh for real symmetric
    print(f"  A:\n{A}")
    print(f"  eigenvalues  = {np.round(vals, 4)}")
    print(f"  eigenvectors (columns):\n{np.round(vecs, 4)}")
    print(f"  Q orthogonal: Q^T Q = I? {np.allclose(vecs.T @ vecs, np.eye(3))}")
    print(f"  A = Q Λ Q^T? {np.allclose(vecs @ np.diag(vals) @ vecs.T, A)}")


# ── Diagonalisation ───────────────────────────────────────────────────────────
def diagonalisation():
    print("\n=== Diagonalisation A = PDP⁻¹ ===")
    A = np.array([[3, 1], [0, 2]], dtype=float)
    vals, vecs = np.linalg.eig(A)
    P    = vecs
    D    = np.diag(vals)
    Pinv = np.linalg.inv(P)

    print(f"  A:\n{A}")
    print(f"  D (diagonal):\n{D}")
    print(f"  P @ D @ P⁻¹ = A? {np.allclose(P @ D @ Pinv, A)}")

    # Matrix power via diagonalisation: A^10 = P D^10 P^-1
    k = 10
    Ak = P @ np.diag(vals**k) @ Pinv
    print(f"\n  A^{k} via diag:\n{np.round(Ak, 2)}")
    print(f"  verify numpy   :\n{np.round(np.linalg.matrix_power(A.astype(int), k), 2)}")


# ── Characteristic polynomial ────────────────────────────────────────────────
def characteristic_poly():
    print("\n=== Characteristic Polynomial det(A - λI) = 0 ===")
    A = np.array([[2, 1], [1, 2]], dtype=float)
    # Expand manually for 2×2: λ² - tr(A)λ + det(A)
    tr  = np.trace(A)
    det = np.linalg.det(A)
    print(f"  A:\n{A}")
    print(f"  tr(A)={tr}, det(A)={det}")
    print(f"  char poly: λ² - {tr}λ + {det:.0f} = 0")
    roots = np.roots([1, -tr, det])
    print(f"  roots λ = {np.sort(roots)}")
    print(f"  numpy eig λ = {np.sort(np.linalg.eig(A)[0])}")


# ── Power iteration (dominant eigenvector) ────────────────────────────────────
def power_iteration(A, num_iter=50, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(A.shape[0])
    v /= np.linalg.norm(v)
    for _ in range(num_iter):
        v = A @ v
        lam = np.linalg.norm(v)
        v /= lam
    return lam, v


def power_iteration_demo():
    print("\n=== Power Iteration ===")
    A = np.array([[4, 1, 0],
                  [1, 3, 1],
                  [0, 1, 2]], dtype=float)
    lam_power, v_power = power_iteration(A)
    true_vals, true_vecs = np.linalg.eig(A)
    dominant_idx = np.argmax(np.abs(true_vals))

    print(f"  Power iteration  λ_max ≈ {lam_power:.6f}")
    print(f"  numpy eig        λ_max = {np.abs(true_vals[dominant_idx]):.6f}")
    # Align sign
    v_ref = true_vecs[:, dominant_idx].real
    v_ref *= np.sign(np.dot(v_power, v_ref))
    print(f"  eigenvector diff: {np.linalg.norm(v_power - v_ref):.2e}")


# ── PCA intuition via eigendecomposition ──────────────────────────────────────
def pca_eigen():
    print("\n=== PCA Intuition via Eigendecomposition ===")
    rng  = np.random.default_rng(7)
    data = rng.multivariate_normal(mean=[0,0],
                                   cov=[[3, 2], [2, 2]],
                                   size=300)
    X   = data - data.mean(axis=0)
    Cov = (X.T @ X) / (len(X) - 1)
    print(f"  covariance matrix:\n{np.round(Cov, 4)}")

    vals, vecs = np.linalg.eigh(Cov)
    idx  = np.argsort(vals)[::-1]    # sort descending
    vals = vals[idx];  vecs = vecs[:, idx]
    explained = vals / vals.sum() * 100

    print(f"  eigenvalues    : {np.round(vals, 4)}")
    print(f"  explained var% : {np.round(explained, 2)}")
    print(f"  PC1 direction  : {np.round(vecs[:, 0], 4)}")
    print(f"  PC2 direction  : {np.round(vecs[:, 1], 4)}")

    # Project onto PC1
    X_pca = X @ vecs[:, :1]
    print(f"  X projected onto PC1 — std={X_pca.std():.4f}")


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    definition()
    symmetric_matrix()
    diagonalisation()
    characteristic_poly()
    power_iteration_demo()
    pca_eigen()
