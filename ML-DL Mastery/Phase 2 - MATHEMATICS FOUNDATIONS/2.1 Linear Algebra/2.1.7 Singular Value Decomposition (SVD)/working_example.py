"""
Working Example: Singular Value Decomposition (SVD)
Covers computation, geometric meaning, low-rank approximation,
image compression, PCA via SVD, and pseudoinverse.
"""
import numpy as np
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_svd")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -- 1. SVD decomposition ------------------------------------------------------
def svd_basics():
    print("=== SVD Basics: A = U Sigma Vᵀ ===")
    A = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9],
                  [10,11,12]], dtype=float)
    U, s, Vt = np.linalg.svd(A, full_matrices=True)
    print(f"  A shape  : {A.shape}")
    print(f"  U shape  : {U.shape}  (left singular vectors, orthogonal)")
    print(f"  s        : {np.round(s, 4)}  (singular values, descending)")
    print(f"  Vt shape : {Vt.shape}  (right singular vectors, orthogonal)")

    # Reconstruct A
    Sigma = np.zeros_like(A)
    Sigma[:len(s), :len(s)] = np.diag(s)
    A_reconstructed = U @ Sigma @ Vt
    print(f"  ||A - USigmaVᵀ|| = {np.linalg.norm(A - A_reconstructed):.2e}")

    # Properties
    print(f"\n  U orthogonal: UᵀU = I? {np.allclose(U.T @ U, np.eye(U.shape[1]))}")
    print(f"  V orthogonal: VᵀV = I? {np.allclose(Vt.T @ Vt, np.eye(Vt.shape[0]))}")
    print(f"  rank(A)  = {np.linalg.matrix_rank(A)}")
    print(f"  non-zero singular values: {(s > 1e-10).sum()}")


# -- 2. Economy (thin) SVD -----------------------------------------------------
def thin_svd():
    print("\n=== Thin (Economy) SVD ===")
    m, n = 6, 4
    rng  = np.random.default_rng(0)
    A    = rng.standard_normal((m, n))
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    print(f"  A ({m}×{n}): full SVD gives U ({m}×{m}), thin gives U ({m}×{n})")
    print(f"  thin U: {U.shape}  s: {s.shape}  Vt: {Vt.shape}")
    print(f"  singular values: {np.round(s, 4)}")


# -- 3. Low-rank approximation -------------------------------------------------
def low_rank_approx():
    print("\n=== Low-Rank Approximation (Eckart-Young theorem) ===")
    rng = np.random.default_rng(42)
    A = rng.standard_normal((8, 6))
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    full_norm = np.linalg.norm(A, 'fro')

    for k in [1, 2, 3, 6]:
        Ak = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
        error = np.linalg.norm(A - Ak, 'fro')
        explained = (s[:k]**2).sum() / (s**2).sum() * 100
        print(f"  rank-{k}: error={error:.4f}  explained variance={explained:.1f}%")


# -- 4. Image compression ------------------------------------------------------
def image_compression():
    print("\n=== Image Compression via SVD ===")
    rng = np.random.default_rng(0)
    # Synthetic 64×64 "image" with structure
    x = np.linspace(0, 4*np.pi, 64)
    img = np.outer(np.sin(x), np.cos(x)) + 0.2 * rng.standard_normal((64,64))

    U, s, Vt = np.linalg.svd(img, full_matrices=False)
    ranks = [1, 3, 5, 10, 20, 64]

    fig, axes = plt.subplots(1, len(ranks), figsize=(16, 3))
    for ax, k in zip(axes, ranks):
        approx = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
        err    = np.linalg.norm(img - approx, 'fro')
        ax.imshow(approx, cmap="gray")
        ax.set_title(f"k={k}\nerr={err:.2f}", fontsize=8)
        ax.axis("off")
    fig.suptitle("SVD Image Compression", fontsize=11)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "svd_image_compression.png")
    plt.savefig(path, dpi=90)
    plt.close()
    print(f"  Saved: {path}")

    storage_fractions = {k: k*(64+64+1)/(64*64) for k in ranks}
    for k, frac in storage_fractions.items():
        print(f"  rank-{k:2d}: stores {frac:.2%} of original")


# -- 5. PCA via SVD -----------------------------------------------------------
def pca_via_svd():
    print("\n=== PCA via SVD ===")
    rng  = np.random.default_rng(5)
    data = rng.multivariate_normal(mean=[0,0,0],
                                   cov=[[3,2,1],[2,2,1],[1,1,1]],
                                   size=500)
    X    = data - data.mean(axis=0)        # centre
    U, s, Vt = np.linalg.svd(X, full_matrices=False)

    # Principal components = rows of Vt
    explained = s**2 / (s**2).sum() * 100
    print(f"  X shape: {X.shape}")
    print(f"  singular values: {np.round(s, 4)}")
    print(f"  explained var%: {np.round(explained, 2)}")

    X_pc = X @ Vt.T[:, :2]   # project onto first 2 PCs
    print(f"  projected to 2D: {X_pc.shape}  std={np.round(X_pc.std(axis=0),4)}")


# -- 6. Pseudo-inverse via SVD -------------------------------------------------
def pseudo_inverse():
    print("\n=== Pseudoinverse via SVD ===")
    A = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)   # non-square
    U, s, Vt = np.linalg.svd(A, full_matrices=False)

    # A⁺ = V Sigma⁺ Uᵀ
    s_inv = 1 / s
    Aplus  = Vt.T @ np.diag(s_inv) @ U.T
    np_Aplus = np.linalg.pinv(A)

    print(f"  A ({A.shape}):\n{A}")
    print(f"  A⁺ via SVD ({Aplus.shape}):\n{np.round(Aplus, 4)}")
    print(f"  matches np.linalg.pinv: {np.allclose(Aplus, np_Aplus)}")
    print(f"  A A⁺ A = A: {np.allclose(A @ Aplus @ A, A)}")
    print(f"  A⁺ A A⁺ = A⁺: {np.allclose(Aplus @ A @ Aplus, Aplus)}")


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    svd_basics()
    thin_svd()
    low_rank_approx()
    image_compression()
    pca_via_svd()
    pseudo_inverse()
