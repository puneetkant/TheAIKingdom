"""
Working Example 2: Vector Spaces — Span, Basis, Rank, Null Space in ML Context
===============================================================================
Demonstrates: basis, span, linear independence (rank check), null space (SVD),
column space, orthogonal basis (Gram-Schmidt), applications to feature spaces.

Run:  python working_example2.py
"""
from pathlib import Path
try:
    import numpy as np
except ImportError:
    raise SystemExit("pip install numpy")

DATA = Path(__file__).parent / "data"
DATA.mkdir(exist_ok=True)

def demo_linear_independence():
    print("=== Linear Independence & Rank ===")
    # Independent vectors
    A = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    print(f"  Standard basis rank: {np.linalg.matrix_rank(A)}  (= n, full rank)")

    # Dependent vectors (col3 = col1 + col2)
    B = np.array([[1., 2., 3.], [4., 5., 9.], [2., 1., 3.]])
    print(f"  Dependent matrix rank: {np.linalg.matrix_rank(B)}  (< 3, redundant)")

def gram_schmidt(A: np.ndarray) -> np.ndarray:
    """Gram-Schmidt orthonormalisation of columns of A."""
    Q = np.zeros_like(A, dtype=float)
    for i in range(A.shape[1]):
        v = A[:, i].astype(float)
        for j in range(i):
            v -= np.dot(Q[:, j], A[:, i]) * Q[:, j]
        Q[:, i] = v / np.linalg.norm(v)
    return Q

def demo_gram_schmidt():
    print("\n=== Gram-Schmidt Orthonormalisation ===")
    A = np.array([[1., 1., 0.], [1., 0., 1.], [0., 1., 1.]], dtype=float)
    Q = gram_schmidt(A)
    print(f"  Q columns orthonormal check (Q^T Q ~= I):\n{(Q.T @ Q).round(6)}")

def demo_null_space():
    print("\n=== Null Space (via SVD) ===")
    # A matrix with a non-trivial null space
    A = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]], dtype=float)
    U, s, Vt = np.linalg.svd(A)
    null_mask = np.abs(s) < 1e-10
    # Zero singular values -> corresponding right singular vectors span null space
    # For 3×3 with rank 2, the 3rd row of Vt is in null space
    null_vecs = Vt[s < 1e-10] if any(s < 1e-10) else Vt[-1:]
    print(f"  Singular values: {s.round(6)}")
    print(f"  rank(A) = {np.linalg.matrix_rank(A)}")
    print(f"  Null vector approx: {Vt[-1].round(4)}")
    print(f"  A @ null_vec ~= 0: {(A @ Vt[-1]).round(10)}")

def demo_feature_space():
    print("\n=== Feature Space: Column Space as Information Carrier ===")
    import urllib.request, csv
    dest = DATA / "cal_housing.csv"
    if not dest.exists():
        try: urllib.request.urlretrieve(
            "https://huggingface.co/datasets/scikit-learn/california-housing/resolve/main/cal_housing.csv", dest)
        except Exception:
            dest.write_text("MedInc,HouseAge,AveRooms,AveBedrms\n"
                            "3.0,25,5.0,1.0\n6.0,15,7.0,1.5\n2.0,40,4.5,0.9\n")
    with open(dest) as f: rows = list(csv.DictReader(f))
    cols = ["MedInc","HouseAge","AveRooms","AveBedrms"]
    X = np.array([[float(r[c]) for c in cols] for r in rows[:100]])
    print(f"  Feature matrix: {X.shape}")
    print(f"  Rank: {np.linalg.matrix_rank(X)}  (should be 4 -- full column rank)")
    # Add a redundant feature
    X_red = np.hstack([X, (X[:,0] + X[:,1]).reshape(-1,1)])
    print(f"  With redundant col rank: {np.linalg.matrix_rank(X_red)}  (still 4)")

def demo_column_space():
    print("\n=== Column Space Projection ===")
    # Project vector b onto the column space of A using normal equations
    A = np.array([[1., 0.], [0., 1.], [1., 1.]], dtype=float)
    b = np.array([1., 2., 4.])
    # Projection: P = A(A^T A)^{-1} A^T,  proj = P b
    AtA_inv = np.linalg.inv(A.T @ A)
    P = A @ AtA_inv @ A.T
    proj = P @ b
    residual = b - proj
    print(f"  A (3x2):\n{A}")
    print(f"  b = {b}")
    print(f"  proj onto col(A) = {proj.round(4)}")
    print(f"  residual (perp to col space) = {residual.round(6)}")
    print(f"  residual . col0 ~= 0: {np.dot(residual, A[:,0]):.2e}")
    print(f"  residual . col1 ~= 0: {np.dot(residual, A[:,1]):.2e}")
    print(f"  ||residual||^2 = {np.dot(residual, residual):.6f}  (least-squares error)")


def demo_rank_nullity():
    print("\n=== Rank-Nullity Theorem ===")
    # rank(A) + nullity(A) = n (number of columns)
    rng = np.random.default_rng(0)
    cases = [
        ("full-rank 3x4",  rng.normal(size=(3, 4))),
        ("rank-2 4x5",     np.array([[1.,0.,1.,0.,0.],
                                     [0.,1.,0.,1.,0.],
                                     [1.,1.,1.,1.,0.],
                                     [2.,1.,2.,1.,0.]])),
        ("rank-1 2x3",     np.outer([1., 2.], [3., 1., 4.])),
    ]
    for name, A in cases:
        r       = np.linalg.matrix_rank(A)
        n_cols  = A.shape[1]
        nullity = n_cols - r          # rank-nullity theorem definition
        holds   = (r + nullity == n_cols)
        print(f"  {name}: shape={A.shape}, rank={r}, "
              f"nullity={nullity}, rank+nullity={r+nullity}, "
              f"n={n_cols}, theorem: {holds}")


def demo_change_of_basis():
    print("\n=== Change of Basis ===")
    # Express vector v in a new basis B = {b1, b2}
    # Coordinates in new basis: c = B^{-1} v
    b1 = np.array([1., 1.]) / np.sqrt(2)   # 45-degree basis
    b2 = np.array([-1., 1.]) / np.sqrt(2)
    B = np.column_stack([b1, b2])           # change-of-basis matrix
    v = np.array([3., 1.])
    c = np.linalg.solve(B, v)              # coordinates in new basis
    v_reconstructed = B @ c
    print(f"  Standard basis vector v = {v}")
    print(f"  New basis: b1={b1.round(4)}, b2={b2.round(4)}")
    print(f"  Coordinates in new basis: c = {c.round(4)}")
    print(f"  Reconstruct v from c:     B @ c = {v_reconstructed.round(4)}")
    print(f"  Match: {np.allclose(v, v_reconstructed)}")
    # Orthonormal basis: inverse = transpose
    print(f"  B orthonormal check (B^T B = I): err="
          f"{np.linalg.norm(B.T @ B - np.eye(2)):.2e}")


def demo_subspace_dimension():
    print("\n=== Subspace Dimensions and Direct Sum ===")
    # dim(U + V) = dim(U) + dim(V) - dim(U intersect V)
    # Simulate with random matrices
    rng = np.random.default_rng(7)
    # U: 3D subspace of R^5 (columns of U_mat)
    U_mat = rng.normal(size=(5, 3))
    # V: 3D subspace of R^5 with 2 shared directions
    shared = U_mat[:, :2]
    extra  = rng.normal(size=(5, 1))
    V_mat  = np.hstack([shared, extra])
    # dim(U+V) via rank of [U | V]
    UV = np.hstack([U_mat, V_mat])
    dim_U  = np.linalg.matrix_rank(U_mat)
    dim_V  = np.linalg.matrix_rank(V_mat)
    dim_UV = np.linalg.matrix_rank(UV)
    dim_intersect = dim_U + dim_V - dim_UV
    print(f"  dim(U) = {dim_U},  dim(V) = {dim_V}")
    print(f"  dim(U + V) = {dim_UV}  (rank of [U|V])")
    print(f"  dim(U intersect V) = dim(U)+dim(V)-dim(U+V) = {dim_intersect}")
    print(f"  Inclusion-exclusion holds: {dim_U + dim_V - dim_intersect == dim_UV}")
    # R^n as direct sum of col(A) and null(A^T)
    A = rng.normal(size=(4, 5))
    r = np.linalg.matrix_rank(A)
    print(f"\n  For A ({A.shape}): rank={r}, left-nullity={A.shape[0]-r}")
    print(f"  col(A) + left-null(A) = R^{A.shape[0]} (fundamental theorem)")


def demo_linear_combination_span():
    print("\n=== Linear Combinations and Span ===")
    v1 = np.array([1., 0., 1.])
    v2 = np.array([0., 1., 1.])
    print(f"  Basis vectors: v1={v1}, v2={v2}")
    print(f"  Span{{v1, v2}} = 2D plane in R^3")
    # Membership test: least-squares residual near 0 => w in span
    A = np.column_stack([v1, v2])
    print(f"\n  Membership test via least-squares residual:")
    for label, w in [
        ("w=[2,3,5] (2v1+3v2, IN span)", np.array([2., 3., 5.])),
        ("w=[1,1,1] (NOT in span)",       np.array([1., 1., 1.])),
        ("w=[0,0,1] (NOT in span)",       np.array([0., 0., 1.])),
    ]:
        x, _, _, _ = np.linalg.lstsq(A, w, rcond=None)
        err = np.linalg.norm(w - A @ x)
        print(f"    {label}: in_span={err < 1e-9}  (residual={err:.2e})")
    # Redundant vs independent additions to the spanning set
    v3_dep = v1 + 2 * v2                        # in span -> rank stays 2
    v3_ind = np.array([1., -1., 0.])            # independent -> rank becomes 3
    print(f"\n  rank([v1, v2, v1+2v2])  = "
          f"{np.linalg.matrix_rank(np.column_stack([v1, v2, v3_dep]))}  (dependent, rank stays 2)")
    print(f"  rank([v1, v2, [1,-1,0]])= "
          f"{np.linalg.matrix_rank(np.column_stack([v1, v2, v3_ind]))}  (independent, rank = 3)")
    # Projection to find coords in ONB basis
    Q = gram_schmidt(A)
    p = np.array([3., 4., 8.])       # NOT in span: 3v1+4v2=[3,4,7], not [3,4,8]
    coords = Q.T @ p
    proj_p = Q @ coords
    print(f"\n  p={p} (not in span{{v1,v2}})")
    print(f"  Projection onto span: {proj_p.round(4)}")
    print(f"  Proj error (p not in span): {np.linalg.norm(p - proj_p):.4f}")


if __name__ == "__main__":
    demo_linear_independence()
    demo_gram_schmidt()
    demo_null_space()
    demo_feature_space()
    demo_column_space()
    demo_rank_nullity()
    demo_change_of_basis()
    demo_subspace_dimension()
    demo_linear_combination_span()
