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
    print(f"  Q columns orthonormal check (Q^T Q ≈ I):\n{(Q.T @ Q).round(6)}")

def demo_null_space():
    print("\n=== Null Space (via SVD) ===")
    # A matrix with a non-trivial null space
    A = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]], dtype=float)
    U, s, Vt = np.linalg.svd(A)
    null_mask = np.abs(s) < 1e-10
    # Zero singular values → corresponding right singular vectors span null space
    # For 3×3 with rank 2, the 3rd row of Vt is in null space
    null_vecs = Vt[s < 1e-10] if any(s < 1e-10) else Vt[-1:]
    print(f"  Singular values: {s.round(6)}")
    print(f"  rank(A) = {np.linalg.matrix_rank(A)}")
    print(f"  Null vector approx: {Vt[-1].round(4)}")
    print(f"  A @ null_vec ≈ 0: {(A @ Vt[-1]).round(10)}")

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
    print(f"  Rank: {np.linalg.matrix_rank(X)}  (should be 4 — full column rank)")
    # Add a redundant feature
    X_red = np.hstack([X, (X[:,0] + X[:,1]).reshape(-1,1)])
    print(f"  With redundant col rank: {np.linalg.matrix_rank(X_red)}  (still 4)")

if __name__ == "__main__":
    demo_linear_independence()
    demo_gram_schmidt()
    demo_null_space()
    demo_feature_space()
