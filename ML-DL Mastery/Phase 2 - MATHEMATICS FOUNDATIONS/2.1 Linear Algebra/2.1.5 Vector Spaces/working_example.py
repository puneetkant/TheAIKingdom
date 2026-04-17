"""
Working Example: Vector Spaces
Covers subspaces, span, basis, dimension, null space, column space,
row space, change of basis, and coordinate representations.
"""
import numpy as np


# ── Checking subspace axioms ──────────────────────────────────────────────────
def subspace_checks():
    print("=== Subspace Axioms ===")
    print("  A subset W of Rⁿ is a subspace if:")
    print("    1. Zero vector 0 ∈ W")
    print("    2. Closed under addition: u,v ∈ W → u+v ∈ W")
    print("    3. Closed under scalar multiplication: c∈R, v∈W → cv ∈ W\n")

    def is_subspace(name, check_zero, check_add, check_scalar):
        all_ok = check_zero and check_add and check_scalar
        status = "IS a subspace" if all_ok else "NOT a subspace"
        print(f"  {name}: {status}")
        if not check_zero:   print("    ✗ fails: zero vector not included")
        if not check_add:    print("    ✗ fails: not closed under addition")
        if not check_scalar: print("    ✗ fails: not closed under scalar mult")

    # Span of a set of vectors — always a subspace
    is_subspace("span({v1,v2}) in R³",      True, True, True)
    # Set of vectors with positive entries — not closed under scalar mult
    is_subspace("vectors with all x_i > 0", False, False, False)
    # Plane through origin
    is_subspace("plane ax+by+cz=0",         True, True, True)
    # Plane NOT through origin (ax+by+cz=d, d≠0)
    is_subspace("plane ax+by+cz=1",         False, False, True)


# ── Span and linear independence ──────────────────────────────────────────────
def span_and_independence():
    print("\n=== Span & Linear Independence ===")
    v1 = np.array([1, 2, 3], dtype=float)
    v2 = np.array([4, 5, 6], dtype=float)
    v3 = v1 + v2                            # linearly dependent
    v4 = np.array([0, 0, 1], dtype=float)   # independent of v1, v2

    sets = {
        "{v1}":         [v1],
        "{v1, v2}":     [v1, v2],
        "{v1, v2, v3}": [v1, v2, v3],       # v3 = v1+v2
        "{v1, v2, v4}": [v1, v2, v4],
    }
    for label, vecs in sets.items():
        M    = np.array(vecs)
        rank = np.linalg.matrix_rank(M)
        n    = len(vecs)
        dep  = "(dependent)" if rank < n else "(independent)"
        print(f"  {label:<22}: rank={rank}/{n}  {dep}")


# ── Finding a basis ───────────────────────────────────────────────────────────
def find_basis():
    print("\n=== Finding a Basis (Column Space) ===")
    A = np.array([
        [1, 2, 3, 4],
        [2, 4, 6, 8],   # 2 × row 0
        [0, 1, 2, 3],
        [1, 3, 5, 7],   # row0 + row2
    ], dtype=float)
    print(f"  A:\n{A}")
    rank = np.linalg.matrix_rank(A)
    print(f"  rank(A) = {rank}  → basis of column space has {rank} vectors")

    # Use QR to extract basis
    Q, R = np.linalg.qr(A)
    # Pivot columns via SVD
    U, s, Vt = np.linalg.svd(A)
    tol = max(A.shape) * np.finfo(float).eps * s[0]
    num_basis = (s > tol).sum()
    print(f"  singular values: {np.round(s, 4)}")
    print(f"  basis vectors (first {num_basis} left singular vectors):")
    for i in range(num_basis):
        print(f"    {np.round(U[:, i], 4)}")


# ── Four fundamental subspaces ────────────────────────────────────────────────
def four_subspaces():
    print("\n=== Four Fundamental Subspaces ===")
    A = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]], dtype=float)
    m, n = A.shape
    r    = np.linalg.matrix_rank(A)

    print(f"  A is {m}×{n}, rank r={r}")
    print(f"\n  Column space C(A)  : dim = r = {r}")
    print(f"  Null space  N(A)   : dim = n-r = {n-r}")
    print(f"  Row space   C(Aᵀ)  : dim = r = {r}")
    print(f"  Left null   N(Aᵀ)  : dim = m-r = {m-r}")

    # Null space via SVD
    U, s, Vt = np.linalg.svd(A)
    tol = max(A.shape) * np.finfo(float).eps * s[0]
    null_vecs = Vt[s < tol]   # rows of Vt where singular value ≈ 0
    print(f"\n  Null space basis vectors:")
    for v in null_vecs:
        print(f"    {np.round(v, 6)}  → A @ v = {np.round(A @ v, 10)}")


# ── Change of basis ───────────────────────────────────────────────────────────
def change_of_basis():
    print("\n=== Change of Basis ===")
    # Standard basis e1, e2 → new basis b1, b2
    b1 = np.array([1, 1], dtype=float)
    b2 = np.array([1, -1], dtype=float)
    P  = np.column_stack([b1, b2])    # columns = new basis vectors

    v_std = np.array([3, 1], dtype=float)   # vector in standard coords
    v_new = np.linalg.inv(P) @ v_std        # coords in new basis

    print(f"  new basis: b1={b1}, b2={b2}")
    print(f"  v in standard coords: {v_std}")
    print(f"  v in new basis coords: {v_new}")
    print(f"  reconstruct: {v_new[0]}*b1 + {v_new[1]}*b2 = {v_new[0]*b1 + v_new[1]*b2}")

    # Transform a matrix to new basis
    A_std = np.array([[2, 0], [0, 3]], dtype=float)
    A_new = np.linalg.inv(P) @ A_std @ P
    print(f"\n  A in standard basis:\n{A_std}")
    print(f"  A in new basis:\n{np.round(A_new, 4)}")


# ── Gram-Schmidt orthogonalisation ────────────────────────────────────────────
def gram_schmidt(vecs):
    Q = []
    for v in vecs:
        u = v.astype(float).copy()
        for q in Q:
            u -= np.dot(u, q) * q
        norm = np.linalg.norm(u)
        if norm > 1e-10:
            Q.append(u / norm)
    return np.array(Q)


def gram_schmidt_demo():
    print("\n=== Gram-Schmidt Orthogonalisation ===")
    vecs = [np.array([1, 1, 0.]),
            np.array([1, 0, 1.]),
            np.array([0, 1, 1.])]
    Q = gram_schmidt(vecs)
    print(f"  orthonormal basis Q:\n{np.round(Q, 4)}")
    print(f"  Q @ Q.T ≈ I: {np.allclose(Q @ Q.T, np.eye(3))}")


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    subspace_checks()
    span_and_independence()
    find_basis()
    four_subspaces()
    change_of_basis()
    gram_schmidt_demo()
