"""
Working Example: Inner Product Spaces
Covers inner products, norms, Cauchy-Schwarz, orthogonality,
projections, Gram-Schmidt, QR, and function spaces.
"""
import numpy as np


# ── 1. Inner product axioms ───────────────────────────────────────────────────
def inner_product_axioms():
    print("=== Inner Product Axioms ===")
    print("  An inner product ⟨·,·⟩ on vector space V must satisfy:")
    print("    1. Linearity in first arg:   ⟨au+bv, w⟩ = a⟨u,w⟩ + b⟨v,w⟩")
    print("    2. Conjugate symmetry:       ⟨u,v⟩ = conj(⟨v,u⟩)")
    print("    3. Positive definiteness:    ⟨v,v⟩ ≥ 0, = 0 iff v = 0\n")

    u = np.array([1., 2., 3.])
    v = np.array([4., 5., 6.])
    w = np.array([7., 8., 9.])
    a, b = 2.0, 3.0

    ip = lambda x, y: np.dot(x, y)     # standard inner product

    print(f"  u={u}, v={v}, w={w}")
    print(f"  Linearity:  ⟨au+bv, w⟩ = {ip(a*u+b*v, w):.4f}")
    print(f"              a⟨u,w⟩+b⟨v,w⟩ = {a*ip(u,w)+b*ip(v,w):.4f}")
    print(f"  Symmetry:   ⟨u,v⟩ = {ip(u,v):.4f},  ⟨v,u⟩ = {ip(v,u):.4f}")
    print(f"  Pos-def:    ⟨u,u⟩ = {ip(u,u):.4f}  (≥0 ✓)")


# ── 2. Induced norm and Cauchy-Schwarz ────────────────────────────────────────
def cauchy_schwarz():
    print("\n=== Cauchy-Schwarz Inequality |⟨u,v⟩| ≤ ||u|| ||v|| ===")
    pairs = [
        (np.array([1., 0., 0.]), np.array([0., 1., 0.])),   # orthogonal
        (np.array([1., 2., 3.]), np.array([2., 4., 6.])),   # parallel
        (np.array([1., 2., 3.]), np.array([4., 5., 6.])),   # general
    ]
    for u, v in pairs:
        lhs = abs(np.dot(u, v))
        rhs = np.linalg.norm(u) * np.linalg.norm(v)
        ok  = lhs <= rhs + 1e-12
        cos = lhs / rhs
        print(f"  u={u} v={v}")
        print(f"    |⟨u,v⟩|={lhs:.4f}  ||u||·||v||={rhs:.4f}  satisfied={ok}  cos θ={cos:.4f}\n")


# ── 3. Orthogonality and orthogonal complement ────────────────────────────────
def orthogonality():
    print("=== Orthogonality ===")
    u = np.array([1., 0., 0.])
    v = np.array([0., 1., 0.])
    w = np.array([0., 0., 1.])
    print(f"  e1·e2={np.dot(u,v):.0f}, e1·e3={np.dot(u,w):.0f}, e2·e3={np.dot(v,w):.0f}  (standard basis is orthonormal)")

    # Mutual orthogonality of eigenvectors of symmetric matrix
    A = np.array([[3, 1, 0],
                  [1, 3, 0],
                  [0, 0, 5]], dtype=float)
    vals, vecs = np.linalg.eigh(A)
    print(f"\n  Eigenvectors of symmetric A:")
    for i in range(vecs.shape[1]):
        for j in range(i+1, vecs.shape[1]):
            dot = np.dot(vecs[:,i], vecs[:,j])
            print(f"    v{i+1}·v{j+1} = {dot:.2e}  (should ≈ 0)")


# ── 4. Orthogonal projection ──────────────────────────────────────────────────
def projections():
    print("\n=== Projections ===")
    # Project v onto u
    u = np.array([1., 1., 0.])
    v = np.array([3., 1., 2.])
    proj_v_onto_u = (np.dot(v, u) / np.dot(u, u)) * u
    perp          = v - proj_v_onto_u

    print(f"  v = {v}, u = {u}")
    print(f"  proj_u(v)  = {proj_v_onto_u}")
    print(f"  v - proj   = {perp}  (orthogonal to u? {np.isclose(np.dot(perp, u), 0)})")

    # Projection matrix P = uu^T / u^Tu
    u_col = u.reshape(-1, 1)
    P     = u_col @ u_col.T / np.dot(u, u)
    print(f"\n  Projection matrix P:\n{np.round(P, 4)}")
    print(f"  P is idempotent (P²=P): {np.allclose(P @ P, P)}")
    print(f"  P is symmetric  (Pᵀ=P): {np.allclose(P, P.T)}")

    # Projection onto subspace spanned by two vectors
    basis = np.array([[1., 0., 0.], [0., 1., 0.]]).T   # xy-plane
    P_sub = basis @ np.linalg.inv(basis.T @ basis) @ basis.T
    w     = np.array([3., 4., 5.])
    print(f"\n  Project w={w} onto xy-plane:")
    print(f"  P w = {P_sub @ w}  (should be [3,4,0])")


# ── 5. Gram-Schmidt and QR ────────────────────────────────────────────────────
def gram_schmidt(A):
    """Gram-Schmidt on columns of A → returns Q, R."""
    m, n = A.shape
    Q = np.zeros_like(A)
    R = np.zeros((n, n))
    for j in range(n):
        v = A[:, j].copy()
        for i in range(j):
            R[i, j] = np.dot(Q[:, i], A[:, j])
            v -= R[i, j] * Q[:, i]
        R[j, j] = np.linalg.norm(v)
        Q[:, j] = v / R[j, j]
    return Q, R


def gram_schmidt_demo():
    print("\n=== Gram-Schmidt → QR ===")
    A = np.array([[1., 1., 0.],
                  [1., 0., 1.],
                  [0., 1., 1.]])
    Q, R = gram_schmidt(A)
    Q_np, R_np = np.linalg.qr(A)
    print(f"  A:\n{A}")
    print(f"  Q (our GS):\n{np.round(Q, 4)}")
    print(f"  Q orthonormal: {np.allclose(Q.T @ Q, np.eye(3))}")
    print(f"  QR = A:        {np.allclose(Q @ R, A)}")
    print(f"  matches np QR: {np.allclose(np.abs(Q), np.abs(Q_np))}")


# ── 6. Weighted inner product ─────────────────────────────────────────────────
def weighted_inner_product():
    print("\n=== Weighted Inner Product ⟨u,v⟩_W = uᵀWv ===")
    W = np.diag([1., 2., 3.])   # positive definite weight matrix
    u = np.array([1., 1., 1.])
    v = np.array([2., 0., 1.])
    ip_w = u @ W @ v
    ip_std = np.dot(u, v)
    print(f"  W = diag(1,2,3),  u={u},  v={v}")
    print(f"  standard ⟨u,v⟩  = {ip_std}")
    print(f"  weighted  ⟨u,v⟩_W = {ip_w}")
    # Induced norm
    norm_w = np.sqrt(u @ W @ u)
    print(f"  weighted ||u||_W = sqrt(⟨u,u⟩_W) = {norm_w:.4f}")


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    inner_product_axioms()
    cauchy_schwarz()
    orthogonality()
    projections()
    gram_schmidt_demo()
    weighted_inner_product()
