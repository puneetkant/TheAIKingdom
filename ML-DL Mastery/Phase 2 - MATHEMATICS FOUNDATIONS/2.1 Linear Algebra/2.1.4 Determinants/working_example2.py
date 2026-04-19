"""
Working Example 2: Determinants — Properties and ML Applications
================================================================
Demonstrates: det via np.linalg.det, properties, geometric interpretation,
invertibility check, volume scaling, Jacobian determinant in change-of-variable.

Run:  python working_example2.py
"""
from pathlib import Path
try:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit("pip install numpy matplotlib")

OUTPUT = Path(__file__).parent / "output"
OUTPUT.mkdir(exist_ok=True)

def demo_det_properties():
    print("=== Determinant Properties ===")
    A = np.array([[2., 1.], [5., 3.]])
    B = np.array([[1., 2.], [3., 4.]])
    print(f"  det(A) = {np.linalg.det(A):.4f}")
    print(f"  det(A.T) = {np.linalg.det(A.T):.4f}  (= det(A))")
    print(f"  det(AB) = {np.linalg.det(A@B):.4f}")
    print(f"  det(A)*det(B) = {np.linalg.det(A)*np.linalg.det(B):.4f}  (multiplicative)")
    print(f"  det(2A) = {np.linalg.det(2*A):.4f}  (= 2^n * det(A) = {2**2 * np.linalg.det(A):.4f})")

def demo_invertibility():
    print("\n=== Invertibility Check ===")
    for name, A in [("Full rank", np.array([[1.,2.],[3.,4.]])),
                    ("Singular",  np.array([[1.,2.],[2.,4.]]))]:         
        d = np.linalg.det(A)
        inv = "invertible" if abs(d) > 1e-10 else "SINGULAR"
        print(f"  {name}: det={d:.4f}  -> {inv}")

def demo_geometric_area():
    print("\n=== Geometric: Parallelogram Area ===")
    # Area of parallelogram spanned by vectors a, b = |det([a|b])|
    a = np.array([3., 0.])
    b = np.array([1., 2.])
    area = abs(np.linalg.det(np.column_stack([a, b])))
    print(f"  a={a}, b={b} -> area = |det| = {area:.4f}")

    fig, ax = plt.subplots(figsize=(5,5))
    corners = np.array([[0,0], a, a+b, b, [0,0]])
    ax.fill(corners[:,0], corners[:,1], alpha=0.3, color="steelblue", label=f"Area={area:.2f}")
    for v, c, l in [(a, "blue","a"), (b, "red","b")]:
        ax.quiver(0,0,v[0],v[1],angles="xy",scale_units="xy",scale=1,color=c,label=l,width=0.02)
    ax.set_xlim(-0.5,5); ax.set_ylim(-0.5,3); ax.set_aspect("equal"); ax.grid(0.3); ax.legend()
    ax.set_title("Parallelogram = |det([a|b])|")
    fig.savefig(OUTPUT/"det_parallelogram.png", dpi=120, bbox_inches="tight")
    plt.close(fig); print(f"  Saved: det_parallelogram.png")

def demo_3x3_and_cofactor():
    print("\n=== 3×3 Determinant ===")
    A = np.array([[1.,2.,3.],[4.,5.,6.],[7.,8.,10.]])
    print(f"  A =\n{A}")
    print(f"  det(A) = {np.linalg.det(A):.4f}")
    print(f"  rank(A) = {np.linalg.matrix_rank(A)}")  # <3 -> near-singular

def demo_cramers_rule():
    print("\n=== Cramer's Rule (2x2 System) ===")
    # Solve: 2x + y = 5,  x + 3y = 10
    A = np.array([[2., 1.], [1., 3.]])
    b = np.array([5., 10.])
    det_A = np.linalg.det(A)
    A1 = A.copy(); A1[:, 0] = b
    A2 = A.copy(); A2[:, 1] = b
    x1 = np.linalg.det(A1) / det_A
    x2 = np.linalg.det(A2) / det_A
    print(f"  System: 2x + y = 5,  x + 3y = 10")
    print(f"  det(A)        = {det_A:.4f}")
    print(f"  x1 (Cramer's) = {x1:.4f}")
    print(f"  x2 (Cramer's) = {x2:.4f}")
    x_np = np.linalg.solve(A, b)
    print(f"  numpy solve:   x1={x_np[0]:.4f}, x2={x_np[1]:.4f}")
    print(f"  Match: {np.allclose([x1, x2], x_np)}")


def demo_eigenvalue_det_relation():
    print("\n=== det = Product of Eigenvalues ===")
    A = np.array([[4., 2.], [1., 3.]])
    eigenvalues = np.linalg.eigvals(A)
    det_direct = np.linalg.det(A)
    det_via_eig = np.prod(eigenvalues).real
    print(f"  A = {A.tolist()}")
    print(f"  Eigenvalues: {eigenvalues.real.round(4)}")
    print(f"  det(A) direct     = {det_direct:.6f}")
    print(f"  prod(eigenvalues) = {det_via_eig:.6f}")
    print(f"  Match: {np.isclose(det_direct, det_via_eig)}")
    # Upper-triangular: eigenvalues are diagonal entries
    B = np.array([[6., 1., 2.], [0., 3., 1.], [0., 0., 2.]])
    eigs_B = np.linalg.eigvals(B)
    print(f"  Upper-triangular 3x3: det={np.linalg.det(B):.4f}, "
          f"prod(eigs)={np.prod(eigs_B).real:.4f}")


def demo_lu_determinant():
    print("\n=== Determinant via LU Decomposition ===")
    try:
        from scipy.linalg import lu
        A = np.array([[3., 1., 2.], [6., 3., 4.], [3., 1., 5.]])
        P, L, U = lu(A)
        # det(A) = det(P^-1) * det(L) * det(U)
        # det(L) = 1 for unit lower triangular; det(U) = product of diag
        det_P = np.linalg.det(P)
        det_U = np.prod(np.diag(U))
        det_via_lu = det_P * det_U
        det_direct = np.linalg.det(A)
        print(f"  A =\n{A}")
        print(f"  det(P) = {det_P:.4f}  (permutation sign)")
        print(f"  det(U) = {det_U:.6f}  (product of U diagonal)")
        print(f"  det via LU = {det_via_lu:.6f}")
        print(f"  det direct = {det_direct:.6f}")
        print(f"  Match: {np.isclose(det_via_lu, det_direct)}")
    except ImportError:
        print("  scipy not available -- skipping LU det demo")


def demo_det_geometric_scaling():
    print("\n=== det as Volume Scaling Factor ===")
    # |det(kI)| = k^n  — scaling all sides by k scales volume by k^n
    print(f"  {'k':>6}  {'|det(kI_2)|':>12}  {'k^2':>8}")
    for k in [0.5, 1.0, 2.0, 3.0, 5.0]:
        d = abs(np.linalg.det(k * np.eye(2)))
        print(f"  {k:>6.2f}  {d:>12.4f}  {k**2:>8.4f}")
    # |det(A)| = product of singular values
    A = np.array([[2., 1., 0.], [0., 3., 1.], [1., 0., 2.]])
    sv = np.linalg.svd(A, compute_uv=False)
    print(f"\n  A (3x3): det={np.linalg.det(A):.4f}")
    print(f"  Singular values: {sv.round(4)}")
    print(f"  prod(sv) = {np.prod(sv):.4f}  (= |det(A)|)")
    print(f"  Match: {np.isclose(abs(np.linalg.det(A)), np.prod(sv))}")


def demo_det_condition_number():
    print("\n=== Near-Singularity: det vs Condition Number ===")
    # A = [[1,2],[3,6+eps]]: det = eps, cond -> inf as eps -> 0
    print(f"  {'eps':>10}  {'det(A)':>12}  {'cond(A)':>14}  {'status':>10}")
    for eps in [1.0, 0.1, 0.01, 1e-4, 1e-8, 1e-14]:
        A = np.array([[1., 2.], [3., 6. + eps]])
        d = np.linalg.det(A)
        c = np.linalg.cond(A)
        status = "OK" if c < 1e6 else ("ill-cond" if c < 1e12 else "singular")
        print(f"  {eps:>10.2e}  {d:>12.4e}  {c:>14.4e}  {status:>10}")
    print("  (small det alone does not indicate ill-conditioning -- scale matters)")
    # Scaled example: large det but poorly conditioned
    B = np.array([[1e8, 1e8 + 1e-8], [1e8 - 1e-8, 1e8]])
    print(f"\n  Large-scale B: det={np.linalg.det(B):.4e}, cond={np.linalg.cond(B):.4e}")


def demo_adjugate_and_trace():
    print("\n=== Adjugate Matrix and Trace-Eigenvalue Relations ===")
    # adj(A) is the transpose of the cofactor matrix; A * adj(A) = det(A) * I
    A = np.array([[1., 2., 3.], [0., 4., 5.], [1., 0., 6.]])
    n = A.shape[0]
    det_A = np.linalg.det(A)
    adj = np.zeros_like(A)
    for i in range(n):
        for j in range(n):
            minor = np.delete(np.delete(A, j, 0), i, 1)
            adj[i, j] = ((-1) ** (i + j)) * np.linalg.det(minor)
    print(f"  A =\n{A}")
    print(f"  det(A) = {det_A:.4f}")
    print(f"  adj(A):\n{adj.round(4)}")
    err_inv = np.linalg.norm(adj / det_A - np.linalg.inv(A))
    err_id  = np.linalg.norm(adj @ A - det_A * np.eye(n))
    print(f"  adj(A)/det(A) ~= inv(A):  err={err_inv:.2e}")
    print(f"  adj(A) @ A    = det(A)*I: err={err_id:.2e}")
    # Trace = sum of eigenvalues  (det = product of eigenvalues shown earlier)
    print(f"\n  Trace and Eigenvalue Identities:")
    for mat, label in [(A, "A"), (A.T @ A, "A^T A"), (A + A.T, "A+A^T")]:
        tr      = float(np.trace(mat))
        eig_sum = float(np.sum(np.linalg.eigvals(mat).real))
        det_m   = float(np.linalg.det(mat))
        eig_prod = float(np.prod(np.linalg.eigvals(mat)).real)
        print(f"    {label:<8}: trace={tr:>8.4f}==sum(eigs)={eig_sum:>8.4f} "
              f" det={det_m:>10.4f}==prod(eigs)={eig_prod:>10.4f}")
    # Cayley-Hamilton: A satisfies its own characteristic polynomial
    M = np.array([[4., 2.], [1., 3.]])
    CH = M @ M - np.trace(M) * M + np.linalg.det(M) * np.eye(2)
    print(f"\n  Cayley-Hamilton 2x2: A^2 - tr*A + det*I = 0  "
          f"(max|err|={np.max(np.abs(CH)):.2e})")
    print("  (Characteristic poly: det(lam*I - A) = lam^n - tr*lam^(n-1) + ... +/- det)")


if __name__ == "__main__":
    demo_det_properties()
    demo_invertibility()
    demo_geometric_area()
    demo_3x3_and_cofactor()
    demo_cramers_rule()
    demo_eigenvalue_det_relation()
    demo_lu_determinant()
    demo_det_geometric_scaling()
    demo_det_condition_number()
    demo_adjugate_and_trace()
