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
                    ("Singular",  np.array([[1.,2.],[2.,4.]]))):
        d = np.linalg.det(A)
        inv = "invertible" if abs(d) > 1e-10 else "SINGULAR"
        print(f"  {name}: det={d:.4f}  → {inv}")

def demo_geometric_area():
    print("\n=== Geometric: Parallelogram Area ===")
    # Area of parallelogram spanned by vectors a, b = |det([a|b])|
    a = np.array([3., 0.])
    b = np.array([1., 2.])
    area = abs(np.linalg.det(np.column_stack([a, b])))
    print(f"  a={a}, b={b} → area = |det| = {area:.4f}")

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
    print(f"  rank(A) = {np.linalg.matrix_rank(A)}")  # <3 → near-singular

if __name__ == "__main__":
    demo_det_properties()
    demo_invertibility()
    demo_geometric_area()
    demo_3x3_and_cofactor()
