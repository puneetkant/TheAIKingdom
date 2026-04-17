"""
Working Example: Vectors
Covers vector operations, norms, dot/cross products, projections,
angles, unit vectors — using both pure Python and NumPy.
"""
import numpy as np
import math


# ── Pure Python vector (for conceptual clarity) ───────────────────────────────
def pure_python_vectors():
    print("=== Vectors — Pure Python ===")

    def vadd(a, b): return [x + y for x, y in zip(a, b)]
    def vsub(a, b): return [x - y for x, y in zip(a, b)]
    def smul(s, v): return [s * x for x in v]
    def dot(a, b):  return sum(x * y for x, y in zip(a, b))
    def norm(v):    return math.sqrt(sum(x**2 for x in v))

    u = [1, 2, 3]
    v = [4, 5, 6]
    print(f"  u         = {u}")
    print(f"  v         = {v}")
    print(f"  u + v     = {vadd(u, v)}")
    print(f"  u - v     = {vsub(u, v)}")
    print(f"  2 * u     = {smul(2, u)}")
    print(f"  dot(u,v)  = {dot(u, v)}")
    print(f"  ||u||     = {norm(u):.4f}")


# ── NumPy vectors ─────────────────────────────────────────────────────────────
def numpy_vectors():
    print("\n=== Vectors — NumPy ===")
    u = np.array([1, 2, 3], dtype=float)
    v = np.array([4, 5, 6], dtype=float)

    print(f"  u              = {u}")
    print(f"  v              = {v}")
    print(f"  u + v          = {u + v}")
    print(f"  u - v          = {u - v}")
    print(f"  3 * u          = {3 * u}")
    print(f"  u · v (dot)    = {np.dot(u, v)}")
    print(f"  u · v (einsum) = {np.einsum('i,i->', u, v)}")
    print(f"  ||u|| (L2)     = {np.linalg.norm(u):.4f}")
    print(f"  ||u|| (L1)     = {np.linalg.norm(u, ord=1):.4f}")
    print(f"  ||u|| (L∞)     = {np.linalg.norm(u, ord=np.inf):.4f}")


# ── Cross product (3D only) ───────────────────────────────────────────────────
def cross_product():
    print("\n=== Cross Product (3D) ===")
    u = np.array([1, 0, 0])   # x-axis
    v = np.array([0, 1, 0])   # y-axis
    w = np.cross(u, v)
    print(f"  u × v = {w}   (should be z-axis [0,0,1])")

    # Area of parallelogram
    a = np.array([3, 0, 0])
    b = np.array([0, 2, 0])
    area = np.linalg.norm(np.cross(a, b))
    print(f"  ||a × b|| = {area}  (area of parallelogram with sides {a}, {b})")


# ── Angle between vectors ─────────────────────────────────────────────────────
def angle_between():
    print("\n=== Angle Between Vectors ===")
    pairs = [
        (np.array([1, 0]), np.array([0, 1])),
        (np.array([1, 1]), np.array([1, 0])),
        (np.array([1, 0]), np.array([-1, 0])),
        (np.array([1, 2, 3]), np.array([4, 5, 6])),
    ]
    for u, v in pairs:
        cos_theta = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
        cos_theta = np.clip(cos_theta, -1, 1)   # guard numerical error
        theta_deg = math.degrees(math.acos(cos_theta))
        print(f"  angle({u}, {v}) = {theta_deg:.2f}°")


# ── Unit vector and projection ────────────────────────────────────────────────
def unit_and_projection():
    print("\n=== Unit Vectors & Projection ===")
    v = np.array([3.0, 4.0])
    unit_v = v / np.linalg.norm(v)
    print(f"  v         = {v}")
    print(f"  unit(v)   = {unit_v}  ||unit||={np.linalg.norm(unit_v):.6f}")

    # Scalar and vector projection of u onto v
    u = np.array([6.0, 2.0])
    scalar_proj = np.dot(u, unit_v)
    vector_proj = scalar_proj * unit_v
    print(f"\n  u         = {u}")
    print(f"  scalar proj of u onto v = {scalar_proj:.4f}")
    print(f"  vector proj of u onto v = {vector_proj}")

    # Orthogonal decomposition
    perpendicular = u - vector_proj
    print(f"  u = proj + perp: {vector_proj} + {perpendicular}")
    print(f"  perpendicular · v = {np.dot(perpendicular, v):.10f}  (≈ 0)")


# ── Linear (in)dependence ─────────────────────────────────────────────────────
def linear_dependence():
    print("\n=== Linear (In)dependence ===")
    # 3 vectors in R³ — rank tells us about dependence
    sets = [
        np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),   # independent (basis)
        np.array([[1, 2, 3], [2, 4, 6], [0, 1, 0]]),   # row 2 = 2*row 1
        np.array([[1, 2], [3, 4], [5, 6]]),             # 3 vectors in R²
    ]
    for M in sets:
        rank = np.linalg.matrix_rank(M)
        n    = M.shape[0]
        dep  = "(dependent)" if rank < n else "(independent)"
        print(f"  rank={rank}/{n}  {dep}  vectors={M.tolist()}")


# ── Practical: cosine similarity ─────────────────────────────────────────────
def cosine_similarity():
    print("\n=== Cosine Similarity (NLP use-case) ===")
    docs = {
        "Python programming language": np.array([1, 1, 0, 0, 1]),
        "Java programming language":   np.array([0, 1, 1, 0, 1]),
        "Machine learning algorithms": np.array([0, 0, 0, 1, 0]),
    }
    names = list(docs.keys())
    vecs  = list(docs.values())
    print("  Pairwise cosine similarity:")
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            u, v = vecs[i], vecs[j]
            sim = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9)
            print(f"    {names[i][:25]:<28} ↔ {names[j][:25]:<28}  sim={sim:.3f}")


if __name__ == "__main__":
    pure_python_vectors()
    numpy_vectors()
    cross_product()
    angle_between()
    unit_and_projection()
    linear_dependence()
    cosine_similarity()
