"""
Working Example 2: Vectors — ML-focused vector operations with NumPy
====================================================================
Downloads California Housing data and demonstrates vectors in ML context:
  - Vector construction and basic ops (add, scale, dot)
  - Norms (L1, L2, Linf) and normalisation
  - Cosine similarity for feature comparison
  - Basis decomposition and projections
  - Gradient vectors in SGD update
  - Vectorised computation speed vs Python loops

Run:  python working_example2.py
"""
import math
import time
import urllib.request
from pathlib import Path

try:
    import numpy as np
    import matplotlib.pyplot as plt
except ImportError:
    print("pip install numpy matplotlib"); raise SystemExit(1)

DATA   = Path(__file__).parent / "data"
OUTPUT = Path(__file__).parent / "output"
DATA.mkdir(exist_ok=True); OUTPUT.mkdir(exist_ok=True)


# ── Download ────────────────────────────────────────────────────────────────
def download_cal_housing() -> Path:
    dest = DATA / "cal_housing.csv"
    if dest.exists(): return dest
    try:
        urllib.request.urlretrieve(
            "https://huggingface.co/datasets/scikit-learn/california-housing/resolve/main/cal_housing.csv",
            dest
        )
        print(f"Downloaded {dest.name}")
    except Exception:
        import csv
        rows = [["MedInc","HouseAge","AveRooms","AveBedrms","Population","AveOccup","Latitude","Longitude","MedHouseVal"]]
        rng = __import__("random"); rng.seed(42)
        for _ in range(200):
            rows.append([round(rng.uniform(1,10),4), rng.randint(1,50), round(rng.uniform(3,8),4),
                         round(rng.uniform(0.8,2),4), rng.randint(100,5000), round(rng.uniform(2,5),4),
                         round(rng.uniform(32,42),4), round(rng.uniform(-124,-114),4), round(rng.uniform(0.5,5),4)])
        dest.write_text("\n".join(",".join(str(x) for x in r) for r in rows))
    return dest


def load_feature_matrix(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    import csv
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    feature_cols = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup"]
    X = np.array([[float(r[c]) for c in feature_cols] for r in rows])
    y = np.array([float(r["MedHouseVal"]) for r in rows])
    return X, y, feature_cols


# ── 1. Basic vector ops ───────────────────────────────────────────────────────
def demo_basic_ops() -> None:
    print("=== Basic Vector Operations ===")
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])

    print(f"  a = {a}")
    print(f"  b = {b}")
    print(f"  a + b   = {a + b}")
    print(f"  2 * a   = {2 * a}")
    print(f"  a · b   = {np.dot(a, b):.4f}  (dot product)")
    print(f"  |a|₂    = {np.linalg.norm(a):.4f}  (L2 norm)")
    print(f"  |a|₁    = {np.linalg.norm(a, 1):.4f}  (L1 norm)")
    print(f"  |a|∞    = {np.linalg.norm(a, np.inf):.4f}  (L∞ norm)")


# ── 2. Norms and normalisation (feature scaling) ─────────────────────────────
def demo_normalisation(X: np.ndarray, col_names: list[str]) -> None:
    print("\n=== Norms & Normalisation ===")
    # L2-normalise each sample (row)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X_unit = X / (norms + 1e-9)
    print(f"  Original row norm: mean={np.mean(norms):.3f}")
    print(f"  After L2-norm:     mean={np.mean(np.linalg.norm(X_unit, axis=1)):.6f}  (≈ 1.0)")

    # Min-max normalise each column
    col_min  = X.min(axis=0)
    col_max  = X.max(axis=0)
    X_minmax = (X - col_min) / (col_max - col_min + 1e-9)
    print(f"  Min-max col ranges: {X_minmax.min(axis=0).round(3)} … {X_minmax.max(axis=0).round(3)}")


# ── 3. Cosine similarity (feature comparison) ─────────────────────────────────
def demo_cosine_similarity(X: np.ndarray) -> None:
    print("\n=== Cosine Similarity ===")
    def cosine(u: np.ndarray, v: np.ndarray) -> float:
        return float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9))

    # Compare first 3 samples
    for i in range(3):
        for j in range(i + 1, 3):
            sim = cosine(X[i], X[j])
            print(f"  cos(x[{i}], x[{j}]) = {sim:.4f}")


# ── 4. Vector projection ───────────────────────────────────────────────────────
def demo_projection() -> None:
    print("\n=== Vector Projection ===")
    # Project b onto a: proj_a(b) = (a·b/|a|²) * a
    a = np.array([3.0, 0.0])
    b = np.array([2.0, 4.0])
    proj = (np.dot(a, b) / np.dot(a, a)) * a
    rejection = b - proj
    print(f"  a = {a},  b = {b}")
    print(f"  proj_a(b) = {proj}  (parallel component)")
    print(f"  rejection = {rejection}  (perpendicular component)")
    print(f"  Check dot(proj, rejection) ≈ 0: {np.dot(proj, rejection):.10f}")


# ── 5. SGD gradient vector demo ───────────────────────────────────────────────
def demo_sgd_gradient(X: np.ndarray, y: np.ndarray) -> None:
    print("\n=== SGD Gradient Vector ===")
    n, d = X.shape
    w = np.zeros(d)
    lr = 0.001
    losses = []

    for epoch in range(50):
        y_hat  = X @ w
        error  = y_hat - y
        grad   = X.T @ error / n    # gradient vector ∈ ℝ^d
        w     -= lr * grad
        loss   = float(np.mean(error ** 2))
        losses.append(loss)

    print(f"  Initial loss: {losses[0]:.4f}")
    print(f"  Final loss:   {losses[-1]:.4f}")
    print(f"  Gradient norm at end: {np.linalg.norm(grad):.6f}")
    print(f"  Weight vector: {w.round(4)}")


# ── 6. Speed: vectorised vs Python loops ─────────────────────────────────────
def demo_speed() -> None:
    print("\n=== Speed: Vectorised vs Python Loops ===")
    n = 10_000
    a = np.random.randn(n)
    b = np.random.randn(n)

    t0 = time.perf_counter()
    result_py = sum(ai * bi for ai, bi in zip(a.tolist(), b.tolist()))
    t_py = time.perf_counter() - t0

    t0 = time.perf_counter()
    result_np = np.dot(a, b)
    t_np = time.perf_counter() - t0

    print(f"  Python loop: {t_py*1000:.3f}ms  result={result_py:.6f}")
    print(f"  NumPy dot:   {t_np*1000:.3f}ms  result={result_np:.6f}")
    print(f"  Speedup: {t_py/max(t_np, 1e-9):.1f}×")


# ── 7. Visualise 2-D vectors ─────────────────────────────────────────────────
def plot_vectors() -> None:
    origin = np.zeros(2)
    vectors = {
        "a=(3,1)": np.array([3.0, 1.0]),
        "b=(1,3)": np.array([1.0, 3.0]),
        "a+b":     np.array([4.0, 4.0]),
        "proj":    np.array([3.0 * np.dot([3,1],[1,3]) / np.dot([3,1],[3,1]), 1.0 * np.dot([3,1],[1,3]) / np.dot([3,1],[3,1])]),
    }
    colors = ["steelblue", "coral", "green", "purple"]
    fig, ax = plt.subplots(figsize=(6, 6))
    for (label, v), c in zip(vectors.items(), colors):
        ax.quiver(0, 0, v[0], v[1], angles="xy", scale_units="xy", scale=1,
                  color=c, label=label, width=0.015)
    ax.set_xlim(-1, 6); ax.set_ylim(-1, 6)
    ax.set_aspect("equal"); ax.grid(alpha=0.3)
    ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
    ax.legend(); ax.set_title("2D Vectors and Projection")
    fig.savefig(OUTPUT / "vectors.png", dpi=120, bbox_inches="tight")
    print(f"\n  Saved: vectors.png")
    plt.close(fig)


if __name__ == "__main__":
    path = download_cal_housing()
    X, y, cols = load_feature_matrix(path)
    print(f"Feature matrix: {X.shape}  (n_samples × n_features)")

    demo_basic_ops()
    demo_normalisation(X, cols)
    demo_cosine_similarity(X)
    demo_projection()
    demo_sgd_gradient(X, y)
    demo_speed()
    plot_vectors()
