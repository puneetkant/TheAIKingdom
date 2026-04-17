"""
Working Example 2: SVD — Dimensionality Reduction and Matrix Approximation
==========================================================================
np.linalg.svd on Cal Housing: truncated SVD, low-rank approximation,
PCA via SVD, condition number, pseudo-inverse, LSA text demo.

Run:  python working_example2.py
"""
import csv, urllib.request
from pathlib import Path
try:
    import numpy as np
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit("pip install numpy matplotlib")

DATA   = Path(__file__).parent / "data"
OUTPUT = Path(__file__).parent / "output"
DATA.mkdir(exist_ok=True); OUTPUT.mkdir(exist_ok=True)

def download():
    dest = DATA / "cal_housing.csv"
    if not dest.exists():
        try: urllib.request.urlretrieve(
            "https://huggingface.co/datasets/scikit-learn/california-housing/resolve/main/cal_housing.csv", dest)
        except Exception:
            import random; random.seed(7)
            rows=["MedInc,HouseAge,AveRooms,AveBedrms,Population,AveOccup"]
            for _ in range(300):
                rows.append(",".join([str(round(random.uniform(1,10),3)),str(random.randint(1,52)),
                    str(round(random.uniform(3,8),3)),str(round(random.uniform(0.8,2),3)),
                    str(random.randint(100,5000)),str(round(random.uniform(2,5),3))]))
            dest.write_text("\n".join(rows))
    with open(dest) as f: rows=list(csv.DictReader(f))
    feat=["MedInc","HouseAge","AveRooms","AveBedrms","Population","AveOccup"]
    X = np.array([[float(r[c]) for c in feat] for r in rows[:300]])
    return (X - X.mean(0)) / (X.std(0) + 1e-9)  # standardise

def demo_svd_basics(X):
    print("=== SVD Decomposition ===")
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    print(f"  X shape: {X.shape}")
    print(f"  U: {U.shape}, s: {s.shape}, Vt: {Vt.shape}")
    print(f"  Singular values: {s.round(3)}")
    print(f"  Reconstruction error: {np.linalg.norm(X - U @ np.diag(s) @ Vt):.6f}")

def demo_low_rank(X):
    print("\n=== Low-Rank Approximation ===")
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    total_energy = np.sum(s**2)
    for k in [1, 2, 3, 6]:
        X_k = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
        err = np.linalg.norm(X - X_k, "fro")
        var_explained = np.sum(s[:k]**2) / total_energy
        print(f"  k={k}: Frobenius error={err:.4f}  variance explained={var_explained:.4f}")

def demo_pca_via_svd(X):
    print("\n=== PCA via SVD (vs eigh) ===")
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    # Scores (projections) = U * s
    scores = U * s
    print(f"  PC1 variance captured: {s[0]**2 / np.sum(s**2):.4f}")
    print(f"  PC2 variance captured: {s[1]**2 / np.sum(s**2):.4f}")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(scores[:, 0], scores[:, 1], alpha=0.3, s=8, c="coral")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_title("SVD-based PCA: Cal Housing")
    fig.savefig(OUTPUT / "svd_pca.png", dpi=120, bbox_inches="tight")
    plt.close(fig); print(f"  Saved: svd_pca.png")

def demo_pseudo_inverse(X):
    print("\n=== Moore-Penrose Pseudo-Inverse ===")
    A = X[:10, :4]        # 10×4 overdetermined
    b = X[:10, 0]
    # A⁺ = V Σ⁻¹ U^T
    Ap = np.linalg.pinv(A)
    w  = Ap @ b
    print(f"  w = {w.round(4)}")
    print(f"  Aw-b ≈ 0: {np.allclose(A @ w, b, atol=1e-6)}")

def demo_condition():
    print("\n=== Condition Number via SVD ===")
    for name, A in [("Well-conditioned", np.eye(4)), ("Ill-conditioned", np.array([[1.,1.],[1.,1.+1e-9]]))]:
        _, s, _ = np.linalg.svd(A)
        cond = s[0] / (s[-1] + 1e-15)
        print(f"  {name}: σ_max/σ_min = {cond:.3e}")

if __name__ == "__main__":
    X = download()
    demo_svd_basics(X)
    demo_low_rank(X)
    demo_pca_via_svd(X)
    demo_pseudo_inverse(X)
    demo_condition()
