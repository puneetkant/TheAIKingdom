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
    # A⁺ = V Sigma^-1 U^T
    Ap = np.linalg.pinv(A)
    w  = Ap @ b
    print(f"  w = {w.round(4)}")
    print(f"  Aw-b ~= 0: {np.allclose(A @ w, b, atol=1e-6)}")

def demo_condition():
    print("\n=== Condition Number via SVD ===")
    for name, A in [("Well-conditioned", np.eye(4)), ("Ill-conditioned", np.array([[1.,1.],[1.,1.+1e-9]]))]:
        _, s, _ = np.linalg.svd(A)
        cond = s[0] / (s[-1] + 1e-15)
        print(f"  {name}: sigma_max/sigma_min = {cond:.3e}")

def demo_lsa(X):
    print("\n=== Latent Semantic Analysis (LSA) via SVD ===")
    # Build a tiny term-document matrix
    np.random.seed(7)
    terms = ["algebra", "matrix", "vector", "neural", "gradient", "eigen"]
    n_docs = 8
    # Synthetic term-frequency matrix (terms x docs)
    TF = np.abs(np.random.randn(len(terms), n_docs))
    U, s, Vt = np.linalg.svd(TF, full_matrices=False)
    # Top-2 concept space
    k = 2
    doc_coords  = np.diag(s[:k]) @ Vt[:k, :]   # (k, n_docs)
    term_coords = U[:, :k]                       # (n_terms, k)
    print(f"  TF matrix: {TF.shape}  ->  LSA rank-{k} approx")
    print(f"  Top-2 singular values: {s[:2].round(3)}")
    # Query: similarity to 'algebra matrix'
    query_vec = TF[:, 0]   # use first doc as query proxy
    q_proj = U[:, :k].T @ query_vec / (np.linalg.norm(query_vec) + 1e-9)
    sims = [np.dot(q_proj, doc_coords[:, d]) /
            (np.linalg.norm(doc_coords[:, d]) + 1e-9) for d in range(n_docs)]
    print(f"  LSA cosine sims to doc-0: {[round(s,3) for s in sims]}")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(doc_coords[0], doc_coords[1], color="steelblue", s=80, label="Docs")
    for i, tc in enumerate(term_coords):
        ax.annotate(terms[i], tc, fontsize=8, color="tomato")
    ax.scatter(term_coords[:, 0], term_coords[:, 1], color="tomato", s=40)
    ax.set_title("LSA: Documents in 2D Concept Space")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.savefig(OUTPUT / "lsa_svd.png", dpi=120, bbox_inches="tight")
    plt.close(fig); print("  Saved: lsa_svd.png")


def demo_image_compression(X):
    print("\n=== SVD Image Compression (synthetic) ===")
    # Use X as a 'grayscale image patch' (300x6 -> pad to square for demo)
    img = X[:50, :].copy()
    U, s, Vt = np.linalg.svd(img, full_matrices=False)
    total_e = np.sum(s**2)
    print(f"  Image shape: {img.shape}")
    for k in [1, 2, 3, 5]:
        img_k = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
        psnr = 20 * np.log10(img.max() / (np.linalg.norm(img - img_k) / img.size**0.5 + 1e-9))
        var = np.sum(s[:k]**2) / total_e
        print(f"  k={k}: var_explained={var:.3f}  PSNR~={psnr:.1f} dB")


if __name__ == "__main__":
    X = download()
    demo_svd_basics(X)
    demo_low_rank(X)
    demo_pca_via_svd(X)
    demo_pseudo_inverse(X)
    demo_condition()
    demo_lsa(X)
    demo_image_compression(X)
