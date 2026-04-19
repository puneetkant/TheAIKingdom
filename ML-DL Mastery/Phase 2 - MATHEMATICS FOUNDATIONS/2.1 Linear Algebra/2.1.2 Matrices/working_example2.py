"""
Working Example 2: Matrices — NumPy matrix operations for ML
=============================================================
Feature matrix operations: creation, arithmetic, transpose, multiply,
broadcasting, stacking, slicing; OLS solution; feature covariance;
softmax weight matrix demo.

Run:  python working_example2.py
"""
import urllib.request, csv
from pathlib import Path
try:
    import numpy as np
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit("pip install numpy matplotlib")

DATA   = Path(__file__).parent / "data"
OUTPUT = Path(__file__).parent / "output"
DATA.mkdir(exist_ok=True); OUTPUT.mkdir(exist_ok=True)

def download() -> np.ndarray:
    dest = DATA / "cal_housing.csv"
    if not dest.exists():
        try:
            urllib.request.urlretrieve(
                "https://huggingface.co/datasets/scikit-learn/california-housing/resolve/main/cal_housing.csv", dest)
        except Exception:
            import random; random.seed(0)
            rows = [["MedInc","HouseAge","AveRooms","AveBedrms","Population","MedHouseVal"]]
            for _ in range(100):
                rows.append([round(random.uniform(1,10),3), random.randint(1,52),
                              round(random.uniform(3,8),3), round(random.uniform(0.8,2),3),
                              random.randint(100,5000), round(random.uniform(0.5,5),3)])
            dest.write_text("\n".join(",".join(str(x) for x in r) for r in rows))
    with open(dest) as f:
        rows = list(csv.DictReader(f))
    feat = ["MedInc","HouseAge","AveRooms","AveBedrms","Population"]
    X = np.array([[float(r[c]) for c in feat] for r in rows[:200]])
    y = np.array([float(r["MedHouseVal"]) for r in rows[:200]])
    return X, y

def demo_matrix_basics(X):
    print("=== Matrix Basics ===")
    print(f"  Shape: {X.shape}  dtype: {X.dtype}")
    print(f"  X[:3]:\n{X[:3].round(3)}")
    print(f"  X.T shape: {X.T.shape}")
    print(f"  Row means: {X.mean(axis=0).round(3)}")
    print(f"  Col sums:  {X.sum(axis=1)[:5].round(3)}")

def demo_matrix_multiply(X, y):
    print("\n=== Matrix Multiply (OLS) ===")
    # Add bias column
    Xb = np.hstack([np.ones((X.shape[0], 1)), X])
    # Normal equations: w = (X^T X)^{-1} X^T y
    w = np.linalg.solve(Xb.T @ Xb, Xb.T @ y)
    y_hat = Xb @ w
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    print(f"  Weights: {w.round(4)}")
    print(f"  R² = {r2:.4f}")

def demo_covariance_matrix(X):
    print("\n=== Covariance Matrix ===")
    X_c = X - X.mean(axis=0)
    cov = (X_c.T @ X_c) / (X.shape[0] - 1)
    print(f"  Shape: {cov.shape}")
    print(f"  Diagonal (variances): {np.diag(cov).round(3)}")
    # Correlation matrix
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)
    print(f"  Correlation MedInc<->HouseAge: {corr[0,1]:.4f}")

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(im)
    feat = ["MedInc","HouseAge","AveRooms","AveBedrms","Pop"]
    ax.set_xticks(range(5)); ax.set_xticklabels(feat, rotation=45)
    ax.set_yticks(range(5)); ax.set_yticklabels(feat)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{corr[i,j]:.2f}", ha="center", va="center", fontsize=7)
    ax.set_title("Correlation Matrix")
    plt.tight_layout(); fig.savefig(OUTPUT/"corr_matrix.png", dpi=120, bbox_inches="tight")
    plt.close(fig); print(f"  Saved: corr_matrix.png")

def demo_broadcasting():
    print("\n=== Broadcasting ===")
    # Standardise each column without a loop
    X = np.random.randn(100, 4)
    mu  = X.mean(axis=0)   # shape (4,)
    std = X.std(axis=0)    # shape (4,)
    X_std = (X - mu) / std  # broadcasts automatically
    print(f"  Original mean:  {X.mean(axis=0).round(4)}")
    print(f"  Scaled mean:    {X_std.mean(axis=0).round(4)}")
    print(f"  Scaled std:     {X_std.std(axis=0).round(4)}")

def demo_softmax_weights():
    print("\n=== Softmax Weight Matrix (3-class) ===")
    n, d, k = 20, 5, 3
    X  = np.random.randn(n, d)
    W  = np.random.randn(d, k) * 0.1   # weight matrix in ℝ^{dxk}
    b  = np.zeros(k)
    Z  = X @ W + b                       # logits in ℝ^{nxk}
    eZ = np.exp(Z - Z.max(axis=1, keepdims=True))  # numerically stable
    P  = eZ / eZ.sum(axis=1, keepdims=True)         # softmax probabilities
    print(f"  W shape: {W.shape},  Z shape: {Z.shape},  P shape: {P.shape}")
    print(f"  Row prob sums (should be 1): {P.sum(axis=1)[:4].round(6)}")

if __name__ == "__main__":
    X, y = download()
    demo_matrix_basics(X)
    demo_matrix_multiply(X, y)
    demo_covariance_matrix(X)
    demo_broadcasting()
    demo_softmax_weights()
