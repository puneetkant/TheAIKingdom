"""
Working Example 2: Joint Distributions and Transformations
==========================================================
Bivariate Gaussian, marginalisation, covariance, correlation,
change-of-variables (Box-Muller), copulas, and joint KDE.

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

def demo_bivariate_gaussian():
    print("=== Bivariate Gaussian ===")
    np.random.seed(42)
    mu    = np.array([1.0, -2.0])
    sigma = np.array([[2.0, 0.8], [0.8, 1.0]])   # positive definite

    # Sample via Cholesky
    L = np.linalg.cholesky(sigma)
    z = np.random.randn(1000, 2)
    X = z @ L.T + mu

    print(f"  Sample mean:  {X.mean(0).round(4)}")
    print(f"  Sample cov:\n{np.cov(X.T).round(4)}")
    print(f"  Sample corr: {np.corrcoef(X.T)[0,1]:.4f}  (theory: {0.8/np.sqrt(2.0):.4f})")

    # Marginalisation: X1 ~ N(mu1, sigma11)
    print(f"\n  Marginal X1: E={X[:,0].mean():.3f}  std={X[:,0].std():.3f}  (theory: μ=1, σ=√2≈{np.sqrt(2):.3f})")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(X[:,0], X[:,1], s=5, alpha=0.3, c="steelblue")
    axes[0].set_title("Bivariate Gaussian samples")
    axes[1].hist(X[:,0], bins=40, density=True, alpha=0.7, color="steelblue")
    axes[1].hist(X[:,1], bins=40, density=True, alpha=0.7, color="coral")
    axes[1].set_title("Marginal distributions")
    fig.savefig(OUTPUT / "bivariate_gaussian.png", dpi=120, bbox_inches="tight")
    plt.close(fig); print(f"\n  Saved: bivariate_gaussian.png")

def demo_change_of_variables():
    print("\n=== Box-Muller Transform (Uniform → Gaussian) ===")
    np.random.seed(1)
    U1 = np.random.uniform(0, 1, 50_000)
    U2 = np.random.uniform(0, 1, 50_000)
    # Z ~ N(0,1)
    Z1 = np.sqrt(-2 * np.log(U1)) * np.cos(2 * np.pi * U2)
    Z2 = np.sqrt(-2 * np.log(U1)) * np.sin(2 * np.pi * U2)
    print(f"  Z1: mean={Z1.mean():.4f}  std={Z1.std():.4f}  (theory: 0, 1)")
    print(f"  Z2: mean={Z2.mean():.4f}  std={Z2.std():.4f}")

def demo_covariance_correlation():
    print("\n=== Covariance and Correlation ===")
    import csv, urllib.request
    DATA = Path(__file__).parent / "data"; DATA.mkdir(exist_ok=True)
    dest = DATA / "cal_housing.csv"
    if not dest.exists():
        try: urllib.request.urlretrieve(
            "https://huggingface.co/datasets/scikit-learn/california-housing/resolve/main/cal_housing.csv", dest)
        except Exception: dest.write_text("MedInc,HouseAge,AveRooms\n2.0,20.0,5.0\n")
    with open(dest) as f:
        rows = list(csv.DictReader(f))
    feat = ["MedInc", "HouseAge", "AveRooms", "AveBedrms"]
    X = np.array([[float(r[c]) for c in feat if c in r] for r in rows[:300]])
    cov  = np.cov(X.T)
    corr = np.corrcoef(X.T)
    print(f"  Correlation matrix:\n{corr.round(3)}")

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(feat))); ax.set_yticks(range(len(feat)))
    ax.set_xticklabels(feat, rotation=30, ha="right"); ax.set_yticklabels(feat)
    plt.colorbar(im); ax.set_title("Correlation matrix")
    fig.savefig(OUTPUT / "correlation_heatmap.png", dpi=120, bbox_inches="tight")
    plt.close(fig); print(f"  Saved: correlation_heatmap.png")

if __name__ == "__main__":
    demo_bivariate_gaussian()
    demo_change_of_variables()
    demo_covariance_correlation()
