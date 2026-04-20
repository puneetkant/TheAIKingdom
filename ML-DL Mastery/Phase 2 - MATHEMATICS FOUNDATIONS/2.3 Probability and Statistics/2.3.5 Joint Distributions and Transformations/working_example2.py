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
    import matplotlib
    matplotlib.use("Agg")
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
    print(f"\n  Marginal X1: E={X[:,0].mean():.3f}  std={X[:,0].std():.3f}  (theory: mu=1, sigma=sqrt(2)~={np.sqrt(2):.3f})")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(X[:,0], X[:,1], s=5, alpha=0.3, c="steelblue")
    axes[0].set_title("Bivariate Gaussian samples")
    axes[1].hist(X[:,0], bins=40, density=True, alpha=0.7, color="steelblue")
    axes[1].hist(X[:,1], bins=40, density=True, alpha=0.7, color="coral")
    axes[1].set_title("Marginal distributions")
    fig.savefig(OUTPUT / "bivariate_gaussian.png", dpi=120, bbox_inches="tight")
    plt.close(fig); print(f"\n  Saved: bivariate_gaussian.png")

def demo_change_of_variables():
    print("\n=== Box-Muller Transform (Uniform -> Gaussian) ===")
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

def demo_joint_2d_gaussian():
    print("\n=== Joint 2D Gaussian (rho=0.7) ===")
    np.random.seed(50)
    rho = 0.7
    cov = np.array([[1.0, rho], [rho, 1.0]])
    L = np.linalg.cholesky(cov)
    z = np.random.randn(2000, 2)
    samples = z @ L.T
    print(f"  Sample correlation: {np.corrcoef(samples.T)[0, 1]:.4f}  (target: {rho})")

    fig = plt.figure(figsize=(8, 8))
    gs = fig.add_gridspec(2, 2, hspace=0.05, wspace=0.05)
    ax_main  = fig.add_subplot(gs[1, 0])
    ax_top   = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)
    ax_corner = fig.add_subplot(gs[0, 1])
    ax_corner.set_visible(False)

    ax_main.scatter(samples[:, 0], samples[:, 1], s=4, alpha=0.3, color="steelblue")
    ax_main.set_xlabel("X1")
    ax_main.set_ylabel("X2")
    ax_top.hist(samples[:, 0], bins=50, density=True, color="steelblue", alpha=0.7)
    ax_top.set_title(f"Joint 2D Gaussian, rho={rho}")
    plt.setp(ax_top.get_xticklabels(), visible=False)
    ax_right.hist(samples[:, 1], bins=50, density=True, color="coral", alpha=0.7,
                  orientation="horizontal")
    plt.setp(ax_right.get_yticklabels(), visible=False)

    fig.savefig(OUTPUT / "joint_bivariate.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: joint_bivariate.png")


def demo_sum_of_uniforms():
    print("\n=== Sum of Uniforms: Z = X + Y, X,Y ~ Uniform(0,1) ===")
    np.random.seed(60)
    n_samples = 10_000
    X = np.random.uniform(0, 1, n_samples)
    Y = np.random.uniform(0, 1, n_samples)
    Z = X + Y
    # Triangular(0,2) PDF
    z_range = np.linspace(0, 2, 300)
    tri_pdf = np.where(z_range <= 1, z_range, 2.0 - z_range)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(Z, bins=60, density=True, alpha=0.7, color="steelblue", label="Empirical Z=X+Y")
    ax.plot(z_range, tri_pdf, "r-", lw=2, label="Triangular PDF")
    ax.set_xlabel("Z = X + Y")
    ax.set_ylabel("Density")
    ax.set_title("Sum of two Uniform(0,1) ~ Triangular(0,2)")
    ax.legend()
    fig.savefig(OUTPUT / "sum_of_uniforms.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    theory_std = 1.0 / np.sqrt(6)
    print(f"  Z mean={Z.mean():.4f}  std={Z.std():.4f}  (theory: 1.0, {theory_std:.4f})")
    print("  Saved: sum_of_uniforms.png")


def demo_order_statistics():
    print("\n=== Order Statistics: Min, Median, Max of Uniform(0,1) ===")
    np.random.seed(70)
    n = 10
    n_trials = 8000
    draws = np.random.uniform(0, 1, size=(n_trials, n))
    mins    = draws.min(axis=1)
    medians = np.median(draws, axis=1)
    maxs    = draws.max(axis=1)

    labels = [f"Min (X_(1))", f"Median (X_({n//2}))", f"Max (X_({n}))"]
    colors = ["steelblue", "coral", "seagreen"]
    stats  = [mins, medians, maxs]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, stat, label, color in zip(axes, stats, labels, colors):
        ax.hist(stat, bins=50, density=True, alpha=0.75, color=color)
        ax.set_title(label)
        ax.set_xlabel("Value")
    axes[0].set_ylabel("Density")
    plt.suptitle(f"Order Statistics from Uniform(0,1), n={n}, trials={n_trials}")
    plt.tight_layout()
    fig.savefig(OUTPUT / "order_stats.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Min:    mean={mins.mean():.4f}   (theory: 1/{n+1} = {1/(n+1):.4f})")
    print(f"  Median: mean={medians.mean():.4f}  (theory: 0.5)")
    print(f"  Max:    mean={maxs.mean():.4f}   (theory: {n}/{n+1} = {n/(n+1):.4f})")
    print("  Saved: order_stats.png")


if __name__ == "__main__":
    demo_bivariate_gaussian()
    demo_change_of_variables()
    demo_covariance_correlation()
    demo_joint_2d_gaussian()
    demo_sum_of_uniforms()
    demo_order_statistics()
