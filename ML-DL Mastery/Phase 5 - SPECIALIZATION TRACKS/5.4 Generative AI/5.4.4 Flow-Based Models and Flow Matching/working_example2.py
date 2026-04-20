"""
Working Example 2: Normalizing Flows — RealNVP coupling layer (1-D & 2-D)
==========================================================================
Implements affine coupling transformations and log-determinant computation.

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

def affine_coupling_forward(x, s, t):
    """Affine coupling: z = x * exp(s) + t.  Log-det = sum(s)."""
    z = x * np.exp(s) + t
    log_det = np.sum(s, axis=-1)
    return z, log_det

def affine_coupling_inverse(z, s, t):
    """Inverse: x = (z - t) * exp(-s)."""
    return (z - t) * np.exp(-s)

def demo():
    np.random.seed(42)
    print("=== Normalizing Flows: Affine Coupling ===")
    # 2-D moon-like data
    n = 300
    theta = np.linspace(0, np.pi, n)
    x = np.stack([np.cos(theta), np.sin(theta) + np.random.randn(n)*0.1], axis=1)

    # Simple scale/translate learned from data (MLE proxy)
    s = np.array([0.5, -0.3])    # per-dim scale log
    t = np.array([0.2, 0.5])     # per-dim translation

    z, ld = affine_coupling_forward(x, s, t)
    x_rec = affine_coupling_inverse(z, s, t)

    print(f"  Max reconstruction error: {np.abs(x - x_rec).max():.2e}")
    print(f"  Mean log-det: {ld.mean():.4f}")
    print(f"  Latent mean: {z.mean(axis=0).round(3)}")

    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    axes[0].scatter(x[:, 0], x[:, 1], s=5, alpha=0.5); axes[0].set_title("Data space x")
    axes[1].scatter(z[:, 0], z[:, 1], s=5, alpha=0.5, color="orange"); axes[1].set_title("Latent space z")
    plt.tight_layout(); plt.savefig(OUTPUT / "flow_coupling.png"); plt.close()
    print("  Saved flow_coupling.png")

def demo_flow_matching():
    """Flow matching: interpolate between source and target distributions."""
    print("\n=== Flow Matching (Linear Interpolation) ===")
    rng = np.random.default_rng(0)
    n = 200
    # Source: standard Gaussian
    x0 = rng.standard_normal((n, 2))
    # Target: two-moon-like structure
    theta = np.linspace(0, np.pi, n)
    x1 = np.stack([np.cos(theta), np.sin(theta) + rng.normal(0, 0.1, n)], axis=1)

    # Conditional flow matching: x_t = (1-t)*x0 + t*x1  (linear interpolant)
    for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
        xt = (1-t)*x0 + t*x1
        print(f"  t={t:.2f}: mean={xt.mean(axis=0).round(3)}  std={xt.std(axis=0).round(3)}")

    # Vector field u_t(x) = x1 - x0 (constant for linear flow)
    u = x1 - x0
    print(f"  Velocity field norm (mean): {np.linalg.norm(u, axis=1).mean():.4f}")


def demo_normalizing_flow_nll():
    """Compute negative log-likelihood for the affine coupling model."""
    print("\n=== NLL Estimation under Affine Flow ===")
    np.random.seed(5)
    n = 500
    theta = np.linspace(0, np.pi, n)
    x = np.stack([np.cos(theta), np.sin(theta) + np.random.randn(n)*0.05], axis=1)
    s = np.array([0.3, -0.2]); t = np.array([0.1, 0.4])
    z, log_det = affine_coupling_forward(x, s, t)
    # Under a standard Gaussian prior: log p(x) = log p(z) + log|det J|
    log_pz  = -0.5 * (z**2 + np.log(2*np.pi)).sum(axis=1)
    log_px  = log_pz + log_det
    nll     = -log_px.mean()
    print(f"  Mean log p(z):    {log_pz.mean():.4f}")
    print(f"  Mean log|det J|:  {log_det.mean():.4f}")
    print(f"  Mean NLL:         {nll:.4f}")


if __name__ == "__main__":
    demo()
    demo_flow_matching()
    demo_normalizing_flow_nll()
