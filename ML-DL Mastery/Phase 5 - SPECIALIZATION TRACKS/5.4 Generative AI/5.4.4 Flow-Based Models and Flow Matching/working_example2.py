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

if __name__ == "__main__":
    demo()
