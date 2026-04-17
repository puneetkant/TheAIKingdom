"""
Working Example 2: 3D Vision — point cloud basics and 3D bounding box concepts
===============================================================================
Demonstrates 3D coordinate systems, projection, and point cloud operations.

Run:  python working_example2.py
"""
from pathlib import Path
try:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
except ImportError:
    raise SystemExit("pip install numpy matplotlib")

OUTPUT = Path(__file__).parent / "output"
OUTPUT.mkdir(exist_ok=True)

def make_point_cloud(n=200, seed=42):
    """Synthetic point cloud: sphere + plane."""
    rng = np.random.default_rng(seed)
    # Sphere
    theta = rng.uniform(0, 2*np.pi, n//2); phi = rng.uniform(0, np.pi, n//2)
    r = rng.uniform(0.9, 1.1, n//2)
    xs = r * np.sin(phi) * np.cos(theta)
    ys = r * np.sin(phi) * np.sin(theta)
    zs = r * np.cos(phi)
    sphere = np.stack([xs, ys, zs], axis=1)
    # Plane
    plane = rng.uniform(-2, 2, (n//2, 3)); plane[:, 2] = -1.5 + rng.normal(0, 0.05, n//2)
    return np.vstack([sphere, plane])

def project_perspective(pts, f=1.0):
    """Perspective projection to 2D."""
    return pts[:, :2] / (pts[:, 2:3] + f)

def demo():
    print("=== 3D Vision: Point Cloud Operations ===")
    pts = make_point_cloud()
    print(f"  Point cloud shape: {pts.shape}")
    print(f"  Centroid: {pts.mean(axis=0).round(3)}")
    print(f"  Bounds: X={pts[:,0].min():.2f}..{pts[:,0].max():.2f}, Z={pts[:,2].min():.2f}..{pts[:,2].max():.2f}")

    # Simple normal estimation via PCA of local neighbourhood
    cov = np.cov(pts.T)
    vals, vecs = np.linalg.eigh(cov)
    normal = vecs[:, 0]  # eigenvector with smallest eigenvalue = dominant normal
    print(f"  Dominant plane normal: {normal.round(3)}")

    # Perspective projection
    proj2d = project_perspective(pts + np.array([0, 0, 4]))  # shift forward

    fig = plt.figure(figsize=(12, 4))
    ax1 = fig.add_subplot(131, projection="3d")
    ax1.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=2, alpha=0.3)
    ax1.set_title("3D Point Cloud")
    ax2 = fig.add_subplot(132)
    ax2.scatter(proj2d[:, 0], proj2d[:, 1], s=2, alpha=0.3)
    ax2.set_title("Perspective Projection")
    ax3 = fig.add_subplot(133)
    ax3.hist(pts[:, 2], bins=20, orientation="horizontal")
    ax3.set_title("Z-depth Histogram")
    plt.tight_layout(); plt.savefig(OUTPUT / "3d_vision.png"); plt.close()
    print("  Saved 3d_vision.png")

if __name__ == "__main__":
    demo()
