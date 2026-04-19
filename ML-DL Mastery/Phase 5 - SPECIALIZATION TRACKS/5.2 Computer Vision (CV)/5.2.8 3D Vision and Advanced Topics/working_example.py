"""
Working Example: 3D Vision and Advanced Topics
Covers depth estimation, stereo vision, point clouds,
optical flow, NeRF, and 3D object detection.
"""
import numpy as np
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_3d_vision")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -- 1. Camera models and geometry ---------------------------------------------
def camera_geometry():
    print("=== Camera Geometry ===")
    print("  Pinhole camera model:")
    print("    [u]   [f  0  cx] [X]")
    print("    [v] = [0  f  cy] [Y] / Z")
    print("    [1]   [0  0   1] [Z]")
    print()

    # Intrinsic matrix example
    fx, fy = 500, 500
    cx, cy = 320, 240
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0,  0,  1]], dtype=float)
    print(f"  Intrinsic matrix K:")
    for row in K:
        print(f"    [{row[0]:.0f}  {row[1]:.0f}  {row[2]:.0f}]")

    # Project a 3D point
    P_3d = np.array([1.0, 0.5, 5.0])   # (X, Y, Z) in camera coords
    p_img = K @ P_3d
    p_img /= p_img[2]
    print(f"\n  3D point: {P_3d}")
    print(f"  Projected 2D: ({p_img[0]:.1f}, {p_img[1]:.1f}) px")

    # Unproject (assuming depth known)
    Z = 5.0
    u, v = p_img[:2]
    X = (u - cx) / fx * Z
    Y = (v - cy) / fy * Z
    print(f"  Back-projected: ({X:.3f}, {Y:.3f}, {Z:.3f})  (should match original)")


# -- 2. Stereo vision and depth ------------------------------------------------
def stereo_vision():
    print("\n=== Stereo Vision ===")
    print("  Two cameras separated by baseline b; same horizontal row (epipolar)")
    print("  Disparity d = x_left - x_right")
    print("  Depth Z = f × b / d")
    print()

    f = 500  # pixels
    b = 0.12  # meters
    print(f"  Camera: f={f}px  baseline={b}m")
    print()
    print(f"  {'Distance (Z)':>15}  {'Disparity (d)':>15}  {'px/m':>10}")
    for Z in [0.5, 1.0, 2.0, 5.0, 10.0]:
        d = f * b / Z
        print(f"    {Z:>10.1f}m   {d:>12.2f}px   {d/b:>8.1f}")

    print()
    print("  Stereo matching pipeline:")
    steps = [
        "1. Rectify images (align epipolar lines to horizontal)",
        "2. Compute cost volume: |I_L(x,y) - I_R(x-d,y)| for each d",
        "3. Aggregate costs (box filter or learning)",
        "4. WTA (winner-take-all) per pixel -> disparity map",
        "5. Post-process: sub-pixel refinement, hole filling",
    ]
    for s in steps:
        print(f"    {s}")

    print()
    print("  Deep stereo networks:")
    nets = [
        ("DispNet",    "End-to-end with correlation layer; 2016"),
        ("PSMNet",     "Pyramid stereo matching; stacked hourglass; 2018"),
        ("RAFT-Stereo","Iterative field updates; sota on KITTI; 2021"),
        ("CREStereo",  "Cascade recurrent network; 2022"),
    ]
    for m, d in nets:
        print(f"    {m:<14} {d}")


# -- 3. Point clouds ----------------------------------------------------------
def point_clouds():
    print("\n=== Point Clouds ===")
    print("  Representation: set of (x, y, z) [+ r, g, b, intensity, normal, ...]")
    print("  Sensors: LiDAR (ToF), structured light, RGBD camera, MVS reconstruction")
    print()

    # Simulate a small point cloud (sphere)
    rng = np.random.default_rng(0)
    theta = rng.uniform(0, np.pi, 200)
    phi   = rng.uniform(0, 2*np.pi, 200)
    r     = rng.uniform(0.9, 1.1, 200)
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    pc = np.stack([x, y, z], axis=1)

    print(f"  Point cloud: {pc.shape[0]} points")
    print(f"  Bounding box: "
          f"x=[{x.min():.2f},{x.max():.2f}] "
          f"y=[{y.min():.2f},{y.max():.2f}] "
          f"z=[{z.min():.2f},{z.max():.2f}]")
    print(f"  Centroid: {pc.mean(0).round(4)}")

    # Simple nearest-neighbour computation
    dists = np.linalg.norm(pc[0] - pc[1:10], axis=1)
    print(f"  Distances from point 0 to points 1-9: {np.round(dists, 3)}")

    print()
    print("  Deep learning on point clouds:")
    models = [
        ("PointNet",    "Per-point MLP + global max-pool; permutation invariant; 2017"),
        ("PointNet++",  "Hierarchical local grouping; robust to density variations"),
        ("VoxelNet",    "Voxelise + sparse 3D CNN; LiDAR detection"),
        ("PointPillar", "Pillars (vertical voxels) + 2D CNN; fast LiDAR detection"),
        ("Point Transformer","Self-attention on point sets; sota classification"),
        ("Mamba3D",     "State-space model on ordered points; 2024"),
    ]
    for m, d in models:
        print(f"    {m:<18} {d}")


# -- 4. Optical flow -----------------------------------------------------------
def optical_flow():
    print("\n=== Optical Flow ===")
    print("  Estimate per-pixel motion between two frames")
    print("  Brightness constancy: I(x, y, t) = I(x+u, y+v, t+1)")
    print()
    print("  Lucas-Kanade (local, parametric):")
    print("    [Ix·Ix  Ix·Iy][u]   [-Ix·It]")
    print("    [Iy·Ix  Iy·Iy][v] = [-Iy·It]")
    print("    Solved in small patch window (e.g. 5×5)")
    print()
    print("  Horn-Schunck (global, variational):")
    print("    Adds spatial smoothness constraint")
    print("    Minimise: integralintegral (Ix·u + Iy·v + It)² + lambda(|∇u|² + |∇v|²) dx dy")
    print()

    # Simulate optical flow on synthetic motion
    H, W = 8, 8
    # Frame 1: object at (3,3)-(5,5)
    I1 = np.zeros((H, W))
    I1[3:6, 3:6] = 1.0
    # Frame 2: object shifted by (1,1)
    I2 = np.zeros((H, W))
    I2[4:7, 4:7] = 1.0

    It = I2 - I1
    # Gradient via finite differences
    Ix = np.gradient(I1, axis=1)
    Iy = np.gradient(I1, axis=0)
    gt_u, gt_v = 1.0, 1.0

    # Compute AT·A for whole image (LK global version)
    A = np.column_stack([Ix.flatten(), Iy.flatten()])
    b = -It.flatten()
    AtA = A.T @ A + 0.01 * np.eye(2)   # ridge for stability
    Atb = A.T @ b
    uv  = np.linalg.solve(AtA, Atb)

    print(f"  Synthetic test: object moves (+1, +1)")
    print(f"  Estimated flow: u={uv[0]:.3f}  v={uv[1]:.3f}  (expected: 1.0, 1.0)")
    print()
    print("  Deep optical flow models:")
    models = [
        ("FlowNet",   "CNN for end-to-end flow; 2015"),
        ("FlowNet2",  "Stacked architecture; improved accuracy; 2017"),
        ("PWC-Net",   "Pyramidal, warping, cost volume; lightweight; 2018"),
        ("RAFT",      "Recurrent All-Pairs Field Transforms; sota; 2020"),
        ("FlowFormer","Transformer-based; sota on Sintel; 2022"),
    ]
    for m, d in models:
        print(f"    {m:<12} {d}")


# -- 5. NeRF (Neural Radiance Fields) -----------------------------------------
def nerf_overview():
    print("\n=== NeRF (Neural Radiance Fields) ===")
    print("  Mildenhall et al. (2020)")
    print("  Represent a 3D scene as: f(x, y, z, theta, phi) -> (r, g, b, sigma)")
    print("    (x,y,z) = 3D location  (theta,phi) = viewing direction")
    print("    Output: colour (rgb) + volume density (sigma)")
    print()
    print("  Rendering via volume rendering integral:")
    print("    C(r) = integral T(t) sigma(r(t)) c(r(t), d) dt")
    print("    T(t) = exp(-integral sigma(r(s)) ds)   (transmittance)")
    print()
    print("  Training:")
    print("    1. Cast rays from known camera positions")
    print("    2. Sample points along each ray")
    print("    3. Query MLP for (rgb, sigma) at each point")
    print("    4. Composite colours via volume rendering")
    print("    5. Minimise ||C_pred - C_gt||²")
    print()
    print("  NeRF variants:")
    variants = [
        ("Instant-NGP",  "Hash-grid encoding; 5min training on single GPU"),
        ("Mip-NeRF",     "Anti-aliasing via conical frustums"),
        ("NeRF-W",       "In-the-wild; appearance embeddings"),
        ("3D Gaussian",  "Splatting; explicit Gaussians; real-time render; 2023"),
        ("Zip-NeRF",     "Mip-NeRF + hash grid; sota quality + speed"),
    ]
    for v, d in variants:
        print(f"    {v:<14} {d}")


# -- 6. Monocular depth estimation ---------------------------------------------
def monocular_depth():
    print("\n=== Monocular Depth Estimation ===")
    print("  Predict per-pixel depth from a single RGB image")
    print("  Inherently ill-posed -> needs learned priors")
    print()
    print("  Loss functions:")
    losses = [
        ("Scale-Invariant Log",    "Delta(log d_i - log d_i) — handles scale ambiguity"),
        ("AbsRel",                 "|d-d| / d — relative absolute error"),
        ("delta1, delta2, delta3 accuracy",    "Fraction with max(d/d, d/d) < 1.25^k"),
        ("Berhu (reverse Huber)",  "L1 for small errors, L2 for large"),
    ]
    for l, d in losses:
        print(f"    {l:<26} {d}")
    print()
    print("  Models:")
    models = [
        ("MiDaS",          "Multi-dataset training; scale/shift invariant"),
        ("DPT",            "Vision Transformer encoder; dense prediction"),
        ("Depth Anything", "Foundation model; 62M images; V1+V2; zero-shot"),
        ("Marigold",       "Diffusion model for depth; affine-invariant"),
        ("UniDepth",       "Universal monocular depth; scale-aware + relative"),
    ]
    for m, d in models:
        print(f"    {m:<16} {d}")


if __name__ == "__main__":
    camera_geometry()
    stereo_vision()
    point_clouds()
    optical_flow()
    nerf_overview()
    monocular_depth()
