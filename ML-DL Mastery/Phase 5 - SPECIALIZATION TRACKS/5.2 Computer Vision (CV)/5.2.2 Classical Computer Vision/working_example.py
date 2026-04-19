"""
Working Example: Classical Computer Vision
Covers edge detection, corner detection, SIFT concepts, feature matching,
and image segmentation — implemented from scratch with numpy.
"""
import numpy as np
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_classical_cv")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def make_test_image(H=64, W=64):
    img = np.zeros((H, W), dtype=np.float64)
    # Horizontal and vertical edges
    img[15:45, :] = 0.5
    img[20:40, 20:44] = 1.0
    # Add gradient
    img += np.linspace(0, 0.3, W)
    return np.clip(img, 0, 1)


def convolve2d(img, kernel):
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    H, W   = img.shape
    padded = np.pad(img, ((ph, ph), (pw, pw)), mode="reflect")
    out    = np.zeros_like(img)
    for i in range(H):
        for j in range(W):
            out[i, j] = (padded[i:i+kh, j:j+kw] * kernel).sum()
    return out


# -- 1. Edge Detection ---------------------------------------------------------
def edge_detection():
    print("=== Edge Detection ===")
    img = make_test_image()

    # Sobel kernels
    Kx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=float)
    Ky = Kx.T

    Gx = convolve2d(img, Kx)
    Gy = convolve2d(img, Ky)
    G  = np.sqrt(Gx**2 + Gy**2)
    G  = G / G.max()

    # Angle
    angle = np.degrees(np.arctan2(Gy, Gx))

    print(f"  Image: {img.shape}  range [{img.min():.2f}, {img.max():.2f}]")
    print(f"  Gradient mag: max={G.max():.4f}  mean={G.mean():.4f}")
    print(f"  Strong edges (>0.5): {(G > 0.5).sum()} pixels")

    # Canny-like: non-max suppression + thresholding
    def simple_canny(G, low=0.15, high=0.35):
        strong = G > high
        weak   = (G > low) & ~strong
        # Connect weak edges to strong (simplified)
        from scipy.ndimage import label
        labeled, n = label(strong | weak)
        for region_id in range(1, n+1):
            mask = labeled == region_id
            if (mask & strong).any():
                strong = strong | mask
        return strong.astype(float)

    try:
        edges = simple_canny(G)
        print(f"  Canny edges: {int(edges.sum())} edge pixels")
    except Exception:
        edges = (G > 0.3).astype(float)

    fig, axes = plt.subplots(1, 4, figsize=(14, 3))
    axes[0].imshow(img, cmap="gray"); axes[0].set_title("Original")
    axes[1].imshow(Gx, cmap="RdBu"); axes[1].set_title("Gx (Sobel-X)")
    axes[2].imshow(G,  cmap="gray"); axes[2].set_title("Gradient Magnitude")
    axes[3].imshow(edges, cmap="gray"); axes[3].set_title("Edge Map")
    for ax in axes: ax.axis("off")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "edge_detection.png")
    plt.savefig(path, dpi=90); plt.close()
    print(f"  Plot: {path}")


# -- 2. Corner Detection (Harris) ---------------------------------------------
def corner_detection():
    print("\n=== Harris Corner Detection ===")
    print("  Principle: corners have large gradients in all directions")
    print("  M = [Sigma Ix², Sigma IxIy ; Sigma IxIy, Sigma Iy²]  (structure tensor)")
    print("  R = det(M) - k·trace(M)²")
    print("     R >> 0 -> corner")
    print("     R < 0  -> edge")
    print("     R ~= 0  -> flat")
    print()

    img = make_test_image()
    Kx  = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=float)
    Ix  = convolve2d(img, Kx)
    Iy  = convolve2d(img, Kx.T)

    w = np.ones((3,3)) / 9   # uniform window (Gaussian in practice)
    Ixx = convolve2d(Ix**2, w)
    Ixy = convolve2d(Ix*Iy, w)
    Iyy = convolve2d(Iy**2, w)

    k = 0.04
    det   = Ixx * Iyy - Ixy**2
    trace = Ixx + Iyy
    R     = det - k * trace**2

    threshold  = 0.001 * R.max()
    corners    = (R > threshold)
    n_corners  = corners.sum()

    print(f"  R: min={R.min():.4f}  max={R.max():.4f}")
    print(f"  Corners detected (R>{threshold:.5f}): {n_corners}")

    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    axes[0].imshow(img, cmap="gray"); axes[0].set_title("Original")
    axes[1].imshow(R, cmap="RdBu"); axes[1].set_title("Harris R")
    axes[2].imshow(img, cmap="gray")
    ys, xs = np.where(corners)
    axes[2].scatter(xs, ys, c="red", s=8); axes[2].set_title("Corners")
    for ax in axes: ax.axis("off")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "harris_corners.png")
    plt.savefig(path, dpi=90); plt.close()
    print(f"  Plot: {path}")


# -- 3. SIFT overview ----------------------------------------------------------
def sift_overview():
    print("\n=== SIFT (Scale-Invariant Feature Transform) ===")
    print("  Authors: Lowe (2004)  — patented, now open (2020)")
    print()
    print("  Pipeline:")
    steps = [
        ("1. Scale-space extrema",  "Build DoG (Difference of Gaussians) pyramid; "
                                    "find local min/max across scales"),
        ("2. Keypoint localisation", "Reject low-contrast and edge-response points"),
        ("3. Orientation assign",    "Compute dominant gradient direction; "
                                     "ensures rotation invariance"),
        ("4. Descriptor (128-D)",    "4×4 spatial grid × 8 bins histogram of gradients"),
    ]
    for name, desc in steps:
        print(f"  {name}:")
        print(f"    {desc}")
    print()
    print("  Properties: scale-invariant, rotation-invariant, partially illumination-invariant")
    print("  Applications: image stitching, object recognition, SLAM")
    print()
    print("  Python usage:")
    print("    import cv2")
    print("    sift = cv2.SIFT_create()")
    print("    kp, des = sift.detectAndCompute(gray_img, None)  # des: (N, 128)")
    print()
    print("  Related: SURF (faster), ORB (free, binary), BRISK, AKAZE")


# -- 4. Histogram of Oriented Gradients (HoG) ---------------------------------
def hog_demo():
    print("\n=== HOG (Histogram of Oriented Gradients) ===")
    img = make_test_image(32, 32)

    Kx = np.array([[-1,0,1]], dtype=float)
    Ky = Kx.T
    Gx = convolve2d(img, Kx)
    Gy = convolve2d(img, Ky)
    G  = np.sqrt(Gx**2 + Gy**2)
    theta = np.degrees(np.arctan2(Gy, Gx)) % 180  # unsigned (0-180)

    # Build HOG for one cell (8×8)
    n_bins = 9
    bin_w  = 180 / n_bins
    cell   = G[:8, :8]
    ang    = theta[:8, :8]
    hist   = np.zeros(n_bins)
    for i in range(8):
        for j in range(8):
            b1 = int(ang[i,j] / bin_w) % n_bins
            b2 = (b1 + 1) % n_bins
            hist[b1] += cell[i,j]
            hist[b2] += 0

    print(f"  Cell HOG histogram (9 bins, 0-180°):")
    for bi in range(n_bins):
        bar = "#" * int(hist[bi] / (hist.max()+1e-9) * 12)
        print(f"    Bin {bi*20:>3}-{(bi+1)*20:>3}°: {bar} ({hist[bi]:.3f})")

    H, W = img.shape
    cells_h, cells_w = H // 8, W // 8
    print(f"\n  Full image {H}×{W}: {cells_h}×{cells_w} cells × 9 bins = "
          f"{cells_h * cells_w * 9} features")
    print(f"  With 2x2 block normalisation -> {(cells_h-1)*(cells_w-1)*4*9} features")
    print()
    print("  HOG for pedestrian detection (Dalal & Triggs 2005):")
    print("    64x128 window -> 3780-dim HOG -> Linear SVM -> state-of-art in 2005")


# -- 5. Image segmentation -----------------------------------------------------
def image_segmentation():
    print("\n=== Image Segmentation ===")
    print("  Semantic:  assign a class label to every pixel")
    print("  Instance:  detect and segment each individual object")
    print("  Panoptic:  semantic + instance combined")
    print()
    print("  Classical methods:")

    # K-means segmentation
    rng = np.random.default_rng(7)
    img = make_test_image(32, 32)
    pixels = img.flatten().reshape(-1, 1)
    K = 3
    centroids = np.array([[0.2], [0.6], [0.95]])

    for _ in range(20):
        dists  = np.abs(pixels - centroids.T)  # (N, K)
        labels = dists.argmin(1)
        for k in range(K):
            if (labels == k).any():
                centroids[k] = pixels[labels == k].mean(0)

    seg = labels.reshape(img.shape)
    print(f"  K-means segmentation (K=3):")
    for k in range(K):
        print(f"    Cluster {k}: {(labels==k).sum()} pixels  centroid={centroids[k,0]:.3f}")

    print()
    print("  Deep learning methods:")
    dl = [
        ("FCN",         "Fully Convolutional Network — first end-to-end semantic seg"),
        ("U-Net",       "Encoder-decoder with skip connections; medical imaging"),
        ("DeepLab",     "Atrous conv + ASPP; state-of-art on PASCAL VOC"),
        ("Mask R-CNN",  "Faster R-CNN + mask branch; instance segmentation"),
        ("SAM",         "Segment Anything Model (Meta) — zero-shot, any prompt"),
        ("SAM2",        "Video SAM; tracks objects across frames"),
    ]
    for m, d in dl:
        print(f"    {m:<14} {d}")


if __name__ == "__main__":
    edge_detection()
    corner_detection()
    sift_overview()
    hog_demo()
    image_segmentation()
