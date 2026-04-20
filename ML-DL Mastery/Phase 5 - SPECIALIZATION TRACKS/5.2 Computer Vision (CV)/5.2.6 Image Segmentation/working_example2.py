"""
Working Example 2: Image Segmentation — pixel-level clustering with K-means
============================================================================
Semantic segmentation proxy using K-means colour clustering.

Run:  python working_example2.py
"""
from pathlib import Path
try:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans
    from sklearn.datasets import load_digits
except ImportError:
    raise SystemExit("pip install scikit-learn numpy matplotlib")

OUTPUT = Path(__file__).parent / "output"
OUTPUT.mkdir(exist_ok=True)

def make_synthetic_scene(H=32, W=64, seed=42):
    """3-region synthetic image: background, object1, object2."""
    rng = np.random.default_rng(seed)
    img = np.zeros((H, W, 3), dtype=np.float32)
    # Background (blue-ish)
    img[:, :, 2] = 0.6 + rng.random((H, W)) * 0.1
    # Object 1 (red square)
    img[5:20, 5:25, 0] = 0.9 + rng.random((15, 20)) * 0.1
    img[5:20, 5:25, 2] = 0.1
    # Object 2 (green rect)
    img[10:28, 35:55, 1] = 0.8 + rng.random((18, 20)) * 0.1
    img[10:28, 35:55, 2] = 0.2
    return img

def segment_kmeans(img, n_clusters=3, seed=42):
    pixels = img.reshape(-1, 3)
    km = KMeans(n_clusters=n_clusters, n_init=5, random_state=seed)
    labels = km.fit_predict(pixels)
    return labels.reshape(img.shape[:2])

def demo():
    print("=== Image Segmentation via K-means Clustering ===")
    img = make_synthetic_scene()
    for k in [2, 3, 4]:
        seg = segment_kmeans(img, n_clusters=k)
        print(f"  k={k}: unique segments = {len(np.unique(seg))}")

    seg = segment_kmeans(img, n_clusters=3)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img); axes[0].set_title("Original Scene")
    axes[1].imshow(seg, cmap="tab10"); axes[1].set_title("K-means Segmentation (k=3)")
    # Per-segment masks
    for i in range(3):
        axes[2].contour(seg == i, levels=[0.5], colors=[f"C{i}"])
    axes[2].imshow(img); axes[2].set_title("Segment Boundaries")
    for ax in axes: ax.axis("off")
    plt.tight_layout(); plt.savefig(OUTPUT / "segmentation.png"); plt.close()
    print("  Saved segmentation.png")

def demo_iou_metric():
    """Compute Intersection-over-Union between predicted and ground truth masks."""
    print("\n=== Intersection-over-Union (IoU) ===")
    img = make_synthetic_scene()
    seg = segment_kmeans(img, n_clusters=3)
    H, W = seg.shape
    gt = np.zeros_like(seg); gt[:H//2, :W//2] = 1
    pred_mask = (seg == seg[H//4, W//4]).astype(int)
    intersection = (gt & pred_mask).sum()
    union = (gt | pred_mask).sum()
    iou = intersection / (union + 1e-8)
    print(f"  GT mask pixels:   {gt.sum()}")
    print(f"  Pred mask pixels: {pred_mask.sum()}")
    print(f"  IoU: {iou:.4f}")


def demo_segmentation_types():
    """Describe semantic, instance, and panoptic segmentation."""
    print("\n=== Segmentation Task Types ===")
    tasks = [
        ("Semantic",  "Every pixel gets a class label", "DeepLab, FCN"),
        ("Instance",  "Each object instance labelled separately", "Mask R-CNN"),
        ("Panoptic",  "Semantic + instance unified", "Panoptic FPN"),
        ("K-means",   "Unsupervised colour-based cluster", "This demo"),
    ]
    print(f"  {'Type':12s}  {'Description':45s}  {'Example'}")
    for t, d, e in tasks:
        print(f"  {t:12s}  {d:45s}  {e}")


if __name__ == "__main__":
    demo()
    demo_iou_metric()
    demo_segmentation_types()
