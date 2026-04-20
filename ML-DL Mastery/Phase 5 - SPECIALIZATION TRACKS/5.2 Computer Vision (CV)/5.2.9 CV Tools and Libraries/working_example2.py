"""
Working Example 2: CV Tools and Libraries — torchvision, PIL, sklearn, OpenCV overview
========================================================================================
Demonstrates image loading/processing capabilities across libraries.

Run:  python working_example2.py
"""
from pathlib import Path
import importlib

try:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.datasets import load_digits
except ImportError:
    raise SystemExit("pip install numpy matplotlib scikit-learn")

OUTPUT = Path(__file__).parent / "output"
OUTPUT.mkdir(exist_ok=True)

def check_libs():
    libs = {
        "cv2": "OpenCV — classical CV, image I/O",
        "PIL": "Pillow — image manipulation",
        "torchvision": "torchvision — datasets, transforms, models",
        "skimage": "scikit-image — image processing",
        "albumentations": "Albumentations — augmentation",
    }
    print("=== CV Library Status ===")
    for lib, desc in libs.items():
        avail = importlib.util.find_spec(lib) is not None
        print(f"  {'[OK]' if avail else '[X]'} {lib:20s} — {desc}")
    print()

def demo_numpy_cv():
    print("=== Numpy Image Processing Demo ===")
    digits = load_digits()
    img = digits.images[7] / 16.0  # (8, 8) normalised

    # Simple operations
    flipped = img[:, ::-1]
    rotated = np.rot90(img)
    zoomed  = np.kron(img, np.ones((4, 4)))  # 4x upscale via Kronecker

    fig, axes = plt.subplots(1, 4, figsize=(10, 2.5))
    for ax, im, title in zip(axes, [img, flipped, rotated, zoomed],
                              ["Original 8×8", "H-Flipped", "Rotated 90°", "Upscaled ×4"]):
        ax.imshow(im, cmap="gray"); ax.set_title(title); ax.axis("off")
    plt.tight_layout(); plt.savefig(OUTPUT / "cv_tools.png"); plt.close()
    print("  Saved cv_tools.png")
    print(f"  Upscaled shape: {zoomed.shape}")

def tool_matrix():
    rows = [
        ("OpenCV", "cv2", "Detection, optical flow, C++ interop", "Production"),
        ("Pillow", "PIL", "Format conversion, basic transforms", "Preprocessing"),
        ("torchvision", "torch", "Models, transforms, datasets", "DL research"),
        ("scikit-image", "skimage", "Scientific image processing", "Analysis"),
        ("Albumentations", "albumentations", "Fast GPU augmentation", "Training"),
    ]
    print("\n  CV Tool Matrix:")
    print(f"  {'Library':15s} {'Strengths':35s} {'Use case':15s}")
    print("  " + "-"*68)
    for r in rows:
        print(f"  {r[0]:15s} {r[2]:35s} {r[3]:15s}")

def demo_convolution_filters():
    """Apply low-pass (blur) and high-pass (sharpen) filters."""
    print("\n=== Convolution Filters ===")
    digits = load_digits()
    img = digits.images[2] / 16.0

    def conv2d_filter(img, K):
        H, W = img.shape; kH, kW = K.shape; pad = kH // 2
        p = np.pad(img, pad); out = np.zeros_like(img)
        for i in range(H):
            for j in range(W):
                out[i, j] = (p[i:i+kH, j:j+kW] * K).sum()
        return out

    blur = np.ones((3,3)) / 9
    sharpen = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], float)
    blurred   = conv2d_filter(img, blur).clip(0, 1)
    sharpened = conv2d_filter(img, sharpen).clip(0, 1)
    print(f"  Original std:  {img.std():.4f}")
    print(f"  Blurred  std:  {blurred.std():.4f}")
    print(f"  Sharpened std: {sharpened.std():.4f}")


def demo_histogram_equalization():
    """Histogram equalisation to enhance contrast."""
    print("\n=== Histogram Equalisation ===")
    digits = load_digits()
    img = (digits.images[0] / 16.0 * 255).astype(int)
    hist, _ = np.histogram(img.ravel(), 256, [0, 256])
    cdf = hist.cumsum().astype(float)
    cdf_norm = (cdf - cdf.min()) / (cdf.max() - cdf.min()) * 255
    equalized = cdf_norm[img] / 255.0
    print(f"  Original contrast (std):  {img.std()/255:.4f}")
    print(f"  Equalised contrast (std): {equalized.std():.4f}")


if __name__ == "__main__":
    check_libs()
    demo_numpy_cv()
    tool_matrix()
    demo_convolution_filters()
    demo_histogram_equalization()
