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

if __name__ == "__main__":
    check_libs()
    demo_numpy_cv()
    tool_matrix()
