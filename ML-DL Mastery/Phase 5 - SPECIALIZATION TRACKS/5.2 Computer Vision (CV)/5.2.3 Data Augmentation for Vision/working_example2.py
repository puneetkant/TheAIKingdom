"""
Working Example 2: Data Augmentation for Vision — geometric and color transforms
=================================================================================
Demonstrates augmentation pipeline using numpy only.

Run:  python working_example2.py
"""
from pathlib import Path
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

def horizontal_flip(img): return img[:, ::-1]
def vertical_flip(img):   return img[::-1, :]
def rotate90(img):        return np.rot90(img)
def add_noise(img, std=0.05, seed=42):
    rng = np.random.default_rng(seed)
    return np.clip(img + rng.normal(0, std, img.shape), 0, 1)
def brightness(img, factor=1.5): return np.clip(img * factor, 0, 1)
def cutout(img, size=3, seed=42):
    rng = np.random.default_rng(seed); out = img.copy()
    r, c = rng.integers(0, img.shape[0]-size), rng.integers(0, img.shape[1]-size)
    out[r:r+size, c:c+size] = 0
    return out

def demo():
    print("=== Data Augmentation for Vision ===")
    digits = load_digits()
    img = digits.images[3] / 16.0  # normalise to [0,1]

    augmented = {
        "Original": img,
        "H-Flip": horizontal_flip(img),
        "Rotate 90°": rotate90(img),
        "Noise": add_noise(img),
        "Brightness": brightness(img, 1.8),
        "Cutout": cutout(img, size=3),
    }

    fig, axes = plt.subplots(2, 3, figsize=(9, 6))
    for ax, (name, aug) in zip(axes.flat, augmented.items()):
        ax.imshow(aug, cmap="gray"); ax.set_title(name); ax.axis("off")
    plt.suptitle("Augmentation Examples")
    plt.tight_layout(); plt.savefig(OUTPUT / "augmentation.png"); plt.close()
    print("  Saved augmentation.png")
    print(f"  {len(augmented)} augmentation types demonstrated")

if __name__ == "__main__":
    demo()
