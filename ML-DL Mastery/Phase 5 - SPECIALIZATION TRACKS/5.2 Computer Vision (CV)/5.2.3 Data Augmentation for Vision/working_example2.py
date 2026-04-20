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

def demo_augmentation_effect():
    """Show that augmentation can improve accuracy on a small dataset."""
    print("\n=== Augmentation Effect on Accuracy ===")
    from sklearn.svm import SVC
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    digits = load_digits()
    X_raw = digits.data / 16.0; y = digits.target
    X_tr, X_te, y_tr, y_te = train_test_split(X_raw, y, train_size=0.1, random_state=42)
    svm = SVC(kernel="rbf", C=5).fit(X_tr, y_tr)
    acc_baseline = accuracy_score(y_te, svm.predict(X_te))
    aug_X, aug_y = list(X_tr), list(y_tr)
    for xi, yi in zip(X_tr.reshape(-1, 8, 8), y_tr):
        aug_X.append(add_noise(xi).ravel()); aug_y.append(yi)
    svm_aug = SVC(kernel="rbf", C=5).fit(np.array(aug_X), np.array(aug_y))
    acc_aug = accuracy_score(y_te, svm_aug.predict(X_te))
    print(f"  Train: {len(X_tr)} -> augmented to {len(aug_X)}")
    print(f"  Baseline accuracy: {acc_baseline:.4f}  Augmented: {acc_aug:.4f}")


def demo_mixup():
    """MixUp augmentation: blend two samples and their labels."""
    print("\n=== MixUp Augmentation ===")
    digits = load_digits()
    X = digits.data / 16.0; y = digits.target
    for alpha in [0.2, 0.5, 0.8]:
        i, j = 0, 1
        mixed_y = alpha * float(y[i]) + (1-alpha) * float(y[j])
        print(f"  alpha={alpha}: soft label={mixed_y:.2f}  (original: {y[i]} and {y[j]})")


if __name__ == "__main__":
    demo()
    demo_augmentation_effect()
    demo_mixup()
