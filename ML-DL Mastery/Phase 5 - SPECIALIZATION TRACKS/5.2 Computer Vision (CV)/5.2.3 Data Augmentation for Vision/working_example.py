"""
Working Example: Data Augmentation for Vision
Covers flipping, rotation, colour jitter, cutout, mixup, cutmix,
AutoAugment concepts, and augmentation best practices.
"""
import numpy as np
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_augmentation")
os.makedirs(OUTPUT_DIR, exist_ok=True)

RNG = np.random.default_rng(42)


def make_image(H=32, W=32, seed=0):
    """Synthetic colourful test image."""
    rng = np.random.default_rng(seed)
    img = np.zeros((H, W, 3), dtype=np.float32)
    img += np.linspace(0.1, 0.9, W)[None, :, None]
    # Add shapes
    img[5:15, 5:15, 0] = 0.9   # red square
    img[18:28, 18:28, 2] = 0.9  # blue square
    img += rng.uniform(0, 0.05, img.shape)  # slight noise
    return np.clip(img, 0, 1)


# -- 1. Geometric augmentations ------------------------------------------------
def flip(img, horizontal=True):
    return img[:, ::-1] if horizontal else img[::-1]


def rotate90(img, k=1):
    return np.rot90(img, k)


def random_crop(img, crop_h, crop_w, rng):
    H, W = img.shape[:2]
    i = rng.integers(0, H - crop_h + 1)
    j = rng.integers(0, W - crop_w + 1)
    return img[i:i+crop_h, j:j+crop_w]


def random_horizontal_flip(img, p=0.5, rng=None):
    rng = rng or np.random.default_rng(0)
    return img[:, ::-1] if rng.uniform() < p else img


def pad_and_crop(img, padding=4, rng=None):
    """Standard training augmentation: pad then random crop."""
    rng = rng or np.random.default_rng(0)
    H, W, C = img.shape
    padded = np.pad(img, ((padding, padding), (padding, padding), (0, 0)),
                    mode="reflect")
    return random_crop(padded, H, W, rng)


def geometric_augmentations():
    print("=== Geometric Augmentations ===")
    img = make_image()
    augs = {
        "Original":     img,
        "H-Flip":       flip(img, horizontal=True),
        "V-Flip":       flip(img, horizontal=False),
        "Rot90":        rotate90(img, k=1),
        "Pad+Crop":     pad_and_crop(img, padding=4, rng=RNG),
    }
    print(f"  Image shape: {img.shape}")
    for name, a in augs.items():
        print(f"  {name:<15} shape={a.shape}  mean={a.mean():.4f}")

    fig, axes = plt.subplots(1, len(augs), figsize=(14, 3))
    for ax, (name, a) in zip(axes, augs.items()):
        ax.imshow(np.clip(a, 0, 1)); ax.set_title(name); ax.axis("off")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "geometric_aug.png")
    plt.savefig(path, dpi=90); plt.close()
    print(f"  Plot: {path}")


# -- 2. Colour augmentations ---------------------------------------------------
def color_jitter(img, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, rng=None):
    rng = rng or np.random.default_rng(0)
    img = img.copy()
    # Brightness
    bf = 1 + rng.uniform(-brightness, brightness)
    img = np.clip(img * bf, 0, 1)
    # Contrast
    cf = 1 + rng.uniform(-contrast, contrast)
    mean = img.mean()
    img  = np.clip((img - mean) * cf + mean, 0, 1)
    # Saturation (convert to grayscale and mix)
    sf   = 1 + rng.uniform(-saturation, saturation)
    gray = (0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2])[:,:,None]
    img  = np.clip(gray + sf * (img - gray), 0, 1)
    # Hue: simple shift
    hf = rng.uniform(-hue, hue)
    img = np.roll(img, shift=int(hf * img.shape[1]), axis=1)
    return img.astype(np.float32)


def grayscale(img, p=0.2, rng=None):
    rng = rng or np.random.default_rng(0)
    if rng.uniform() < p:
        gray = (0.299*img[:,:,0] + 0.587*img[:,:,1] + 0.114*img[:,:,2])[:,:,None]
        return np.repeat(gray, 3, axis=2)
    return img


def colour_augmentations():
    print("\n=== Colour Augmentations ===")
    img  = make_image()
    jittered = color_jitter(img, rng=RNG)
    gray_img = grayscale(img, p=1.0, rng=RNG)

    print(f"  Original mean:  R={img[:,:,0].mean():.3f} G={img[:,:,1].mean():.3f} B={img[:,:,2].mean():.3f}")
    print(f"  Jittered mean:  R={jittered[:,:,0].mean():.3f} G={jittered[:,:,1].mean():.3f} B={jittered[:,:,2].mean():.3f}")
    print(f"  Grayscale:      all channels equal? {np.allclose(gray_img[:,:,0], gray_img[:,:,1])}")

    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    axes[0].imshow(img); axes[0].set_title("Original"); axes[0].axis("off")
    axes[1].imshow(jittered); axes[1].set_title("ColorJitter"); axes[1].axis("off")
    axes[2].imshow(gray_img); axes[2].set_title("Grayscale"); axes[2].axis("off")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "colour_aug.png")
    plt.savefig(path, dpi=90); plt.close()
    print(f"  Plot: {path}")


# -- 3. Cutout / Random Erasing -----------------------------------------------
def cutout(img, n_holes=1, length=8, rng=None):
    rng = rng or np.random.default_rng(0)
    img = img.copy()
    H, W = img.shape[:2]
    for _ in range(n_holes):
        y = rng.integers(0, H)
        x = rng.integers(0, W)
        y1, y2 = max(0, y - length//2), min(H, y + length//2)
        x1, x2 = max(0, x - length//2), min(W, x + length//2)
        img[y1:y2, x1:x2] = 0.5   # fill with gray
    return img


def cutout_demo():
    print("\n=== Cutout / Random Erasing ===")
    img = make_image()
    cut = cutout(img, n_holes=2, length=8, rng=RNG)
    print(f"  Original mean: {img.mean():.4f}")
    print(f"  Cutout mean:   {cut.mean():.4f}  (pixels zeroed)")
    print(f"  Cutout forces model to use multiple cues, not rely on one region")


# -- 4. MixUp -----------------------------------------------------------------
def mixup(img_a, img_b, label_a, label_b, alpha=0.4, rng=None):
    """MixUp augmentation: lambda·img_a + (1-lambda)·img_b"""
    rng = rng or np.random.default_rng(0)
    lam = rng.beta(alpha, alpha)
    img_mix  = lam * img_a + (1 - lam) * img_b
    # Soft label
    label_mix = lam * label_a + (1 - lam) * label_b
    return img_mix, label_mix, lam


def mixup_demo():
    print("\n=== MixUp Augmentation ===")
    img_a = make_image(seed=0)
    img_b = make_image(seed=1)
    label_a = np.array([1, 0, 0])   # one-hot
    label_b = np.array([0, 1, 0])

    mix, label_mix, lam = mixup(img_a, img_b, label_a, label_b, alpha=0.4, rng=RNG)
    print(f"  lambda = {lam:.4f}")
    print(f"  Mixed image shape: {mix.shape}")
    print(f"  Soft label: {np.round(label_mix, 4)}")
    print(f"  Mix = {lam:.2f}·img_a + {1-lam:.2f}·img_b")
    print()
    print("  MixUp improves calibration and generalisation")
    print("  Loss: CE(f(mix), label_mix) = lambda·CE(f(x_a), y_a) + (1-lambda)·CE(f(x_b), y_b)")


# -- 5. CutMix ----------------------------------------------------------------
def cutmix(img_a, img_b, label_a, label_b, alpha=1.0, rng=None):
    """CutMix: paste rectangular region of img_b into img_a."""
    rng = rng or np.random.default_rng(0)
    H, W = img_a.shape[:2]
    lam  = rng.beta(alpha, alpha)
    cut_w = int(W * np.sqrt(1 - lam))
    cut_h = int(H * np.sqrt(1 - lam))
    cx = rng.integers(0, W); cy = rng.integers(0, H)
    x1, x2 = max(0, cx - cut_w//2), min(W, cx + cut_w//2)
    y1, y2 = max(0, cy - cut_h//2), min(H, cy + cut_h//2)
    img_mix = img_a.copy()
    img_mix[y1:y2, x1:x2] = img_b[y1:y2, x1:x2]
    lam_adj  = 1 - (y2-y1)*(x2-x1) / (H * W)
    label_mix = lam_adj * label_a + (1 - lam_adj) * label_b
    return img_mix, label_mix


def cutmix_demo():
    print("\n=== CutMix Augmentation ===")
    img_a   = make_image(seed=0)
    img_b   = make_image(seed=3)
    label_a = np.array([1, 0])
    label_b = np.array([0, 1])
    mix, label = cutmix(img_a, img_b, label_a, label_b, rng=RNG)
    print(f"  Mixed image shape: {mix.shape}")
    print(f"  Soft label: {np.round(label, 4)}")
    print(f"  CutMix keeps local structure; better for dense prediction tasks")


# -- 6. AutoAugment / RandAugment ---------------------------------------------
def autoaugment_overview():
    print("\n=== AutoAugment / RandAugment ===")
    print("  AutoAugment (Cubuk et al. 2018):")
    print("    RL-based search to find optimal augmentation policy")
    print("    Policy: sequence of (operation, probability, magnitude)")
    print("    State-of-art accuracy but slow to find policy")
    print()
    print("  RandAugment (Cubuk et al. 2019):")
    print("    Random N operations from a predefined list, all with same magnitude M")
    print("    Only 2 hyperparameters: N, M")
    print()
    print("  Operations pool:")
    ops = ["Identity", "AutoContrast", "Equalize", "Rotate",
           "Solarize", "Color", "Posterize", "Contrast",
           "Brightness", "Sharpness", "ShearX", "ShearY",
           "TranslateX", "TranslateY"]
    print(f"    {ops}")
    print()
    print("  AugMix: mix multiple augmentation chains -> consistency loss")
    print("  TrivialAugment: single op sampled uniformly, magnitude also random")


if __name__ == "__main__":
    geometric_augmentations()
    colour_augmentations()
    cutout_demo()
    mixup_demo()
    cutmix_demo()
    autoaugment_overview()
