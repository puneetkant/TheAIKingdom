"""
Working Example: Image Basics
Covers pixel representation, colour spaces, image operations,
and histograms — all with numpy (no OpenCV required).
"""
import numpy as np
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_image_basics")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── synthetic test image helpers ──────────────────────────────────────────────
def make_test_image(H=64, W=64):
    """Create a simple synthetic RGB image (gradient + circles)."""
    rng = np.random.default_rng(0)
    img = np.zeros((H, W, 3), dtype=np.uint8)
    # Gradient background
    for c in range(3):
        img[:, :, c] = np.tile(np.linspace(30, 220, W, dtype=np.uint8), (H, 1))
    # Red circle
    for i in range(H):
        for j in range(W):
            if (i-20)**2 + (j-20)**2 < 100:
                img[i, j] = [220, 50, 50]
    # Blue rectangle
    img[35:55, 35:55] = [50, 50, 200]
    return img


# ── 1. Pixel representation ───────────────────────────────────────────────────
def pixel_representation():
    print("=== Pixel Representation ===")
    img = make_test_image()
    print(f"  Image shape: {img.shape}  (H, W, C)")
    print(f"  dtype:  {img.dtype}  range: [{img.min()}, {img.max()}]")
    print(f"  Pixel at (0,0): R={img[0,0,0]} G={img[0,0,1]} B={img[0,0,2]}")
    print(f"  Red channel shape: {img[:,:,0].shape}")
    print()
    print("  Common formats:")
    print("    uint8  : 0-255     (standard display)")
    print("    float32: 0.0-1.0   (normalised, neural network input)")
    print("    float32: -1 to 1   (some generative models)")
    print()
    # Normalise
    img_f = img.astype(np.float32) / 255.0
    print(f"  Normalised pixel at (0,0): {img_f[0,0]}")
    print()
    print("  Image sizes / storage:")
    for H, W in [(28,28), (224,224), (512,512), (1920,1080)]:
        nbytes = H * W * 3
        print(f"    {H}×{W} RGB: {nbytes:>8,} bytes ({nbytes/1024**2:.2f} MB) uncompressed")


# ── 2. Colour spaces ─────────────────────────────────────────────────────────
def colour_spaces():
    print("\n=== Colour Spaces ===")

    def rgb_to_grayscale(img):
        """ITU-R BT.601 luminance."""
        return (0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2])

    def rgb_to_hsv(rgb):
        r, g, b = rgb[:,:,0]/255., rgb[:,:,1]/255., rgb[:,:,2]/255.
        Cmax = np.maximum(r, np.maximum(g, b))
        Cmin = np.minimum(r, np.minimum(g, b))
        D    = Cmax - Cmin + 1e-9
        H    = np.where(Cmax == r, (g - b) / D % 6,
               np.where(Cmax == g, (b - r) / D + 2, (r - g) / D + 4)) / 6
        S    = np.where(Cmax > 0, D / Cmax, 0)
        V    = Cmax
        return np.stack([H, S, V], axis=-1)

    img  = make_test_image()
    gray = rgb_to_grayscale(img)
    hsv  = rgb_to_hsv(img)

    print(f"  RGB  shape: {img.shape}   range [{img.min()}, {img.max()}]")
    print(f"  Gray shape: {gray.shape}  range [{gray.min():.1f}, {gray.max():.1f}]")
    print(f"  HSV  shape: {hsv.shape}   H∈[0,1] S∈[0,1] V∈[0,1]")

    # Save comparison
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    axes[0].imshow(img); axes[0].set_title("RGB"); axes[0].axis("off")
    axes[1].imshow(gray, cmap="gray"); axes[1].set_title("Gray"); axes[1].axis("off")
    axes[2].imshow(img[:,:,0], cmap="Reds"); axes[2].set_title("R channel"); axes[2].axis("off")
    axes[3].imshow(hsv[:,:,0], cmap="hsv"); axes[3].set_title("Hue"); axes[3].axis("off")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "colour_spaces.png")
    plt.savefig(path, dpi=90); plt.close()
    print(f"\n  Colour spaces plot: {path}")

    print()
    print("  Common colour spaces:")
    cs = [("RGB",  "Red, Green, Blue — display standard"),
          ("BGR",  "OpenCV default (channels reversed from RGB)"),
          ("Gray", "Single channel luminance"),
          ("HSV",  "Hue, Saturation, Value — human-intuitive"),
          ("Lab",  "Perceptually uniform; L=lightness, a, b=colour"),
          ("YCbCr","Luma + chroma; used in JPEG, video codecs")]
    for c, d in cs:
        print(f"    {c:<6} {d}")


# ── 3. Image operations ───────────────────────────────────────────────────────
def image_operations():
    print("\n=== Image Operations ===")
    img = make_test_image(64, 64).astype(float)

    # Crop
    crop = img[10:40, 10:40]
    print(f"  Crop [10:40, 10:40]: {img.shape} → {crop.shape}")

    # Resize (nearest neighbour)
    def resize_nn(im, new_h, new_w):
        oh, ow = im.shape[:2]
        iy = (np.arange(new_h) * oh / new_h).astype(int)
        ix = (np.arange(new_w) * ow / new_w).astype(int)
        return im[np.ix_(iy, ix)]

    small = resize_nn(img, 32, 32)
    big   = resize_nn(img, 128, 128)
    print(f"  Resize NN: 64×64 → {small.shape[:2]} and → {big.shape[:2]}")

    # Flip
    hflip = img[:, ::-1]
    vflip = img[::-1, :]
    print(f"  Horizontal flip: {hflip.shape}")
    print(f"  Vertical flip:   {vflip.shape}")

    # Brightness / contrast
    bright = np.clip(img * 1.3 + 20, 0, 255)
    dark   = np.clip(img * 0.5, 0, 255)
    print(f"  Brightness +30%: mean {bright.mean():.1f}  (original {img.mean():.1f})")

    # Gaussian blur (3×3 kernel)
    def gaussian_blur(im, kernel_size=3):
        k = np.array([[1,2,1],[2,4,2],[1,2,1]]) / 16.
        from scipy.ndimage import convolve
        out = np.stack([convolve(im[:,:,c], k) for c in range(3)], axis=-1)
        return out

    try:
        blurred = gaussian_blur(img)
        print(f"  Gaussian blur (3×3): max diff {np.abs(blurred - img).max():.2f}")
    except Exception:
        print("  Gaussian blur: scipy not available")


# ── 4. Histograms ────────────────────────────────────────────────────────────
def histograms():
    print("\n=== Image Histograms ===")
    img = make_test_image()

    # Compute per-channel histograms
    for c, name in [(0,"Red"), (1,"Green"), (2,"Blue")]:
        h, _ = np.histogram(img[:,:,c], bins=8, range=(0,256))
        bar   = "".join("█" * int(v / h.max() * 8) for v in h)
        print(f"  {name}: [{bar}] mean={img[:,:,c].mean():.0f}")

    # Histogram equalisation (grayscale)
    gray = (0.299*img[:,:,0] + 0.587*img[:,:,1] + 0.114*img[:,:,2]).astype(np.uint8)
    hist = np.bincount(gray.flatten(), minlength=256).astype(float)
    cdf  = hist.cumsum()
    cdf_norm = (cdf - cdf.min()) / (cdf.max() - cdf.min()) * 255
    eq_gray  = cdf_norm[gray]

    print(f"\n  Histogram equalisation:")
    print(f"  Gray: mean={gray.mean():.1f}  std={gray.std():.1f}")
    print(f"  Equalized: mean={eq_gray.mean():.1f}  std={eq_gray.std():.1f}")

    # Save histogram plot
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    axes[0].imshow(img)
    axes[0].set_title("Image"); axes[0].axis("off")
    for c, col, name in [(0,"red","Red"),(1,"green","Green"),(2,"blue","Blue")]:
        axes[1].hist(img[:,:,c].flatten(), bins=32, range=(0,256),
                     color=col, alpha=0.6, label=name)
    axes[1].legend(); axes[1].set_title("RGB Histogram")
    axes[2].hist(gray.flatten(), bins=32, range=(0,256), color="gray", alpha=0.6, label="orig")
    axes[2].hist(eq_gray.flatten(), bins=32, range=(0,256), color="blue", alpha=0.4, label="equalized")
    axes[2].legend(); axes[2].set_title("Histogram Equalisation")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "histograms.png")
    plt.savefig(path, dpi=90); plt.close()
    print(f"  Histogram plot: {path}")


if __name__ == "__main__":
    pixel_representation()
    colour_spaces()
    image_operations()
    histograms()
