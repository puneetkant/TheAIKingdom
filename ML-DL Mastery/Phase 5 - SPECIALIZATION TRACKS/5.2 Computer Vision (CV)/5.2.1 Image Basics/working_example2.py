"""
Working Example 2: Image Basics — pixel operations, color channels, histogram
==============================================================================
Demonstrates image fundamentals using numpy only (no OpenCV).

Run:  python working_example2.py
"""
from pathlib import Path
try:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit("pip install numpy matplotlib")

OUTPUT = Path(__file__).parent / "output"
OUTPUT.mkdir(exist_ok=True)

def make_synthetic_image(H=64, W=64):
    rng = np.random.default_rng(42)
    img = np.zeros((H, W, 3), dtype=np.float32)
    # Red gradient
    img[:, :, 0] = np.linspace(0, 1, W)[None, :]
    # Green gradient
    img[:, :, 1] = np.linspace(0, 1, H)[:, None]
    # Blue noise
    img[:, :, 2] = rng.random((H, W)) * 0.3
    img = np.clip(img, 0, 1)
    return img

def demo():
    print("=== Image Basics ===")
    img = make_synthetic_image()
    print(f"  Image shape: {img.shape}  dtype: {img.dtype}")
    print(f"  Pixel [32,32]: R={img[32,32,0]:.3f} G={img[32,32,1]:.3f} B={img[32,32,2]:.3f}")

    # Grayscale conversion (luminance formula)
    gray = 0.299*img[:,:,0] + 0.587*img[:,:,1] + 0.114*img[:,:,2]
    print(f"  Grayscale shape: {gray.shape}  mean={gray.mean():.3f}")

    # Brightness / contrast adjustments
    bright = np.clip(img * 1.5, 0, 1)
    dark   = img * 0.5
    contrast = np.clip((img - 0.5) * 2 + 0.5, 0, 1)

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes[0,0].imshow(img);         axes[0,0].set_title("Original")
    axes[0,1].imshow(gray, cmap="gray"); axes[0,1].set_title("Grayscale")
    axes[0,2].imshow(bright);      axes[0,2].set_title("Brightness +50%")
    axes[1,0].imshow(dark);        axes[1,0].set_title("Brightness -50%")
    axes[1,1].imshow(contrast);    axes[1,1].set_title("High Contrast")
    # Histogram
    axes[1,2].hist(gray.ravel(), bins=32, color="gray")
    axes[1,2].set_title("Intensity Histogram")
    for ax in axes.flat: ax.axis("off") if ax != axes[1,2] else None
    plt.tight_layout(); plt.savefig(OUTPUT / "image_basics.png"); plt.close()
    print("  Saved image_basics.png")

def demo_edge_detection():
    """Sobel-like edge detection using numpy convolution."""
    print("\n=== Edge Detection (Sobel-style) ===")
    img = make_synthetic_image()
    gray = 0.299*img[:,:,0] + 0.587*img[:,:,1] + 0.114*img[:,:,2]
    Kx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=float)
    Ky = np.array([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=float)

    def convolve2d(im, kernel):
        H, W = im.shape; kh, kw = kernel.shape
        ph, pw = kh//2, kw//2
        padded = np.pad(im, ((ph,ph),(pw,pw)), mode="reflect")
        out = np.zeros_like(im)
        for i in range(H):
            for j in range(W):
                out[i,j] = (padded[i:i+kh, j:j+kw] * kernel).sum()
        return out

    Gx = convolve2d(gray, Kx); Gy = convolve2d(gray, Ky)
    G  = np.sqrt(Gx**2 + Gy**2)
    G_norm = G / (G.max() + 1e-8)
    print(f"  Gradient magnitude: min={G_norm.min():.4f}  max={G_norm.max():.4f}")
    print(f"  Strong edge pixels (>0.3): {(G_norm > 0.3).sum()}")


def demo_color_spaces():
    """Analyse RGB channels and derive HSV-style value/saturation."""
    print("\n=== Colour Space Analysis ===")
    img = make_synthetic_image()
    R, G, B = img[:,:,0], img[:,:,1], img[:,:,2]
    Cmax = img.max(axis=2); Cmin = img.min(axis=2)
    V = Cmax
    S = np.where(Cmax > 0, (Cmax - Cmin) / (Cmax + 1e-8), 0)
    print(f"  RGB means: R={R.mean():.3f} G={G.mean():.3f} B={B.mean():.3f}")
    print(f"  Value (brightness): mean={V.mean():.3f} std={V.std():.3f}")
    print(f"  Saturation:         mean={S.mean():.3f} std={S.std():.3f}")
    print(f"  Most saturated pixel: {np.unravel_index(S.argmax(), S.shape)}")


if __name__ == "__main__":
    demo()
    demo_edge_detection()
    demo_color_spaces()
