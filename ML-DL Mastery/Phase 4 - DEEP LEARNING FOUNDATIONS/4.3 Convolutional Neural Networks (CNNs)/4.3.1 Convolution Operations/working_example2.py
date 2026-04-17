"""
Working Example 2: Convolution Operations — manual conv2d, stride, padding effects
===================================================================================
Implements 2D convolution from scratch, compares edge detection filters.

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

def conv2d(image, kernel, stride=1, padding=0):
    """Manual 2D convolution (single-channel)."""
    if padding > 0:
        image = np.pad(image, padding, mode="constant")
    kh, kw = kernel.shape; ih, iw = image.shape
    oh = (ih - kh) // stride + 1; ow = (iw - kw) // stride + 1
    out = np.zeros((oh, ow))
    for i in range(oh):
        for j in range(ow):
            out[i, j] = (image[i*stride:i*stride+kh, j*stride:j*stride+kw] * kernel).sum()
    return out

def demo_filters():
    print("=== Edge Detection Filters ===")
    # Synthetic 16×16 image with a bright square
    img = np.zeros((16, 16)); img[4:12, 4:12] = 1.0

    kernels = {
        "Sobel-H": np.array([[-1,0,1],[-2,0,2],[-1,0,1]], float),
        "Sobel-V": np.array([[-1,-2,-1],[0,0,0],[1,2,1]], float),
        "Laplace": np.array([[0,1,0],[1,-4,1],[0,1,0]], float),
        "Blur":    np.ones((3,3))/9,
    }
    fig, axes = plt.subplots(1, 5, figsize=(14, 3))
    axes[0].imshow(img, cmap="gray"); axes[0].set_title("Input")
    for ax, (name, k) in zip(axes[1:], kernels.items()):
        out = conv2d(img, k, padding=1)
        ax.imshow(out, cmap="gray"); ax.set_title(name)
        print(f"  {name:10s}: output shape={out.shape}  max={out.max():.2f}")
    for ax in axes: ax.axis("off")
    plt.tight_layout(); plt.savefig(OUTPUT / "conv_filters.png"); plt.close()
    print("  Saved conv_filters.png")

def demo_stride_padding():
    print("\n=== Stride and Padding Effects ===")
    img = np.random.default_rng(0).standard_normal((8, 8))
    k = np.ones((3,3))/9
    for stride in [1, 2]:
        for pad in [0, 1]:
            out = conv2d(img, k, stride=stride, padding=pad)
            print(f"  stride={stride} pad={pad}: output shape={out.shape}")

if __name__ == "__main__":
    demo_filters()
    demo_stride_padding()
