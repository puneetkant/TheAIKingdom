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

def demo_1d_convolution():
    print("\n=== 1D Convolution ===")
    signal = np.array([1, 2, 3, 4, 3, 2, 1, 2, 3], dtype=float)
    kernel = np.array([1, 0, -1], dtype=float)

    valid = np.array([
        (signal[i:i+len(kernel)] * kernel).sum()
        for i in range(len(signal) - len(kernel) + 1)
    ])
    padded_same = np.pad(signal, len(kernel) // 2, mode="constant")
    same = np.array([
        (padded_same[i:i+len(kernel)] * kernel).sum()
        for i in range(len(signal))
    ])
    full_pad = len(kernel) - 1
    padded_full = np.pad(signal, full_pad, mode="constant")
    full = np.array([
        (padded_full[i:i+len(kernel)] * kernel).sum()
        for i in range(len(signal) + full_pad)
    ])
    print(f"  Signal:        {signal.astype(int)}")
    print(f"  Kernel [1,0,-1]:")
    print(f"  Valid  ({len(valid)}):  {valid.astype(int)}")
    print(f"  Same   ({len(same)}):  {same.astype(int)}")
    print(f"  Full   ({len(full)}): {full.astype(int)}")
    edge_k = np.array([1, -1], dtype=float)
    edges = np.array([
        (signal[i:i+2] * edge_k).sum()
        for i in range(len(signal) - 1)
    ])
    print(f"  Edge [1,-1] valid ({len(edges)}): {edges.astype(int)}")


def demo_stride_and_dilation():
    print("\n=== Stride and Dilation ===")
    rng = np.random.default_rng(5)
    img = rng.standard_normal((8, 8))
    k = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=float)

    def conv2d_dilated(image, kernel, stride=1, dilation=1):
        kh, kw = kernel.shape
        eff_kh = kh + (kh - 1) * (dilation - 1)
        eff_kw = kw + (kw - 1) * (dilation - 1)
        ih, iw = image.shape
        oh = (ih - eff_kh) // stride + 1
        ow = (iw - eff_kw) // stride + 1
        out = np.zeros((oh, ow))
        for i in range(oh):
            for j in range(ow):
                val = 0.0
                for ki in range(kh):
                    for kj in range(kw):
                        ri = i * stride + ki * dilation
                        ci = j * stride + kj * dilation
                        val += image[ri, ci] * kernel[ki, kj]
                out[i, j] = val
        return out

    out1 = conv2d_dilated(img, k, stride=1, dilation=1)
    out2 = conv2d_dilated(img, k, stride=2, dilation=1)
    out3 = conv2d_dilated(img, k, stride=1, dilation=2)
    print(f"  Input shape:              {img.shape}")
    print(f"  stride=1, dilation=1: shape={out1.shape}, first row={out1[0].round(3)}")
    print(f"  stride=2, dilation=1: shape={out2.shape}, first row={out2[0].round(3)}")
    print(f"  stride=1, dilation=2: shape={out3.shape}, first row={out3[0].round(3)}")


def demo_depthwise_separable():
    print("\n=== Depthwise Separable Convolution ===")
    rng = np.random.default_rng(3)
    H, W, C_in, C_out = 4, 4, 3, 2
    inp = rng.standard_normal((H, W, C_in))
    dw_filters = rng.standard_normal((3, 3, C_in))
    pw_filters = rng.standard_normal((C_in, C_out))

    def depthwise_conv(inp, filters):
        H, W, C = inp.shape
        out = np.zeros((H, W, C))
        for c in range(C):
            ch = np.pad(inp[:, :, c], 1, mode="constant")
            f = filters[:, :, c]
            for i in range(H):
                for j in range(W):
                    out[i, j, c] = (ch[i:i+3, j:j+3] * f).sum()
        return out

    dw_out = depthwise_conv(inp, dw_filters)
    pw_out = (dw_out.reshape(H * W, C_in) @ pw_filters).reshape(H, W, C_out)
    print(f"  Input shape:             {inp.shape}")
    print(f"  After depthwise conv:    {dw_out.shape}")
    print(f"  After pointwise (1x1):   {pw_out.shape}")
    k = 3
    regular_params = k * k * C_in * C_out
    dw_params      = k * k * C_in
    pw_params      = C_in * C_out
    sep_params     = dw_params + pw_params
    ratio          = sep_params / regular_params
    print(f"  Regular conv params (3x3 {C_in}->{C_out}): {regular_params}")
    print(f"  Depthwise params: {dw_params}  Pointwise params: {pw_params}")
    print(f"  Separable total: {sep_params}")
    print(f"  Reduction ratio: {ratio:.3f}  ({ratio*100:.1f}% of regular)")


def demo_receptive_field():
    print("\n=== Receptive Field Analysis ===")

    def rf_standard(num_layers, k=3):
        return 1 + num_layers * (k - 1)

    def rf_dilated(num_layers, k=3, d=2):
        eff_k = d * (k - 1) + 1
        return 1 + num_layers * (eff_k - 1)

    configs = [
        ("1 layer,  3x3, dilation=1", rf_standard(1)),
        ("2 layers, 3x3, dilation=1", rf_standard(2)),
        ("3 layers, 3x3, dilation=1", rf_standard(3)),
        ("1 layer,  3x3, dilation=2", rf_dilated(1)),
        ("2 layers, 3x3, dilation=2", rf_dilated(2)),
    ]
    for desc, rf in configs:
        print(f"  {desc}: RF = {rf}")


if __name__ == "__main__":
    demo_filters()
    demo_stride_padding()
    demo_1d_convolution()
    demo_stride_and_dilation()
    demo_depthwise_separable()
    demo_receptive_field()
