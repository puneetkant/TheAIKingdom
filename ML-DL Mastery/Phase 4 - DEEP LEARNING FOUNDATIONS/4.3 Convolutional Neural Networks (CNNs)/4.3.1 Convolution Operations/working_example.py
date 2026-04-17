"""
Working Example: Convolution Operations
Covers 1D/2D convolution, kernels, stride, padding, dilated convolution,
transposed convolution, depthwise separable convolution, and feature maps.
"""
import numpy as np
from scipy.signal import convolve2d
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_convolution")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Manual 2D convolution ─────────────────────────────────────────────────────
def conv2d(X, K, stride=1, padding=0):
    """2D cross-correlation (standard 'convolution' in DL)."""
    if padding > 0:
        X = np.pad(X, pad_width=padding, mode="constant")
    H, W     = X.shape
    kH, kW   = K.shape
    outH = (H - kH) // stride + 1
    outW = (W - kW) // stride + 1
    out  = np.zeros((outH, outW))
    for i in range(outH):
        for j in range(outW):
            out[i, j] = (X[i*stride:i*stride+kH, j*stride:j*stride+kW] * K).sum()
    return out


# ── 1. Basic convolution ──────────────────────────────────────────────────────
def basic_convolution():
    print("=== Basic 2D Convolution ===")
    print("  Output[i,j] = Σ Σ Input[i+m, j+n] · Kernel[m,n]")
    print("  DL uses cross-correlation (no kernel flip); called 'convolution' by convention")

    # 5×5 input, 3×3 kernel
    X = np.array([
        [1, 2, 3, 0, 1],
        [0, 1, 2, 3, 1],
        [1, 0, 1, 2, 3],
        [2, 1, 0, 1, 2],
        [3, 2, 1, 0, 1],
    ], dtype=float)

    # Edge detection kernel (Sobel-X)
    K_sobel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)
    K_blur  = np.ones((3, 3)) / 9   # average blur

    for name, K in [("Sobel-X (edge)", K_sobel), ("Average blur", K_blur)]:
        out = conv2d(X, K, stride=1, padding=0)
        print(f"\n  Kernel: {name}")
        print(f"  Input shape:  {X.shape}")
        print(f"  Output shape: {out.shape}  (5-3+1=3)")
        print(f"  Output:\n{out.round(2)}")


# ── 2. Padding and stride ─────────────────────────────────────────────────────
def padding_and_stride():
    print("\n=== Padding and Stride ===")
    print("  Output size: floor((H + 2P - kH) / S) + 1")
    print()
    H, kH = 7, 3
    for S in [1, 2]:
        for P in [0, 1]:
            outH = (H + 2*P - kH) // S + 1
            print(f"  H={H}  k={kH}  P={P}  S={S}  →  output={outH}")

    print()
    print("  'same' padding: P = (kH-1)/2  → output same spatial size as input")
    print("  'valid' padding: P = 0        → no padding (output shrinks)")

    X = np.ones((6, 6)) * 3
    K = np.ones((3, 3))
    for stride, pad, desc in [(1, 0, "no pad, stride 1"),
                               (1, 1, "pad=1 (same), stride 1"),
                               (2, 0, "no pad, stride 2"),
                               (2, 1, "pad=1, stride 2")]:
        out = conv2d(X, K, stride=stride, padding=pad)
        print(f"  {desc:<30}: output={out.shape}")


# ── 3. Multiple channels ──────────────────────────────────────────────────────
def multi_channel_conv():
    print("\n=== Multi-Channel Convolution ===")
    print("  Input:  (H, W, C_in)")
    print("  Kernel: (kH, kW, C_in, C_out)  — one filter per output channel")
    print("  Output: (H', W', C_out)")
    print()
    print("  For each output channel:")
    print("    out[..., c] = bias[c] + Σ_cin conv2d(in[...,cin], K[...,cin,c])")

    rng    = np.random.default_rng(0)
    H, W   = 8, 8
    C_in   = 3   # RGB
    C_out  = 4   # 4 filters
    kH = kW = 3

    X_in  = rng.standard_normal((H, W, C_in))
    K_all = rng.standard_normal((kH, kW, C_in, C_out)) * 0.1
    b     = rng.standard_normal(C_out) * 0.01

    # Forward
    out  = np.zeros((H-kH+1, W-kW+1, C_out))
    for cout in range(C_out):
        for cin in range(C_in):
            out[:, :, cout] += conv2d(X_in[:, :, cin], K_all[:, :, cin, cout])
        out[:, :, cout] += b[cout]

    params = kH * kW * C_in * C_out + C_out
    print(f"\n  Input:  ({H},{W},{C_in})")
    print(f"  Kernel: ({kH},{kW},{C_in},{C_out})")
    print(f"  Output: {out.shape}")
    print(f"  Parameters: k²·Cin·Cout + Cout = {params}")


# ── 4. Dilated (atrous) convolution ──────────────────────────────────────────
def dilated_convolution():
    print("\n=== Dilated Convolution ===")
    print("  Expands receptive field without increasing kernel size or parameters")
    print("  Kernel elements are spaced by dilation factor d")
    print("  Effective kernel size: k + (k-1)(d-1) = d(k-1) + 1")

    def conv2d_dilated(X, K, dilation):
        H, W   = X.shape
        kH, kW = K.shape
        dkH    = kH + (kH-1)*(dilation-1)  # effective kernel size
        dkW    = kW + (kW-1)*(dilation-1)
        outH   = H - dkH + 1
        outW   = W - dkW + 1
        out    = np.zeros((max(outH,0), max(outW,0)))
        for i in range(out.shape[0]):
            for j in range(out.shape[1]):
                patch_sum = 0
                for m in range(kH):
                    for n in range(kW):
                        patch_sum += X[i + m*dilation, j + n*dilation] * K[m, n]
                out[i, j] = patch_sum
        return out

    X = np.eye(10)
    K = np.ones((3, 3)) / 9
    for d in [1, 2, 3]:
        out = conv2d_dilated(X, K, dilation=d)
        eff = 3 + (3-1)*(d-1)
        print(f"  Dilation d={d}: effective kernel={eff}×{eff}  output={out.shape}")


# ── 5. Depthwise separable convolution ────────────────────────────────────────
def depthwise_separable():
    print("\n=== Depthwise Separable Convolution ===")
    print("  Standard conv: k²·Cin·Cout parameters")
    print("  DSConv: k²·Cin (depthwise) + Cin·Cout (pointwise 1×1)")
    print("  Reduction factor: 1/Cout + 1/k²  ≈ 8-9× fewer params for k=3")
    print()

    H, W   = 16, 16
    Cin    = 32
    Cout   = 64
    k      = 3

    std_params = k**2 * Cin * Cout
    dw_params  = k**2 * Cin
    pw_params  = Cin * Cout
    ds_params  = dw_params + pw_params

    print(f"  Standard conv: {std_params} params")
    print(f"  DSConv:        {ds_params} params  ({ds_params/std_params*100:.1f}% of standard)")
    print(f"  Used in:       MobileNet, Xception, EfficientNet")


# ── 6. Receptive field ────────────────────────────────────────────────────────
def receptive_field():
    print("\n=== Receptive Field ===")
    print("  RF: region of input that influences a single output unit")
    print("  Stacking conv layers grows RF: RF_L = 1 + Σ (k_l - 1) · Πs_i")
    print()
    # k=3, s=1, no padding
    for n_layers in range(1, 7):
        rf = 1 + (3-1)*n_layers  # stride=1
        print(f"  {n_layers} layers (k=3, s=1): RF = {rf}")
    print()
    print("  With stride=2 at layer 2:")
    rf = 1 + (3-1)*1 + (3-1)*2*1  # rough estimate
    print(f"  Layer1(s=1): RF=3  Layer2(s=2): RF increases faster")


# ── 7. Visualise kernels ──────────────────────────────────────────────────────
def visualise_kernels():
    rng = np.random.default_rng(1)
    # Classic image processing kernels
    kernels = {
        "Edge (Sobel-X)": np.array([[-1,0,1],[-2,0,2],[-1,0,1]]),
        "Edge (Sobel-Y)": np.array([[-1,-2,-1],[0,0,0],[1,2,1]]),
        "Blur (Gauss)":   np.array([[1,2,1],[2,4,2],[1,2,1]]) / 16,
        "Sharpen":        np.array([[0,-1,0],[-1,5,-1],[0,-1,0]]),
        "Emboss":         np.array([[-2,-1,0],[-1,1,1],[0,1,2]]),
        "Laplacian":      np.array([[0,1,0],[1,-4,1],[0,1,0]]),
    }
    # Simple gradient test image
    img = np.zeros((16, 16))
    img[:8, :8] = 1.0
    img[4:12, 4:12] = 0.5

    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    axes = axes.ravel()
    axes[0].imshow(img, cmap='gray'); axes[0].set_title("Original")
    for ax, (name, K) in zip(axes[1:], kernels.items()):
        out = conv2d(img, K, padding=0)
        ax.imshow(out, cmap='RdBu_r'); ax.set_title(name, fontsize=9)
    for ax in axes[len(kernels)+1:]: ax.axis('off')
    plt.suptitle("Convolution with Classic Kernels", fontsize=13)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "conv_kernels.png")
    plt.savefig(path, dpi=90); plt.close()
    print(f"\n  Kernel visualisation saved: {path}")


if __name__ == "__main__":
    basic_convolution()
    padding_and_stride()
    multi_channel_conv()
    dilated_convolution()
    depthwise_separable()
    receptive_field()
    visualise_kernels()
