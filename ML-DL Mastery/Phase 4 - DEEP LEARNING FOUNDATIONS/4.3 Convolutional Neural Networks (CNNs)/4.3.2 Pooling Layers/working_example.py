"""
Working Example: Pooling Layers
Covers max pooling, average pooling, global pooling, adaptive pooling,
L2 pooling, stochastic pooling, and the role pooling plays in CNNs.
"""
import numpy as np
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_pooling")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -- Generic pooling helper ----------------------------------------------------
def _pool2d(X, pool_size, stride, fn):
    H, W = X.shape
    pH, pW = pool_size
    outH = (H - pH) // stride + 1
    outW = (W - pW) // stride + 1
    out  = np.zeros((outH, outW))
    for i in range(outH):
        for j in range(outW):
            region = X[i*stride:i*stride+pH, j*stride:j*stride+pW]
            out[i, j] = fn(region)
    return out

def max_pool2d(X, pool_size=2, stride=2):
    return _pool2d(X, (pool_size, pool_size), stride, np.max)

def avg_pool2d(X, pool_size=2, stride=2):
    return _pool2d(X, (pool_size, pool_size), stride, np.mean)

def l2_pool2d(X, pool_size=2, stride=2):
    return _pool2d(X, (pool_size, pool_size), stride, lambda r: np.sqrt((r**2).mean()))


# -- 1. Max pooling ------------------------------------------------------------
def max_pooling_demo():
    print("=== Max Pooling ===")
    print("  Output[i,j] = max of pool_size window")
    print("  Retains strongest activations -> translation invariance")

    X = np.array([
        [1, 3, 2, 4, 1, 0],
        [0, 5, 1, 2, 3, 1],
        [2, 1, 3, 0, 2, 4],
        [4, 0, 2, 1, 0, 3],
        [1, 2, 0, 4, 1, 2],
        [3, 1, 2, 0, 3, 1],
    ], dtype=float)

    out2 = max_pool2d(X, pool_size=2, stride=2)
    out3 = max_pool2d(X, pool_size=3, stride=3)
    print(f"\n  Input shape:   {X.shape}")
    print(f"  MaxPool(2,s=2) output:\n{out2}")
    print(f"\n  MaxPool(3,s=3) output:\n{out3}")


# -- 2. Average pooling --------------------------------------------------------
def average_pooling_demo():
    print("\n=== Average Pooling ===")
    print("  Output[i,j] = mean of pool_size window")
    print("  Smoother; used in inception modules and for spatial averaging")

    X = np.array([
        [4, 0, 4, 0],
        [0, 4, 0, 4],
        [4, 0, 4, 0],
        [0, 4, 0, 4],
    ], dtype=float)

    max_out = max_pool2d(X, pool_size=2, stride=2)
    avg_out = avg_pool2d(X, pool_size=2, stride=2)
    print(f"\n  Input:\n{X}")
    print(f"\n  Max Pool 2×2:\n{max_out}")
    print(f"\n  Avg Pool 2×2:\n{avg_out}")
    print(f"\n  Note: checkerboard pattern -> avg=2.0 everywhere; max=4.0 everywhere")


# -- 3. Global pooling ---------------------------------------------------------
def global_pooling_demo():
    print("\n=== Global Pooling ===")
    print("  Reduces each feature map to a single value (global max or avg)")
    print("  Common use: replace Flatten -> Dense at end of CNN")
    print("  Benefits: fewer parameters, spatial invariance, handles variable input size")

    rng  = np.random.default_rng(42)
    H, W = 7, 7
    C    = 16  # 16 channels
    feature_maps = rng.standard_normal((H, W, C))

    gap = feature_maps.mean(axis=(0, 1))  # Global Average Pooling
    gmp = feature_maps.max(axis=(0, 1))   # Global Max Pooling

    print(f"\n  Feature maps shape: {feature_maps.shape}")
    print(f"  GAP output shape:   {gap.shape}  (one value per channel)")
    print(f"  GMP output shape:   {gmp.shape}")
    print(f"\n  GAP first 5 values: {gap[:5].round(3)}")
    print(f"  GMP first 5 values: {gmp[:5].round(3)}")
    print(f"\n  Instead of Flatten({H*W*C}) -> Dense, use GAP -> Dense({C})")
    print(f"  Parameter savings: {H*W*C} -> {C} input to classifier")


# -- 4. Strided convolution vs pooling -----------------------------------------
def strided_vs_pooling():
    print("\n=== Strided Convolution vs Pooling ===")
    print("  Both downsample spatial dimensions")
    print()
    print(f"  {'Method':<30} {'Params':<15} {'Notes'}")
    rows = [
        ("Max Pool 2×2, stride 2",  "0",         "Hard downsampling; loses location info"),
        ("Avg Pool 2×2, stride 2",  "0",         "Smoother; used in ResNet average pooling"),
        ("Conv 3×3, stride 2",      "9·Cin·Cout","Learnable; can learn what to keep"),
        ("Conv 1×1, then pool",     "Cin·Cout",  "Bottleneck reduction then spatial"),
    ]
    for name, params, note in rows:
        print(f"  {name:<30} {params:<15} {note}")

    print()
    print("  Modern trend: replace pooling with strided convolution (All-CNN, ResNet v2)")


# -- 5. Pooling and translation invariance -------------------------------------
def translation_invariance():
    print("\n=== Translation Invariance via Pooling ===")

    # Shift a signal and show pool output is similar
    signal = np.zeros(10); signal[4] = 1.0   # spike at position 4
    shifted = np.zeros(10); shifted[5] = 1.0  # spike at position 5

    def max_pool_1d(x, size=2, stride=2):
        return np.array([x[i:i+size].max() for i in range(0, len(x)-size+1, stride)])

    out1 = max_pool_1d(signal)
    out2 = max_pool_1d(shifted)
    print(f"  Signal:  {signal.tolist()}")
    print(f"  Shifted: {shifted.tolist()}")
    print(f"  Pool(signal):  {out1.tolist()}")
    print(f"  Pool(shifted): {out2.tolist()}")
    print(f"  Same?:         {np.allclose(out1, out2)}")
    print(f"\n  Small shifts (< pool_size) produce identical pool outputs")


# -- 6. L2 pooling -------------------------------------------------------------
def l2_pooling_demo():
    print("\n=== L2 Pooling ===")
    print("  Output = sqrt(mean(x²)) over window — energy-based pooling")
    print("  Less common; used in some audio / frequency models")

    X = np.array([[1.,2.,3.,4.],[5.,6.,7.,8.],[1.,0.,2.,1.],[3.,4.,1.,2.]])
    out = l2_pool2d(X, pool_size=2, stride=2)
    print(f"\n  Input:\n{X}")
    print(f"  L2 Pool 2×2:\n{out.round(3)}")


# -- 7. Visualise pooling effect -----------------------------------------------
def visualise_pooling():
    rng = np.random.default_rng(7)
    X   = rng.standard_normal((16, 16))
    # Smooth with a Gaussian-like pattern
    for i in range(16):
        for j in range(16):
            X[i, j] += np.sin(i/3) * np.cos(j/3)

    max_out = max_pool2d(X, pool_size=2, stride=2)
    avg_out = avg_pool2d(X, pool_size=2, stride=2)
    max4    = max_pool2d(X, pool_size=4, stride=4)

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    vmin, vmax = X.min(), X.max()
    for ax, (title, img) in zip(axes, [("Original 16×16", X),
                                        ("Max Pool 2x2->8x8", max_out),
                                        ("Avg Pool 2x2->8x8", avg_out),
                                        ("Max Pool 4x4->4x4", max4)]):
        im = ax.imshow(img, cmap='viridis', vmin=vmin, vmax=vmax)
        ax.set_title(f"{title}\n{img.shape[0]}×{img.shape[1]}", fontsize=9)
        plt.colorbar(im, ax=ax)
    plt.suptitle("Pooling Operations", fontsize=13)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "pooling_comparison.png")
    plt.savefig(path, dpi=90); plt.close()
    print(f"\n  Pooling visualisation saved: {path}")


if __name__ == "__main__":
    max_pooling_demo()
    average_pooling_demo()
    global_pooling_demo()
    strided_vs_pooling()
    translation_invariance()
    l2_pooling_demo()
    visualise_pooling()
