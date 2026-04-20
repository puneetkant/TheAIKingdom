"""
Working Example 2: Pooling Layers — MaxPool, AvgPool, Global Pooling
======================================================================
Manual implementations and effect on feature maps.

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

def maxpool2d(x, size=2, stride=2):
    oh = (x.shape[0] - size) // stride + 1
    ow = (x.shape[1] - size) // stride + 1
    out = np.zeros((oh, ow))
    for i in range(oh):
        for j in range(ow):
            out[i, j] = x[i*stride:i*stride+size, j*stride:j*stride+size].max()
    return out

def avgpool2d(x, size=2, stride=2):
    oh = (x.shape[0] - size) // stride + 1
    ow = (x.shape[1] - size) // stride + 1
    out = np.zeros((oh, ow))
    for i in range(oh):
        for j in range(ow):
            out[i, j] = x[i*stride:i*stride+size, j*stride:j*stride+size].mean()
    return out

def global_avg_pool(x): return x.mean()
def global_max_pool(x): return x.max()

def demo():
    print("=== Pooling Operations ===")
    rng = np.random.default_rng(42)
    fm = rng.standard_normal((8, 8))  # fake feature map
    mx = maxpool2d(fm); av = avgpool2d(fm)
    print(f"  Input:    {fm.shape}")
    print(f"  MaxPool2: {mx.shape}")
    print(f"  AvgPool2: {av.shape}")
    print(f"  GlobalAvg: scalar={global_avg_pool(fm):.4f}")
    print(f"  GlobalMax: scalar={global_max_pool(fm):.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    axes[0].imshow(fm, cmap="viridis"); axes[0].set_title("Feature Map 8×8")
    axes[1].imshow(mx, cmap="viridis"); axes[1].set_title("MaxPool 4×4")
    axes[2].imshow(av, cmap="viridis"); axes[2].set_title("AvgPool 4×4")
    for ax in axes: ax.axis("off")
    plt.tight_layout(); plt.savefig(OUTPUT / "pooling.png"); plt.close()
    print("  Saved pooling.png")

def demo_max_vs_avg_pooling():
    print("\n=== Max vs Average Pooling (6x6 -> 3x3) ===")
    img = np.arange(36, dtype=float).reshape(6, 6)
    mx = maxpool2d(img, size=2, stride=2)
    av = avgpool2d(img, size=2, stride=2)
    print("  Input 6x6:")
    print(img.astype(int))
    print("  MaxPool 3x3:")
    print(mx.astype(int))
    print("  AvgPool 3x3:")
    print(av)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    axes[0].imshow(img, cmap="plasma"); axes[0].set_title("Input 6x6")
    axes[1].imshow(mx, cmap="plasma"); axes[1].set_title("MaxPool 3x3")
    axes[2].imshow(av, cmap="plasma"); axes[2].set_title("AvgPool 3x3")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(OUTPUT / "pooling_comparison.png")
    plt.close()
    print("  Saved pooling_comparison.png")


def demo_global_pooling():
    print("\n=== Global Average and Max Pooling ===")
    rng = np.random.default_rng(7)
    batch = rng.standard_normal((4, 6, 6, 8))
    gap = batch.mean(axis=(1, 2))
    gmp = batch.max(axis=(1, 2))
    print(f"  Input shape:  {batch.shape}")
    print(f"  GAP shape:    {gap.shape}")
    print(f"  GMP shape:    {gmp.shape}")
    print(f"  GAP sample (batch 0): {gap[0].round(3)}")
    print(f"  GMP sample (batch 0): {gmp[0].round(3)}")


def demo_translation_invariance():
    print("\n=== Translation Invariance: Max vs Avg Pooling ===")
    base = np.zeros((5, 5))
    base[1, 1] = 1.0
    shifted = np.zeros((5, 5))
    shifted[1, 2] = 1.0
    mp_base    = maxpool2d(base,    size=2, stride=2)
    mp_shifted = maxpool2d(shifted, size=2, stride=2)
    ap_base    = avgpool2d(base,    size=2, stride=2)
    ap_shifted = avgpool2d(shifted, size=2, stride=2)
    print("  Base pattern (5x5 with 1 at (1,1)):")
    print(base.astype(int))
    print("  Shifted pattern (5x5 with 1 at (1,2)):")
    print(shifted.astype(int))
    print("  MaxPool base:")
    print(mp_base)
    print("  MaxPool shifted:")
    print(mp_shifted)
    print(f"  MaxPool identical? {np.allclose(mp_base, mp_shifted)}  <- translation invariant")
    print("  AvgPool base:")
    print(ap_base)
    print("  AvgPool shifted:")
    print(ap_shifted)
    print(f"  AvgPool identical? {np.allclose(ap_base, ap_shifted)}  <- NOT invariant")


def demo_adaptive_pooling():
    print("\n=== Adaptive Average Pooling ===")

    def adaptive_avg_pool2d(x, target_h, target_w):
        ih, iw = x.shape
        out = np.zeros((target_h, target_w))
        for i in range(target_h):
            r0 = int(np.floor(i * ih / target_h))
            r1 = int(np.ceil((i + 1) * ih / target_h))
            for j in range(target_w):
                c0 = int(np.floor(j * iw / target_w))
                c1 = int(np.ceil((j + 1) * iw / target_w))
                out[i, j] = x[r0:r1, c0:c1].mean()
        return out

    rng = np.random.default_rng(13)
    tests = [(7, 7, 3, 3), (11, 11, 4, 4), (5, 5, 2, 2)]
    for ih, iw, th, tw in tests:
        x = rng.integers(0, 10, (ih, iw)).astype(float)
        out = adaptive_avg_pool2d(x, th, tw)
        print(f"  Input {ih}x{iw} -> Output {out.shape[0]}x{out.shape[1]}")
        print(f"    Input:\n{x.astype(int)}")
        print(f"    Output:\n{out.round(2)}")


if __name__ == "__main__":
    demo()
    demo_max_vs_avg_pooling()
    demo_global_pooling()
    demo_translation_invariance()
    demo_adaptive_pooling()
