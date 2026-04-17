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

if __name__ == "__main__":
    demo()
