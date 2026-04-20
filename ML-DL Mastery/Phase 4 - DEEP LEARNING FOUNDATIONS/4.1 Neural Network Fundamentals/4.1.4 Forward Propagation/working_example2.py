"""
Working Example 2: Forward Propagation — numpy MLP from scratch
================================================================
2-layer MLP: matrix math for forward pass, layer outputs, predictions.

Run:  python working_example2.py
"""
from pathlib import Path
try:
    import numpy as np
    from sklearn.datasets import make_moons
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
except ImportError:
    raise SystemExit("pip install numpy scikit-learn")

def sigmoid(x): return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
def relu(x):    return np.maximum(0, x)

class MLP:
    """Simple 2-hidden-layer MLP (forward pass only)."""
    def __init__(self, layer_sizes, seed=42):
        rng = np.random.default_rng(seed)
        self.params = []
        for i in range(len(layer_sizes) - 1):
            fan_in, fan_out = layer_sizes[i], layer_sizes[i+1]
            W = rng.standard_normal((fan_in, fan_out)) * np.sqrt(2 / fan_in)
            b = np.zeros(fan_out)
            self.params.append((W, b))

    def forward(self, X, verbose=False):
        a = X
        for idx, (W, b) in enumerate(self.params):
            z = a @ W + b
            is_last = idx == len(self.params) - 1
            a = sigmoid(z) if is_last else relu(z)
            if verbose:
                print(f"  Layer {idx+1}: z.shape={z.shape}  "
                      f"act={'sigmoid' if is_last else 'relu'}  "
                      f"out.mean={a.mean():.4f}")
        return a

def demo_forward():
    print("=== Forward Pass (2-layer MLP) ===")
    X, y = make_moons(500, noise=0.2, random_state=42)
    X = StandardScaler().fit_transform(X)

    net = MLP([2, 8, 8, 1])
    out = net.forward(X[:4], verbose=True)
    print(f"\n  Sample outputs (raw): {out[:4, 0].round(4)}")
    print(f"  Predicted classes:    {(out[:4, 0] > 0.5).astype(int).tolist()}")
    print(f"  True classes:         {y[:4].tolist()}")

def demo_matrix_dimensions():
    print("\n=== Matrix Dimension Walkthrough ===")
    n, d = 32, 4   # batch=32, features=4
    layers = [d, 16, 8, 1]
    print(f"  Input:  ({n}, {d})")
    cur = n
    for i, (fin, fout) in enumerate(zip(layers[:-1], layers[1:])):
        print(f"  Layer {i+1}: W({fin},{fout})  z=X@W+b -> ({cur},{fout})")

def demo_batch_vs_single():
    """Compare batched vs sample-by-sample forward pass (vectorisation benefit)."""
    print("\n=== Batch vs Single-Sample Forward Pass ===")
    import time
    X, y = make_moons(2000, noise=0.2, random_state=42)
    X = StandardScaler().fit_transform(X)
    net = MLP([2, 64, 32, 1])

    t0 = time.perf_counter()
    _ = net.forward(X)   # batched
    t_batch = time.perf_counter() - t0

    t0 = time.perf_counter()
    for xi in X:
        _ = net.forward(xi.reshape(1, -1))
    t_single = time.perf_counter() - t0

    print(f"  Batched ({len(X)} samples): {t_batch*1000:.2f} ms")
    print(f"  Loop  ({len(X)} samples): {t_single*1000:.2f} ms")
    print(f"  Speedup: {t_single/t_batch:.1f}x")


def demo_depth_effect():
    """Show how layer depth affects activation statistics (vanishing/exploding risk)."""
    print("\n=== Depth Effect on Activation Statistics ===")
    np.random.seed(42)
    X_rand = np.random.randn(128, 16)
    for depth in [2, 5, 10, 20]:
        sizes = [16] + [32] * depth + [1]
        net   = MLP(sizes)
        a     = X_rand
        for W, b in net.params[:-1]:
            z = a @ W + b
            a = relu(z)
        mean_a = a.mean(); std_a = a.std()
        print(f"  depth={depth:>2}: final hidden mean={mean_a:.4f}  std={std_a:.4f}")


if __name__ == "__main__":
    demo_forward()
    demo_matrix_dimensions()
    demo_batch_vs_single()
    demo_depth_effect()
