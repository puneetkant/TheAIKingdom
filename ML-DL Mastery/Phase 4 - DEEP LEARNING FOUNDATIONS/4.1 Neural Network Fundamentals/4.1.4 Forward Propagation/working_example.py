"""
Working Example: Forward Propagation
Covers layer-by-layer computation, vectorised batch forward pass,
network architecture notation, computation graph, and a complete
from-scratch MLP forward pass for both regression and classification.
"""
import numpy as np


def sigmoid(z):   return 1 / (1 + np.exp(-z.clip(-500, 500)))
def relu(z):      return np.maximum(0, z)
def softmax(z):
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


# ── 1. Single layer forward pass ─────────────────────────────────────────────
def single_layer():
    print("=== Single Layer Forward Pass ===")
    print("  Z = X @ W + b    (linear transformation)")
    print("  A = f(Z)          (apply activation)")
    print()
    rng = np.random.default_rng(0)
    n, d_in, d_out = 4, 3, 2

    X = rng.standard_normal((n, d_in))
    W = rng.standard_normal((d_in, d_out)) * 0.1
    b = np.zeros(d_out)

    Z = X @ W + b
    A = sigmoid(Z)

    print(f"  X shape:  {X.shape}  (batch=4, features=3)")
    print(f"  W shape:  {W.shape}  (in=3, out=2)")
    print(f"  b shape:  {b.shape}  (out=2)")
    print(f"  Z = X@W+b shape: {Z.shape}")
    print(f"  A = σ(Z) shape:  {A.shape}")
    print()
    print(f"  Z[:2]:\n{Z[:2].round(4)}")
    print(f"  A[:2]:\n{A[:2].round(4)}")


# ── 2. Multi-layer MLP forward pass ──────────────────────────────────────────
def mlp_forward():
    print("\n=== MLP Forward Pass (from scratch) ===")
    print("  Architecture: 3 → 8 → 4 → 2 (binary output)")

    class MLP:
        def __init__(self, layer_sizes, seed=0):
            rng = np.random.default_rng(seed)
            self.params = {}
            for i, (d_in, d_out) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
                # He initialisation for ReLU layers
                scale = np.sqrt(2.0 / d_in)
                self.params[f"W{i+1}"] = rng.standard_normal((d_in, d_out)) * scale
                self.params[f"b{i+1}"] = np.zeros(d_out)
            self.n_layers = len(layer_sizes) - 1

        def forward(self, X, verbose=False):
            cache = {"A0": X}
            A = X
            for i in range(1, self.n_layers + 1):
                W = self.params[f"W{i}"]
                b = self.params[f"b{i}"]
                Z = A @ W + b
                if i < self.n_layers:
                    A = relu(Z)
                else:
                    A = sigmoid(Z)  # output layer
                cache[f"Z{i}"] = Z
                cache[f"A{i}"] = A
                if verbose:
                    print(f"  Layer {i}: Z={Z.shape}  A={A.shape}  act={'relu' if i<self.n_layers else 'sigmoid'}")
            return A, cache

    rng = np.random.default_rng(1)
    X   = rng.standard_normal((5, 3))
    net = MLP([3, 8, 4, 2])
    out, cache = net.forward(X, verbose=True)
    print(f"\n  Input X shape:  {X.shape}")
    print(f"  Output A shape: {out.shape}")
    print(f"  Output (probs):\n{out.round(4)}")


# ── 3. Vectorised batch computation ──────────────────────────────────────────
def batch_vs_single():
    print("\n=== Vectorised vs Loop Forward Pass ===")
    rng = np.random.default_rng(2)
    n, d_in, d_out = 1000, 50, 20
    X = rng.standard_normal((n, d_in))
    W = rng.standard_normal((d_in, d_out)) * 0.1
    b = np.zeros(d_out)

    import time

    # Loop
    t0 = time.perf_counter()
    Z_loop = np.zeros((n, d_out))
    for i in range(n):
        Z_loop[i] = X[i] @ W + b
    t_loop = time.perf_counter() - t0

    # Vectorised
    t0 = time.perf_counter()
    Z_vec = X @ W + b
    t_vec = time.perf_counter() - t0

    print(f"  n={n}  d_in={d_in}  d_out={d_out}")
    print(f"  Loop time:       {t_loop*1000:.2f}ms")
    print(f"  Vectorised time: {t_vec*1000:.2f}ms")
    print(f"  Speedup:         {t_loop/max(t_vec,1e-9):.1f}×")
    print(f"  Max abs diff:    {np.abs(Z_loop - Z_vec).max():.2e}  (should be ~0)")


# ── 4. Computation graph ─────────────────────────────────────────────────────
def computation_graph():
    print("\n=== Computation Graph ===")
    print("  Forward pass builds a directed acyclic graph (DAG)")
    print("  Each node = operation; edges = data flow")
    print()
    print("  Example: 2-layer net, one sample")
    print()
    print("  x ─┐")
    print("      ├─ Z1 = W1·x + b1 ─→ A1 = relu(Z1) ─┐")
    print("  W1 ─┘                                      ├─ Z2 = W2·A1+b2 ─→ A2 = σ(Z2) ─→ L(A2,y)")
    print("  b1 ─┘                                  W2 ─┘")
    print("                                         b2 ─┘")
    print()
    print("  Autograd frameworks (PyTorch, JAX) record this graph")
    print("  Backward pass traverses it in reverse to compute gradients")


# ── 5. Complete classification forward pass ────────────────────────────────────
def classification_example():
    print("\n=== Classification Example: Iris (3-class) ===")
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    iris = load_iris()
    X, y = iris.data.astype(float), iris.target
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)

    rng  = np.random.default_rng(3)
    d_in = X_tr.shape[1]  # 4 features
    K    = 3               # classes

    # 4 → 8 → K architecture
    W1 = rng.standard_normal((d_in, 8)) * np.sqrt(2/d_in)
    b1 = np.zeros(8)
    W2 = rng.standard_normal((8, K)) * np.sqrt(2/8)
    b2 = np.zeros(K)

    # Forward
    Z1 = X_te @ W1 + b1
    A1 = relu(Z1)
    Z2 = A1 @ W2 + b2
    A2 = softmax(Z2)

    y_pred = A2.argmax(axis=1)
    acc    = (y_pred == y_te).mean()

    print(f"  Input  (n={X_te.shape[0]}, d=4)")
    print(f"  Layer1 Z1:{Z1.shape}  A1:{A1.shape}")
    print(f"  Layer2 Z2:{Z2.shape}  A2:{A2.shape}")
    print(f"\n  Sample predictions (first 5):")
    print(f"  {'True':<8} {'Pred':<8} {'Probs'}")
    for i in range(5):
        print(f"  {y_te[i]:<8} {y_pred[i]:<8} {A2[i].round(3)}")
    print(f"\n  Accuracy (random weights): {acc:.4f}  (expect ~0.33 random chance)")


if __name__ == "__main__":
    single_layer()
    mlp_forward()
    batch_vs_single()
    computation_graph()
    classification_example()
