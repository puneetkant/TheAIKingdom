"""
Working Example: Weight Initialization
Covers zero init failure, random init, Xavier/Glorot init, He init,
LeCun init, orthogonal init, pre-trained transfer learning init,
and empirical comparison on a deep network.
"""
import numpy as np


def relu(z):    return np.maximum(0, z)
def sigmoid(z): return 1 / (1 + np.exp(-z.clip(-500, 500)))
def tanh(z):    return np.tanh(z)
def softmax(z):
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


# ── 1. Zero initialization failure ───────────────────────────────────────────
def zero_init_failure():
    print("=== Zero Initialization Failure (Symmetry Breaking) ===")
    print("  If all weights = 0, all neurons in a layer produce identical output")
    print("  Identical gradients → all neurons learn the same thing")
    print("  Network collapses to a single neuron effectively")
    print()

    X   = np.array([[1.0, 2.0, 3.0]])   # single sample
    W1  = np.zeros((3, 4))              # all zeros
    b1  = np.zeros(4)
    W2  = np.zeros((4, 2))
    b2  = np.zeros(2)

    Z1 = X @ W1 + b1
    A1 = relu(Z1)
    Z2 = A1 @ W2 + b2
    A2 = softmax(Z2)

    print(f"  Z1 (all zeros):  {Z1.ravel()}")
    print(f"  A1 (all zeros):  {A1.ravel()}")
    print(f"  Z2 (all zeros):  {Z2.ravel()}")
    print(f"  Output (uniform):{A2.ravel()}")
    print(f"  → Gradient ∂L/∂W1 = Xᵀ·δ1 will also be identical for all neurons")


# ── 2. Naive random init (too large/small) ───────────────────────────────────
def naive_random_init():
    print("\n=== Naive Random Initialization ===")
    print("  Small σ → vanishing activations/gradients")
    print("  Large σ → exploding activations/gradients")
    rng = np.random.default_rng(0)
    X   = rng.standard_normal((64, 100))

    for std in [0.001, 0.01, 0.1, 1.0, 10.0]:
        A = X.copy()
        for _ in range(10):
            W = rng.standard_normal((A.shape[1], A.shape[1])) * std
            A = relu(A @ W)
        print(f"  std={std:<6}: after 10 layers  mean|A|={np.abs(A).mean():.4e}  "
              f"std={A.std():.4e}  {'DEAD' if A.std()<1e-10 else ('EXPLODE' if A.std()>1e7 else 'OK')}")


# ── 3. Xavier / Glorot initialization ────────────────────────────────────────
def xavier_init():
    print("\n=== Xavier / Glorot Initialization (tanh/sigmoid) ===")
    print("  W ~ Uniform(-√(6/(n_in+n_out)), √(6/(n_in+n_out)))")
    print("  or W ~ N(0, 2/(n_in+n_out))")
    print("  Keeps variance of activations ~constant across layers")
    print("  Optimal for linear / tanh / sigmoid activations")

    rng = np.random.default_rng(1)
    X   = rng.standard_normal((64, 100))

    def xavier_std(n_in, n_out): return np.sqrt(2 / (n_in + n_out))

    A = X.copy()
    print(f"\n  Layer-by-layer activation stats (tanh, Xavier init):")
    for l in range(10):
        n_in = A.shape[1]
        W    = rng.standard_normal((n_in, n_in)) * xavier_std(n_in, n_in)
        Z    = A @ W
        A    = tanh(Z)
        if l in [0, 4, 9]:
            print(f"  Layer {l+1}: mean={A.mean():+.4f}  std={A.std():.4f}  "
                  f"dead={( np.abs(A)<0.01 ).mean()*100:.1f}%")


# ── 4. He initialization (ReLU) ───────────────────────────────────────────────
def he_init():
    print("\n=== He Initialization (ReLU networks) ===")
    print("  W ~ N(0, 2/n_in)    (Kaiming He 2015)")
    print("  Factor of 2 accounts for ReLU zeroing half the inputs")

    rng = np.random.default_rng(2)
    X   = rng.standard_normal((64, 100))

    def he_std(n_in): return np.sqrt(2 / n_in)

    A = X.copy()
    print(f"\n  Layer-by-layer activation stats (ReLU, He init):")
    for l in range(10):
        n_in = A.shape[1]
        W    = rng.standard_normal((n_in, n_in)) * he_std(n_in)
        Z    = A @ W
        A    = relu(Z)
        if l in [0, 4, 9]:
            print(f"  Layer {l+1}: mean={A.mean():+.4f}  std={A.std():.4f}  "
                  f"dead={(A==0).mean()*100:.1f}%")


# ── 5. LeCun initialization (SELU) ───────────────────────────────────────────
def lecun_init():
    print("\n=== LeCun Initialization (SELU) ===")
    print("  W ~ N(0, 1/n_in)")
    print("  Designed for SELU activation; achieves self-normalisation")

    rng = np.random.default_rng(3)
    X   = rng.standard_normal((64, 100))

    selu_lambda = 1.0507
    selu_alpha  = 1.6733
    def selu(z):
        return selu_lambda * np.where(z >= 0, z, selu_alpha*(np.exp(z.clip(-500,0))-1))

    A = X.copy()
    print(f"\n  Layer-by-layer activation stats (SELU, LeCun init):")
    for l in range(10):
        n_in = A.shape[1]
        W    = rng.standard_normal((n_in, n_in)) * np.sqrt(1 / n_in)
        A    = selu(A @ W)
        if l in [0, 4, 9]:
            print(f"  Layer {l+1}: mean={A.mean():+.4f}  std={A.std():.4f}  (should stay near 0,1)")


# ── 6. Orthogonal initialization ──────────────────────────────────────────────
def orthogonal_init():
    print("\n=== Orthogonal Initialization ===")
    print("  W = Q from QR decomposition of a random matrix")
    print("  Preserves gradient norm in linear networks; good for RNNs")

    rng = np.random.default_rng(4)
    n   = 50
    M   = rng.standard_normal((n, n))
    Q, _ = np.linalg.qr(M)

    # Check orthogonality: Q^T @ Q ≈ I
    diff = np.abs(Q.T @ Q - np.eye(n)).max()
    print(f"\n  n={n}x{n} orthogonal matrix")
    print(f"  |QᵀQ - I|_max = {diff:.2e}  (should be ~0)")
    print(f"  Singular values: min={np.linalg.svd(Q, compute_uv=False).min():.4f}  "
          f"max={np.linalg.svd(Q, compute_uv=False).max():.4f}  (all ≈1)")


# ── 7. Comparison summary ─────────────────────────────────────────────────────
def init_comparison():
    print("\n=== Initialization Comparison ===")
    print(f"  {'Method':<22} {'Formula':<30} {'Best for'}")
    rows = [
        ("Zeros",       "W = 0",                        "NEVER (symmetry breaking)"),
        ("Constant",    "W = c ≠ 0",                   "NEVER (same reason)"),
        ("Small random","W ~ N(0, 0.01)",               "Shallow nets only"),
        ("Xavier uniform","Uniform(±√(6/(n_in+n_out)))","tanh / sigmoid"),
        ("Xavier normal","N(0, 2/(n_in+n_out))",        "tanh / sigmoid"),
        ("He normal",   "N(0, 2/n_in)",                 "ReLU / Leaky ReLU"),
        ("He uniform",  "Uniform(±√(6/n_in))",          "ReLU / Leaky ReLU"),
        ("LeCun",       "N(0, 1/n_in)",                 "SELU"),
        ("Orthogonal",  "QR decomposition",             "RNNs, linear nets"),
        ("Pre-trained", "Transfer learning weights",    "Fine-tuning tasks"),
    ]
    for r in rows:
        print(f"  {r[0]:<22} {r[1]:<30} {r[2]}")


if __name__ == "__main__":
    zero_init_failure()
    naive_random_init()
    xavier_init()
    he_init()
    lecun_init()
    orthogonal_init()
    init_comparison()
