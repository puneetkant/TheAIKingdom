"""
Working Example 2: Weight Initialization — Xavier, He, zero, random comparison
================================================================================
Effect of initialization on gradient flow and training convergence.

Run:  python working_example2.py
"""
from pathlib import Path
try:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.datasets import make_moons
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
except ImportError:
    raise SystemExit("pip install numpy matplotlib scikit-learn")

OUTPUT = Path(__file__).parent / "output"
OUTPUT.mkdir(exist_ok=True)

sigmoid  = lambda x: 1 / (1 + np.exp(-np.clip(x, -500, 500)))
relu     = lambda x: np.maximum(0, x)
relu_d   = lambda x: (x > 0).astype(float)

def init_weights(n_in, n_hidden, strategy, seed=42):
    rng = np.random.default_rng(seed)
    if strategy == "zeros":
        W1 = np.zeros((n_in, n_hidden)); W2 = np.zeros((n_hidden, 1))
    elif strategy == "random_large":
        W1 = rng.standard_normal((n_in, n_hidden)) * 10
        W2 = rng.standard_normal((n_hidden, 1)) * 10
    elif strategy == "xavier":
        lim = np.sqrt(6 / (n_in + n_hidden))
        W1 = rng.uniform(-lim, lim, (n_in, n_hidden))
        W2 = rng.uniform(-np.sqrt(6/(n_hidden+1)), np.sqrt(6/(n_hidden+1)), (n_hidden, 1))
    elif strategy == "he":
        W1 = rng.standard_normal((n_in, n_hidden)) * np.sqrt(2 / n_in)
        W2 = rng.standard_normal((n_hidden, 1)) * np.sqrt(2 / n_hidden)
    b1 = np.zeros(n_hidden); b2 = np.zeros(1)
    return W1, b1, W2, b2

def train(X, y, W1, b1, W2, b2, lr=0.05, epochs=200):
    losses = []
    for ep in range(epochs):
        z1 = X @ W1 + b1; a1 = relu(z1)
        z2 = a1 @ W2 + b2; a2 = sigmoid(z2)
        y2 = y.reshape(-1, 1)
        p  = np.clip(a2, 1e-7, 1-1e-7)
        loss = -np.mean(y2 * np.log(p) + (1-y2) * np.log(1-p))
        losses.append(loss)
        n  = len(y)
        dz2 = (a2 - y2) / n; dW2 = a1.T @ dz2; db2 = dz2.sum(0)
        dz1 = (dz2 @ W2.T) * relu_d(z1); dW1 = X.T @ dz1; db1 = dz1.sum(0)
        W2 -= lr * dW2; b2 -= lr * db2; W1 -= lr * dW1; b1 -= lr * db1
    return losses

def demo_init_comparison():
    print("=== Initialization Strategy Comparison ===")
    X, y = make_moons(600, noise=0.2, random_state=42)
    X = StandardScaler().fit_transform(X)
    X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    strategies = ["zeros", "random_large", "xavier", "he"]
    fig, ax = plt.subplots(figsize=(9, 5))

    for strat in strategies:
        W1, b1, W2, b2 = init_weights(2, 32, strat)
        losses = train(X_tr, y_tr, W1, b1, W2, b2, lr=0.05, epochs=200)
        final = losses[-1]
        print(f"  {strat:15s}: final loss={final:.4f}")
        if np.isfinite(final):
            ax.plot(losses, label=strat)

    ax.set_xlabel("Epoch"); ax.set_ylabel("BCE Loss"); ax.set_title("Weight Initialization")
    ax.legend(); ax.set_ylim(0, 2)
    plt.tight_layout(); plt.savefig(OUTPUT / "weight_init.png"); plt.close()
    print("  Saved weight_init.png")

def demo_activation_statistics():
    """Show how init strategy affects per-layer activation std across depth."""
    print("\n=== Activation Statistics Across Depth ===")
    rng = np.random.default_rng(0)
    X_rand = rng.standard_normal((256, 32))
    depth = 10
    for strat in ["random_large", "xavier", "he"]:
        x = X_rand.copy()
        stds = []
        for layer in range(depth):
            if strat == "random_large":
                W = rng.standard_normal((x.shape[1], 32)) * 5
            elif strat == "xavier":
                lim = np.sqrt(6 / (x.shape[1] + 32))
                W = rng.uniform(-lim, lim, (x.shape[1], 32))
            else:  # he
                W = rng.standard_normal((x.shape[1], 32)) * np.sqrt(2 / x.shape[1])
            b = np.zeros(32)
            x = relu(x @ W + b)
            stds.append(x.std())
        trend = "exploding" if stds[-1] > stds[0]*2 else ("vanishing" if stds[-1] < stds[0]*0.5 else "stable")
        print(f"  {strat:15s}: std layer1={stds[0]:.4f}  std layer10={stds[-1]:.4f}  [{trend}]")


def demo_orthogonal_init():
    """Orthogonal initialisation: preserves gradient norm through depth."""
    print("\n=== Orthogonal Initialisation ===")
    rng = np.random.default_rng(1)
    n = 64
    # Random matrix vs orthogonal
    R = rng.standard_normal((n, n))
    Q, _ = np.linalg.qr(R)   # orthogonal matrix
    eigvals_rand = np.abs(np.linalg.eigvals(R))
    eigvals_orth = np.abs(np.linalg.eigvals(Q))
    print(f"  Random matrix eigenvalue range: [{eigvals_rand.min():.3f}, {eigvals_rand.max():.3f}]")
    print(f"  Orthogonal matrix eigenvalue range: [{eigvals_orth.min():.3f}, {eigvals_orth.max():.3f}]")
    print("  Orthogonal: all eigenvalues = 1 -> gradient norms preserved across layers")
    # Forward pass std comparison
    x = rng.standard_normal((32, n))
    x_rand = x.copy(); x_orth = x.copy()
    for _ in range(10):
        x_rand = np.tanh(x_rand @ R / np.sqrt(n))
        x_orth = np.tanh(x_orth @ Q)
    print(f"  After 10 layers — Random std: {x_rand.std():.4f}  Orthogonal std: {x_orth.std():.4f}")


if __name__ == "__main__":
    demo_init_comparison()
    demo_activation_statistics()
    demo_orthogonal_init()
