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

if __name__ == "__main__":
    demo_init_comparison()
