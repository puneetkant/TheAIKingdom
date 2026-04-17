"""
Working Example 2: Backpropagation — numpy MLP with full forward + backward pass
==================================================================================
Chain rule, gradients w.r.t. W and b, gradient descent training loop.

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

def sigmoid(x): return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
def sigmoid_d(s): return s * (1 - s)   # s = sigmoid(x)
def relu(x):    return np.maximum(0, x)
def relu_d(x):  return (x > 0).astype(float)

class TwoLayerNet:
    """Input → Hidden(ReLU) → Output(Sigmoid), BCE loss."""
    def __init__(self, n_in, n_hidden, lr=0.01, seed=42):
        rng = np.random.default_rng(seed)
        self.lr = lr
        self.W1 = rng.standard_normal((n_in, n_hidden)) * np.sqrt(2 / n_in)
        self.b1 = np.zeros(n_hidden)
        self.W2 = rng.standard_normal((n_hidden, 1)) * np.sqrt(2 / n_hidden)
        self.b2 = np.zeros(1)

    def forward(self, X):
        self.X = X
        self.z1 = X @ self.W1 + self.b1
        self.a1 = relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = sigmoid(self.z2)
        return self.a2

    def backward(self, y):
        n = len(y)
        y = y.reshape(-1, 1)
        # Output layer gradient (BCE + sigmoid)
        dz2 = (self.a2 - y) / n
        dW2 = self.a1.T @ dz2
        db2 = dz2.sum(axis=0)
        # Hidden layer gradient
        da1 = dz2 @ self.W2.T
        dz1 = da1 * relu_d(self.z1)
        dW1 = self.X.T @ dz1
        db1 = dz1.sum(axis=0)
        # Update
        self.W2 -= self.lr * dW2; self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1; self.b1 -= self.lr * db1

    def loss(self, y):
        y = y.reshape(-1, 1)
        p = np.clip(self.a2, 1e-7, 1-1e-7)
        return -np.mean(y * np.log(p) + (1-y) * np.log(1-p))

def demo_backprop():
    print("=== Backpropagation Training (make_moons) ===")
    X, y = make_moons(800, noise=0.2, random_state=42)
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    net = TwoLayerNet(n_in=2, n_hidden=32, lr=0.05)
    losses = []
    for epoch in range(300):
        net.forward(X_train)
        net.backward(y_train)
        if epoch % 50 == 0:
            l = net.loss(y_train)
            losses.append(l)
            preds = (net.a2[:, 0] > 0.5).astype(int)
            acc = (preds == y_train).mean()
            print(f"  Epoch {epoch:4d}: loss={l:.4f}  train_acc={acc:.4f}")

    # Test accuracy
    net.forward(X_test)
    test_acc = ((net.a2[:, 0] > 0.5).astype(int) == y_test).mean()
    print(f"\n  Test accuracy: {test_acc:.4f}")

    # Loss plot
    plt.plot(np.arange(0, 300, 50), losses, marker="o")
    plt.xlabel("Epoch"); plt.ylabel("BCE Loss"); plt.title("Training Loss")
    plt.tight_layout(); plt.savefig(OUTPUT / "backprop_loss.png"); plt.close()
    print("  Saved backprop_loss.png")

if __name__ == "__main__":
    demo_backprop()
