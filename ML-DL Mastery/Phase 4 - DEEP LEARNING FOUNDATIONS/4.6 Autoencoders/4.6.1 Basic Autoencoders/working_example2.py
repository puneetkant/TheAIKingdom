"""
Working Example 2: Basic Autoencoders — encoder-decoder, bottleneck, reconstruction
======================================================================================
Numpy MLP autoencoder on California Housing features: compress 8→3→8.

Run:  python working_example2.py
"""
from pathlib import Path
try:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.datasets import fetch_california_housing
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
except ImportError:
    raise SystemExit("pip install numpy matplotlib scikit-learn")

OUTPUT = Path(__file__).parent / "output"
OUTPUT.mkdir(exist_ok=True)

relu   = lambda x: np.maximum(0, x)
relu_d = lambda x: (x > 0).astype(float)

class Autoencoder:
    """3-layer autoencoder: 8→16→3→16→8."""
    def __init__(self, n_in=8, n_h=16, n_z=3, seed=42):
        rng = np.random.default_rng(seed)
        self.We1 = rng.standard_normal((n_in, n_h)) * np.sqrt(2/n_in);  self.be1 = np.zeros(n_h)
        self.We2 = rng.standard_normal((n_h,  n_z)) * np.sqrt(2/n_h);   self.be2 = np.zeros(n_z)
        self.Wd1 = rng.standard_normal((n_z,  n_h)) * np.sqrt(2/n_z);   self.bd1 = np.zeros(n_h)
        self.Wd2 = rng.standard_normal((n_h, n_in)) * np.sqrt(2/n_h);   self.bd2 = np.zeros(n_in)

    def encode(self, X):
        h = relu(X @ self.We1 + self.be1)
        return relu(h @ self.We2 + self.be2)

    def decode(self, Z):
        h = relu(Z @ self.Wd1 + self.bd1)
        return h @ self.Wd2 + self.bd2

    def forward(self, X):
        Z = self.encode(X); return self.decode(Z), Z

    def train(self, X, lr=0.005, epochs=200):
        losses = []
        n = len(X)
        for ep in range(epochs):
            # Forward
            h1 = relu(X @ self.We1 + self.be1)
            z  = relu(h1 @ self.We2 + self.be2)
            h2 = relu(z @ self.Wd1 + self.bd1)
            x_hat = h2 @ self.Wd2 + self.bd2
            loss = np.mean((x_hat - X)**2)
            losses.append(loss)
            # Backward (MSE gradient)
            dout = 2*(x_hat - X)/n
            self.Wd2 -= lr * h2.T @ dout;  self.bd2 -= lr * dout.sum(0)
            dh2 = (dout @ self.Wd2.T) * relu_d(h2)
            self.Wd1 -= lr * z.T @ dh2;    self.bd1 -= lr * dh2.sum(0)
            dz  = (dh2 @ self.Wd1.T) * relu_d(z)
            self.We2 -= lr * h1.T @ dz;    self.be2 -= lr * dz.sum(0)
            dh1 = (dz @ self.We2.T) * relu_d(h1)
            self.We1 -= lr * X.T @ dh1;    self.be1 -= lr * dh1.sum(0)
        return losses

def demo():
    print("=== Autoencoder on California Housing (8→3→8) ===")
    h = fetch_california_housing(); X = StandardScaler().fit_transform(h.data)
    X_tr, X_te = train_test_split(X, test_size=0.2, random_state=42)

    ae = Autoencoder(n_in=8, n_h=32, n_z=3)
    losses = ae.train(X_tr, lr=0.005, epochs=200)

    x_hat, _ = ae.forward(X_te)
    test_mse = np.mean((x_hat - X_te)**2)
    print(f"  Train MSE (final): {losses[-1]:.4f}")
    print(f"  Test  MSE:         {test_mse:.4f}")
    print(f"  Compression ratio: 8→3 ({3/8*100:.0f}% of original dims)")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(losses); axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("MSE Loss")
    axes[0].set_title("Autoencoder Training")
    axes[1].scatter(X_te[:200, 0], x_hat[:200, 0], alpha=0.4, s=8)
    axes[1].plot([-3,3],[-3,3],'r--'); axes[1].set_xlabel("True"); axes[1].set_ylabel("Reconstructed")
    axes[1].set_title("Feature 0: True vs Reconstructed")
    plt.tight_layout(); plt.savefig(OUTPUT / "autoencoder.png"); plt.close()
    print("  Saved autoencoder.png")

if __name__ == "__main__":
    demo()
