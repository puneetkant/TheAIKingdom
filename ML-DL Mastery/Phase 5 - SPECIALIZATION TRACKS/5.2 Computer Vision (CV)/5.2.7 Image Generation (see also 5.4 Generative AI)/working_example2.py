"""
Working Example 2: Image Generation — VAE latent space sampling (numpy)
=========================================================================
Trains a small VAE on digits and samples new images from latent space.

Run:  python working_example2.py
"""
from pathlib import Path
try:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.datasets import load_digits
    from sklearn.preprocessing import MinMaxScaler
except ImportError:
    raise SystemExit("pip install numpy matplotlib scikit-learn")

OUTPUT = Path(__file__).parent / "output"
OUTPUT.mkdir(exist_ok=True)

relu   = lambda x: np.maximum(0, x)
relu_d = lambda x: (x > 0).astype(float)
sigmoid = lambda x: 1 / (1 + np.exp(-np.clip(x, -50, 50)))

def train_vae_sketch(X, n_z=2, n_h=32, lr=0.002, epochs=150, seed=42):
    """Simplified VAE: encoder → mu/logvar → z → decoder (MSE + KL)."""
    rng = np.random.default_rng(seed)
    d = X.shape[1]
    We = rng.standard_normal((d, n_h)) * 0.05; be = np.zeros(n_h)
    Wmu = rng.standard_normal((n_h, n_z)) * 0.05; bmu = np.zeros(n_z)
    Wlv = rng.standard_normal((n_h, n_z)) * 0.05; blv = np.zeros(n_z)
    Wd1 = rng.standard_normal((n_z, n_h)) * 0.05; bd1 = np.zeros(n_h)
    Wd2 = rng.standard_normal((n_h, d)) * 0.05; bd2 = np.zeros(d)
    losses = []
    for ep in range(epochs):
        h = relu(X @ We + be)
        mu = h @ Wmu + bmu
        lv = h @ Wlv + blv
        eps = rng.standard_normal(mu.shape)
        z = mu + eps * np.exp(0.5 * lv)
        h2 = relu(z @ Wd1 + bd1)
        xh = sigmoid(h2 @ Wd2 + bd2)
        recon = np.mean((xh - X) ** 2)
        kl = -0.5 * np.mean(1 + lv - mu**2 - np.exp(lv))
        losses.append(recon + kl)
        # Simplified backprop through output only
        n = len(X)
        dout = 2*(xh-X)*xh*(1-xh)/n
        Wd2 -= lr * (h2.T @ dout); bd2 -= lr * dout.sum(0)
        dh2 = (dout @ Wd2.T) * relu_d(h2)
        Wd1 -= lr * (z.T @ dh2); bd1 -= lr * dh2.sum(0)
    return (We,be,Wmu,bmu,Wlv,blv,Wd1,bd1,Wd2,bd2), mu, losses

def demo():
    print("=== VAE Image Generation (Digits) ===")
    digits = load_digits()
    X = MinMaxScaler().fit_transform(digits.data)

    params, mu, losses = train_vae_sketch(X, n_z=2, epochs=100)
    We,be,Wmu,bmu,Wlv,blv,Wd1,bd1,Wd2,bd2 = params

    # Sample from prior N(0,1) and decode
    rng = np.random.default_rng(99)
    z_sample = rng.standard_normal((16, 2))
    h2 = relu(z_sample @ Wd1 + bd1)
    xh = sigmoid(h2 @ Wd2 + bd2)
    samples = xh.reshape(-1, 8, 8)

    fig, axes = plt.subplots(2, 8, figsize=(12, 3))
    for ax, s in zip(axes.flat, samples):
        ax.imshow(s, cmap="gray"); ax.axis("off")
    plt.suptitle("VAE Samples (z ~ N(0,I))")
    plt.tight_layout(); plt.savefig(OUTPUT / "vae_samples.png"); plt.close()
    print("  Saved vae_samples.png")
    print(f"  Final ELBO loss: {losses[-1]:.4f}")

if __name__ == "__main__":
    demo()
