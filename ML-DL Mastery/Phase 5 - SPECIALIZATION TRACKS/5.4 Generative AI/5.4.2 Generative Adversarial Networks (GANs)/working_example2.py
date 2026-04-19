"""
Working Example 2: GANs — discriminator/generator loss and training dynamics
=============================================================================
Trains a linear GAN on 1-D Gaussian data. Shows mode collapse vs stable training.

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

def sigmoid(x): return 1 / (1 + np.exp(-np.clip(x, -30, 30)))

class LinearGAN:
    def __init__(self, z_dim=4, out_dim=1, lr=0.01, seed=0):
        rng = np.random.default_rng(seed)
        # Generator: z -> x  (learns to shift z toward real distribution)
        self.Wg = rng.normal(0, 0.1, (z_dim, out_dim))
        self.bg = np.zeros(out_dim)
        # Discriminator: x -> logit (single scalar)
        self.Wd = rng.normal(0, 0.1, (out_dim, 1))
        self.bd = np.zeros(1)
        self.lr = lr; self.z_dim = z_dim

    def generate(self, z): return z @ self.Wg + self.bg    # (n, out_dim)
    def discriminate(self, x): return sigmoid(x @ self.Wd + self.bd)  # (n, 1)

    def d_step(self, x_real, z):
        """Maximize E[log D(x_real)] + E[log(1-D(G(z)))]."""
        x_fake = self.generate(z)
        d_real = self.discriminate(x_real)   # (n, 1)
        d_fake = self.discriminate(x_fake)   # (n, 1)
        loss = -np.mean(np.log(d_real + 1e-8) + np.log(1 - d_fake + 1e-8))
        # Analytic gradients for Wd and bd
        # dL/d(logit_real) = -(1 - D_real),  d(logit)/dWd = x_real
        grad_real = -(1 - d_real) * x_real   # (n, out_dim)
        # dL/d(logit_fake) = D_fake,          d(logit)/dWd = x_fake
        grad_fake = d_fake * x_fake           # (n, out_dim)
        dWd = (grad_real + grad_fake).mean(axis=0, keepdims=True).T  # (out_dim, 1)
        dbd = ((-(1 - d_real) + d_fake)).mean(axis=0)
        self.Wd -= self.lr * dWd
        self.bd -= self.lr * dbd
        return float(loss)

    def g_step(self, z):
        """Minimize E[-log D(G(z))] (non-saturating generator loss)."""
        x_fake = self.generate(z)
        d_fake = self.discriminate(x_fake)   # (n, 1)
        loss = -np.mean(np.log(d_fake + 1e-8))
        # dL/d(logit_fake) = -(1 - D_fake),  d(logit)/dx_fake = Wd.T,  dx_fake/dWg = z.T
        delta = -(1 - d_fake)                # (n, 1): gradient of loss wrt logit
        # grad wrt x_fake: delta * Wd.T  -> shape (n, out_dim)
        grad_xfake = delta @ self.Wd.T       # (n, out_dim)
        # grad wrt Wg: z.T @ grad_xfake / n
        dWg = z.T @ grad_xfake / len(z)     # (z_dim, out_dim)
        dbg = grad_xfake.mean(axis=0)
        self.Wg -= self.lr * dWg
        self.bg -= self.lr * dbg
        return float(loss)

def demo():
    print("=== Linear GAN on 1-D Gaussian ===")
    real_mean, real_std = 3.0, 0.5
    gan = LinearGAN(z_dim=4, out_dim=1, lr=0.05, seed=0)
    d_losses, g_losses, gen_means = [], [], []
    for step in range(500):
        x_real = np.random.normal(real_mean, real_std, (32, 1))
        z = np.random.randn(32, 4)
        d_l = gan.d_step(x_real, z)
        z2  = np.random.randn(32, 4)
        g_l = gan.g_step(z2)
        d_losses.append(d_l); g_losses.append(g_l)
        gen_means.append(float(gan.generate(np.random.randn(64, 4)).mean()))
    print(f"  Real distribution: N({real_mean}, {real_std})")
    print(f"  Final generator mean: {gen_means[-1]:.3f}")
    print(f"  Final D loss: {d_losses[-1]:.4f} | G loss: {g_losses[-1]:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    axes[0].plot(d_losses, label="D loss"); axes[0].plot(g_losses, label="G loss")
    axes[0].legend(); axes[0].set_title("GAN Training Losses")
    axes[1].plot(gen_means, color="orange"); axes[1].axhline(real_mean, ls="--", color="gray")
    axes[1].set_title(f"Generator Mean -> target {real_mean}")
    plt.tight_layout(); plt.savefig(OUTPUT / "gan_training.png"); plt.close()
    print("  Saved gan_training.png")

if __name__ == "__main__":
    demo()
