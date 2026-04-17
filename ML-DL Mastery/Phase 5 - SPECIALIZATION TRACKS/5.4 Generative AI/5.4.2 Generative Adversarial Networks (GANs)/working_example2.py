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
        # Generator: z -> x
        self.Wg = rng.normal(0, 0.1, (out_dim, z_dim))
        self.bg = np.zeros(out_dim)
        # Discriminator: x -> prob
        self.Wd = rng.normal(0, 0.1, (1, out_dim))
        self.bd = np.zeros(1)
        self.lr = lr; self.z_dim = z_dim

    def generate(self, z): return z @ self.Wg.T + self.bg
    def discriminate(self, x): return sigmoid(x @ self.Wd.T + self.bd)

    def d_step(self, x_real, z):
        x_fake = self.generate(z)
        d_real = self.discriminate(x_real)
        d_fake = self.discriminate(x_fake)
        # Maximize log D(real) + log(1-D(fake))
        loss = -np.mean(np.log(d_real + 1e-8) + np.log(1 - d_fake + 1e-8))
        # Grad for Wd, bd
        grad_real = -(1 - d_real) * x_real / len(x_real)
        grad_fake = d_fake * x_fake / len(z)
        self.Wd -= self.lr * (grad_real.T @ np.ones((len(x_real), 1)) / len(x_real) +
                               grad_fake.T @ np.ones((len(z), 1)) / len(z))
        return float(loss)

    def g_step(self, z):
        x_fake = self.generate(z)
        d_fake = self.discriminate(x_fake)
        loss = -np.mean(np.log(d_fake + 1e-8))
        # Rough gradient via finite differences
        eps_val = 1e-4
        for i in range(self.Wg.shape[0]):
            for j in range(self.Wg.shape[1]):
                self.Wg[i,j] += eps_val
                xf2 = self.generate(z)
                df2 = self.discriminate(xf2)
                l2 = -np.mean(np.log(df2 + 1e-8))
                self.Wg[i,j] -= eps_val
                self.Wg[i,j] -= self.lr * (l2 - loss) / eps_val
        return float(loss)

def demo():
    print("=== Linear GAN on 1-D Gaussian ===")
    real_mean, real_std = 3.0, 0.5
    gan = LinearGAN(z_dim=4, out_dim=1)
    d_losses, g_losses, gen_means = [], [], []
    for step in range(300):
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
    axes[1].set_title(f"Generator Mean → target {real_mean}")
    plt.tight_layout(); plt.savefig(OUTPUT / "gan_training.png"); plt.close()
    print("  Saved gan_training.png")

if __name__ == "__main__":
    demo()
