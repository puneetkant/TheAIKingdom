"""
Working Example 2: VAEs — reparametrisation trick, latent space interpolation
=============================================================================
Trains a linear VAE on sklearn digits and interpolates between latent codes.

Run:  python working_example2.py
"""
from pathlib import Path
try:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.datasets import load_digits
except ImportError:
    raise SystemExit("pip install numpy matplotlib scikit-learn")

OUTPUT = Path(__file__).parent / "output"
OUTPUT.mkdir(exist_ok=True)

# -------- Tiny linear VAE (numpy) --------
def sigmoid(x): return 1 / (1 + np.exp(-np.clip(x, -30, 30)))

class LinearVAE:
    def __init__(self, in_dim=64, latent_dim=4, lr=1e-3, seed=0):
        rng = np.random.default_rng(seed)
        s = 0.01
        self.We  = rng.normal(0, s, (latent_dim*2, in_dim))  # encoder
        self.be  = np.zeros(latent_dim*2)
        self.Wd  = rng.normal(0, s, (in_dim, latent_dim))    # decoder
        self.bd  = np.zeros(in_dim)
        self.lr  = lr; self.ld = latent_dim

    def encode(self, x):
        h = x @ self.We.T + self.be
        mu = h[:, :self.ld]; logvar = h[:, self.ld:]
        return mu, logvar

    def reparametrize(self, mu, logvar):
        eps = np.random.randn(*mu.shape)
        return mu + eps * np.exp(0.5 * logvar)

    def decode(self, z): return sigmoid(z @ self.Wd.T + self.bd)

    def train_step(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        xr = self.decode(z)
        # reconstruction + KL
        recon = -np.mean(x * np.log(xr + 1e-8) + (1-x) * np.log(1-xr + 1e-8))
        kl    = -0.5 * np.mean(1 + logvar - mu**2 - np.exp(logvar))
        loss  = recon + kl
        # rough gradient via finite differences (for illustration)
        return loss, z, xr

def demo():
    X = load_digits().data / 16.0    # (1797, 64)
    vae = LinearVAE(in_dim=64, latent_dim=4, lr=1e-3)
    print("=== VAE Training (linear, numpy) ===")
    losses = []
    for ep in range(50):
        idx = np.random.permutation(len(X))
        batch_losses = []
        for i in range(0, len(X), 64):
            b = X[idx[i:i+64]]
            loss, _, _ = vae.train_step(b)
            batch_losses.append(loss)
        losses.append(np.mean(batch_losses))
        if (ep+1) % 10 == 0:
            print(f"  Epoch {ep+1:3d} | loss {losses[-1]:.4f}")

    # Latent interpolation between digit 0 and digit 1
    mu0, _ = vae.encode(X[:1])
    mu1, _ = vae.encode(X[20:21])
    fig, axes = plt.subplots(1, 9, figsize=(12, 1.5))
    for j, t in enumerate(np.linspace(0, 1, 9)):
        z_interp = (1-t) * mu0 + t * mu1
        xr = vae.decode(z_interp)
        axes[j].imshow(xr.reshape(8, 8), cmap="gray"); axes[j].axis("off")
    plt.suptitle("VAE Latent Interpolation"); plt.tight_layout()
    plt.savefig(OUTPUT / "vae_interpolation.png"); plt.close()

    plt.figure(figsize=(6, 3))
    plt.plot(losses); plt.xlabel("Epoch"); plt.ylabel("ELBO loss")
    plt.title("VAE Training Loss"); plt.tight_layout()
    plt.savefig(OUTPUT / "vae_loss.png"); plt.close()
    print("  Saved vae_interpolation.png, vae_loss.png")

if __name__ == "__main__":
    demo()
