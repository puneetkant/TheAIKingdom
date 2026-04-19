"""
Working Example: Generative Adversarial Networks (GANs)
Covers GAN theory, training dynamics, a numpy GAN on 1D data,
mode collapse, and major GAN variants.
"""
import numpy as np
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_gans")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def relu(z):    return np.maximum(0, z)
def leaky(z, a=0.2): return np.where(z > 0, z, a * z)
def sigmoid(z): return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
def dsigmoid(y): return y * (1 - y)


# -- 1. GAN theory -------------------------------------------------------------
def gan_theory():
    print("=== GAN Theory (Goodfellow et al. 2014) ===")
    print()
    print("  Minimax objective:")
    print("    min_G max_D  V(D, G) =")
    print("      E_{x~p_data}[log D(x)] + E_{z~p_z}[log(1 - D(G(z)))]")
    print()
    print("  Optimal discriminator (given fixed G):")
    print("    D*(x) = p_data(x) / (p_data(x) + p_G(x))")
    print()
    print("  At optimality:")
    print("    G* produces p_G = p_data")
    print("    D* = 0.5 everywhere (can't distinguish real from fake)")
    print("    V(D*, G*) = -log 4  (global minimum)")
    print()
    print("  Jensen-Shannon divergence:")
    print("    V(D*, G) = 2·JSD(p_data || p_G) - log 4")
    print("    JSD >= 0; zero iff p_G = p_data")
    print()

    # Check numerics
    v_opt = -np.log(4)
    print(f"  -log(4) = {v_opt:.4f}")
    d = 0.5   # D at optimum
    v_check = np.log(d) + np.log(1 - d)
    print(f"  log(0.5) + log(0.5) = {v_check:.4f} [OK]")


# -- 2. 1D GAN from scratch ----------------------------------------------------
class Generator:
    """z (latent_dim) -> x (1D)"""
    def __init__(self, latent, hidden, out, rng):
        s = 0.05
        self.W1 = rng.standard_normal((latent, hidden)) * s
        self.b1 = np.zeros(hidden)
        self.W2 = rng.standard_normal((hidden, hidden)) * s
        self.b2 = np.zeros(hidden)
        self.W3 = rng.standard_normal((hidden, out)) * s
        self.b3 = np.zeros(out)

    def forward(self, z):
        self.z  = z
        self.h1 = leaky(z  @ self.W1 + self.b1)
        self.h2 = leaky(self.h1 @ self.W2 + self.b2)
        return self.h2 @ self.W3 + self.b3   # linear output

    def backward(self, grad_out, lr):
        dW3 = self.h2.T @ grad_out; db3 = grad_out.sum(0)
        dh2 = grad_out @ self.W3.T * (self.h2 > 0).astype(float)
        dW2 = self.h1.T @ dh2; db2 = dh2.sum(0)
        dh1 = dh2 @ self.W2.T * (self.h1 > 0).astype(float)
        dW1 = self.z.T @ dh1; db1 = dh1.sum(0)
        for p, g in [(self.W3,dW3),(self.b3,db3),(self.W2,dW2),(self.b2,db2),
                     (self.W1,dW1),(self.b1,db1)]:
            p -= lr * np.clip(g, -1, 1)


class Discriminator:
    """x -> D(x) probability"""
    def __init__(self, in_dim, hidden, rng):
        s = 0.05
        self.W1 = rng.standard_normal((in_dim, hidden)) * s
        self.b1 = np.zeros(hidden)
        self.W2 = rng.standard_normal((hidden, 1)) * s
        self.b2 = np.zeros(1)

    def forward(self, x):
        self.x  = x
        self.h1 = leaky(x @ self.W1 + self.b1)
        logits  = self.h1 @ self.W2 + self.b2
        self.out = sigmoid(logits)
        return self.out

    def backward(self, d_out, lr):
        dW2 = self.h1.T @ d_out; db2 = d_out.sum(0)
        dh1 = (d_out @ self.W2.T) * (self.h1 > 0)
        dW1 = self.x.T @ dh1; db1 = dh1.sum(0)
        for p, g in [(self.W2,dW2),(self.b2,db2),(self.W1,dW1),(self.b1,db1)]:
            p -= lr * np.clip(g, -1, 1)


def train_gan():
    print("\n=== 1D GAN Demo (Gaussian target distribution) ===")
    rng = np.random.default_rng(42)
    # Target: mixture of 2 Gaussians
    def sample_real(n):
        c = rng.integers(2, size=n)
        return (c * rng.normal(2, 0.5, n) + (1-c) * rng.normal(-2, 0.5, n)).reshape(-1, 1)

    latent = 4; hidden = 32
    G = Generator(latent, hidden, 1, rng)
    D = Discriminator(1, hidden, rng)

    lr_d = 0.005; lr_g = 0.003; bs = 64; n_epochs = 400
    d_losses = []; g_losses = []

    for ep in range(n_epochs):
        # -- Train D --
        z   = rng.standard_normal((bs, latent))
        x_g = G.forward(z)
        x_r = sample_real(bs)
        d_r = D.forward(x_r); d_g = D.forward(x_g)
        # D loss: maximise log D(x_r) + log(1 - D(x_g))
        loss_D = -(np.log(d_r + 1e-8).mean() + np.log(1 - d_g + 1e-8).mean())
        # Gradient for real: -(1/D(x))
        grad_dr = -(1 / (d_r + 1e-8)) * dsigmoid(d_r) / bs
        D.backward(grad_dr, lr_d)
        D.forward(x_g)
        grad_dg = (1 / (1 - d_g + 1e-8)) * dsigmoid(d_g) / bs
        D.backward(grad_dg, lr_d)

        # -- Train G --
        z   = rng.standard_normal((bs, latent))
        x_g = G.forward(z)
        d_g2 = D.forward(x_g)
        loss_G = -np.log(d_g2 + 1e-8).mean()
        # dL/d_g2 = -1/D(G(z))
        grad_dg2 = -(1/(d_g2+1e-8)) * dsigmoid(d_g2) / bs
        # backprop through D to get grad w.r.t. x_g
        grad_xg = grad_dg2 @ D.W2.T * (D.h1 > 0) @ D.W1.T
        G.backward(grad_xg, lr_g)

        d_losses.append(loss_D); g_losses.append(loss_G)

    # Sample from G
    z_test  = rng.standard_normal((1000, latent))
    samples = G.forward(z_test)
    real    = sample_real(1000)

    print(f"  Epochs: {n_epochs}  Batch: {bs}")
    print(f"  D loss: {d_losses[0]:.4f} -> {d_losses[-1]:.4f}")
    print(f"  G loss: {g_losses[0]:.4f} -> {g_losses[-1]:.4f}")
    print(f"  Generated samples: mean={samples.mean():.3f}  std={samples.std():.3f}")
    print(f"  Real samples:      mean={real.mean():.3f}  std={real.std():.3f}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    axes[0].plot(d_losses, label="D"); axes[0].plot(g_losses, label="G")
    axes[0].legend(); axes[0].set_title("GAN Losses"); axes[0].set_xlabel("Epoch")
    axes[1].hist(real.flatten(), bins=40, alpha=0.5, density=True, label="Real")
    axes[1].hist(samples.flatten(), bins=40, alpha=0.5, density=True, label="Generated")
    axes[1].legend(); axes[1].set_title("Distribution Match")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "gan_1d.png")
    plt.savefig(path, dpi=90); plt.close()
    print(f"  Plot: {path}")


# -- 3. Mode collapse and training instability ---------------------------------
def training_challenges():
    print("\n=== GAN Training Challenges ===")
    problems = [
        ("Mode collapse",         "G maps all z to a few modes; D can't distinguish"),
        ("Vanishing gradient",    "D too strong -> log(1-D(G(z))) ~= 0; no grad for G"),
        ("Non-convergence",       "Oscillations; no guarantee of convergence"),
        ("Checkerboard artifacts","Transposed conv upsampling; use sub-pixel conv"),
        ("Training instability",  "Sensitive to hyperparams; needs careful tuning"),
    ]
    for p, d in problems:
        print(f"  {p:<22} {d}")

    print()
    print("  Fixes:")
    fixes = [
        ("WGAN / WGAN-GP",      "Wasserstein distance; gradient penalty; stable"),
        ("Spectral Norm",       "Normalise D weights by spectral norm"),
        ("Feature matching",    "G matches D's intermediate features"),
        ("Mini-batch disc.",    "D sees multiple samples; detects mode collapse"),
        ("Progressive growing", "Start at low res; grow both G and D (ProGAN)"),
        ("Label smoothing",     "Real labels = 0.9 not 1.0; prevents D overconfidence"),
        ("Instance noise",      "Add noise to D inputs; annealed over training"),
    ]
    for f, d in fixes:
        print(f"  {f:<22} {d}")


# -- 4. Wasserstein GAN -------------------------------------------------------
def wgan_overview():
    print("\n=== WGAN (Wasserstein GAN) ===")
    print("  Arjovsky et al. (2017)")
    print()
    print("  Problem with original GAN: JS divergence")
    print("    When p_data and p_G have disjoint support, JSD = log 2 (constant)")
    print("    -> Zero gradient for G")
    print()
    print("  Wasserstein distance (Earth Mover):")
    print("    W(p, q) = inf_{gammainPi} E_{(x,y)~gamma}[||x - y||]")
    print()
    print("  WGAN approximates W using critic (not discriminator):")
    print("    max_{fin1-Lipschitz} E_{x~p_data}[f(x)] - E_{z}[f(G(z))]")
    print()
    print("  Enforcing 1-Lipschitz:")
    print("    Weight clipping: |w| <= c  (original WGAN; crude)")
    print("    Gradient penalty: E[(||∇f(x)||2 - 1)²]  (WGAN-GP; preferred)")
    print()
    print("  Benefits:")
    print("    More stable training; meaningful loss (correlates with quality)")
    print("    No mode collapse issues; discriminator can be trained to optimality")


# -- 5. GAN variants -----------------------------------------------------------
def gan_variants():
    print("\n=== GAN Variants ===")
    variants = [
        ("DCGAN",      2015, "First deep CNN GAN; BatchNorm; stable training"),
        ("cGAN",       2014, "Class label as extra condition"),
        ("InfoGAN",    2016, "Maximise mutual info between z_c and G(z,c)"),
        ("CycleGAN",   2017, "Unpaired I2I; cycle-consistency loss"),
        ("Pix2Pix",    2017, "Paired I2I; U-Net G; PatchGAN D"),
        ("ProGAN",     2018, "Progressive growing; FFHQ 1024×1024 faces"),
        ("StyleGAN",   2019, "Style-based; AdaIN; disentangled latent"),
        ("StyleGAN2",  2020, "Removes artefacts; path length regularisation"),
        ("BigGAN",     2018, "Large-scale class-conditional; truncation trick"),
        ("VGAN/WGAN",  2017, "Wasserstein distance; stable training"),
        ("GauGAN/SPADE",2019,"Semantic image synthesis; normalisation layers"),
        ("EG3D",       2022, "3D-aware GAN; NeRF + StyleGAN2"),
    ]
    print(f"  {'Model':<14} {'Year'} {'Description'}")
    print(f"  {'-'*14} {'-'*4} {'-'*50}")
    for m, y, d in variants:
        print(f"  {m:<14} {y}  {d}")


if __name__ == "__main__":
    gan_theory()
    train_gan()
    training_challenges()
    wgan_overview()
    gan_variants()
