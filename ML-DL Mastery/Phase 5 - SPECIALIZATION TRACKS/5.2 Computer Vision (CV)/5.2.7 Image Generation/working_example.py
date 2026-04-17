"""
Working Example: Image Generation
Covers VAE for image generation, GAN concepts, diffusion model intuition,
style transfer, and evaluation metrics (FID, IS).
"""
import numpy as np
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_generation")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def softmax(z):
    e = np.exp(z - z.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)

def relu(z): return np.maximum(0, z)
def sigmoid(z): return 1 / (1 + np.exp(-np.clip(z, -500, 500)))


# ── 1. VAE for Image Generation ───────────────────────────────────────────────
class ImageVAE:
    """Minimal VAE using flattened image pixels (numpy only)."""
    def __init__(self, input_dim=64, hidden=32, latent=8, rng=None):
        rng = rng or np.random.default_rng(0)
        s = 0.05
        # Encoder
        self.We1 = rng.standard_normal((input_dim, hidden)) * s
        self.be1 = np.zeros(hidden)
        self.Wmu  = rng.standard_normal((hidden, latent)) * s
        self.bmu  = np.zeros(latent)
        self.Wlv  = rng.standard_normal((hidden, latent)) * s
        self.blv  = np.zeros(latent)
        # Decoder
        self.Wd1 = rng.standard_normal((latent, hidden)) * s
        self.bd1 = np.zeros(hidden)
        self.Wd2 = rng.standard_normal((hidden, input_dim)) * s
        self.bd2 = np.zeros(input_dim)
        self.latent = latent

    def encode(self, x):
        h   = relu(x @ self.We1 + self.be1)
        mu  = h @ self.Wmu + self.bmu
        lv  = h @ self.Wlv + self.blv
        return mu, lv

    def reparametrize(self, mu, lv, rng):
        eps = rng.standard_normal(mu.shape)
        return mu + np.exp(0.5 * lv) * eps

    def decode(self, z):
        h = relu(z @ self.Wd1 + self.bd1)
        return sigmoid(h @ self.Wd2 + self.bd2)

    def loss(self, x, recon):
        recon_loss = -((x * np.log(recon + 1e-8) + (1-x)*np.log(1-recon+1e-8))).sum(1).mean()
        return recon_loss


def vae_image_demo():
    print("=== VAE for Image Generation ===")
    from sklearn.datasets import load_digits
    digits = load_digits()
    X = digits.data / 16.0   # 8×8 images, 64 pixels, [0,1]
    N, D = X.shape
    rng  = np.random.default_rng(0)

    vae = ImageVAE(input_dim=D, hidden=32, latent=4, rng=rng)
    lr  = 0.003; bs = 64

    # Quick training loop
    losses = []
    for ep in range(30):
        idx = rng.permutation(N); ep_l = 0
        for i in range(0, N, bs):
            xb  = X[idx[i:i+bs]]
            mu, lv = vae.encode(xb)
            z   = vae.reparametrize(mu, lv, rng)
            recon = vae.decode(z)
            l   = vae.loss(xb, recon)
            kl  = -0.5 * (1 + lv - mu**2 - np.exp(lv)).sum(1).mean()
            total = l + 0.1 * kl
            ep_l += total * len(xb)
            # Simplified gradient (output layer only for speed)
            dL = (recon - xb) / len(xb)
            dWd2 = relu(z @ vae.Wd1 + vae.bd1).T @ dL
            vae.Wd2 -= lr * np.clip(dWd2, -1, 1)
        losses.append(ep_l / N)

    # Generate from prior
    z_samples = rng.standard_normal((8, 4))
    generated  = vae.decode(z_samples)
    print(f"  Digits dataset: {N} images, {D}-dim")
    print(f"  Latent dim: {vae.latent}")
    print(f"  Training loss: {losses[0]:.4f} → {losses[-1]:.4f}")
    print(f"  Generated samples: {generated.shape}  range [{generated.min():.3f}, {generated.max():.3f}]")

    # Reconstruct some digits
    mu, lv = vae.encode(X[:4])
    z = vae.reparametrize(mu, lv, rng)
    recon = vae.decode(z)
    mse = ((X[:4] - recon)**2).mean()
    print(f"  Reconstruction MSE (4 samples): {mse:.6f}")

    # Visualise
    fig, axes = plt.subplots(2, 8, figsize=(14, 4))
    for j in range(8):
        axes[0,j].imshow(X[j].reshape(8,8), cmap="gray")
        axes[0,j].set_title(f"orig {digits.target[j]}", fontsize=7)
        axes[0,j].axis("off")
        axes[1,j].imshow(generated[j].reshape(8,8), cmap="gray")
        axes[1,j].set_title("gen", fontsize=7); axes[1,j].axis("off")
    plt.suptitle("Top: original  Bottom: VAE generated from N(0,I)", fontsize=9)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "vae_generation.png")
    plt.savefig(path, dpi=90); plt.close()
    print(f"  VAE generation plot: {path}")


# ── 2. GAN concepts ───────────────────────────────────────────────────────────
def gan_overview():
    print("\n=== Generative Adversarial Networks ===")
    print("  Goodfellow et al. (2014)")
    print()
    print("  Framework: minimax game between Generator G and Discriminator D")
    print("  min_G max_D  E[log D(x)] + E[log(1 - D(G(z)))]")
    print()
    print("  G: noise z → fake image   (tries to fool D)")
    print("  D: image → real/fake prob (tries to detect fakes)")
    print()
    print("  Training alternates:")
    print("    1. Update D: real images → label 1, fake images → label 0")
    print("    2. Update G: generate fakes, target D output → 1")
    print()
    print("  GAN variants:")
    variants = [
        ("DCGAN",       "Deep Convolutional GAN; batch norm; stable training"),
        ("cGAN",        "Conditional GAN; class label as extra input"),
        ("CycleGAN",    "Unpaired image-to-image translation; cycle consistency"),
        ("Pix2Pix",     "Paired image translation; L1 + adversarial loss"),
        ("ProGAN",      "Progressive growing; high-res face synthesis"),
        ("StyleGAN",    "Style-based generator; disentangled latent; FFHQ"),
        ("BigGAN",      "Large-scale class-conditional; ImageNet generation"),
        ("WGAN",        "Wasserstein distance; more stable; gradient penalty"),
    ]
    for v, d in variants:
        print(f"    {v:<12} {d}")


# ── 3. Diffusion models ───────────────────────────────────────────────────────
def diffusion_overview():
    print("\n=== Diffusion Models ===")
    print("  Forward process: gradually add Gaussian noise over T steps")
    print("    q(x_t | x_{t-1}) = N(x_t; √(1-β_t)·x_{t-1}, β_t·I)")
    print()
    print("  Reverse process: learn to denoise")
    print("    p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), σ_t²·I)")
    print()
    print("  Simplified training objective:")
    print("    L = E[||ε - ε_θ(x_t, t)||²]   (predict the noise)")
    print()

    # Simulate forward diffusion
    T = 10
    betas = np.linspace(0.001, 0.02, T)
    alphas = 1 - betas
    alpha_bars = np.cumprod(alphas)

    print("  Forward diffusion (noise schedule):")
    rng = np.random.default_rng(0)
    x0  = np.array([1.0, 0.5, -0.3, 0.7])   # fake "image"
    print(f"    x_0 (clean): {x0}")
    for t_idx in [0, 4, 9]:
        ab = alpha_bars[t_idx]
        eps = rng.standard_normal(x0.shape)
        xt  = np.sqrt(ab) * x0 + np.sqrt(1 - ab) * eps
        snr = ab / (1 - ab)
        print(f"    t={t_idx+1:>2}: x_t ≈ {xt.round(3)}  α̅={ab:.4f}  SNR={snr:.4f}")

    print()
    print("  Key models:")
    models = [
        ("DDPM",    "Denoising Diffusion Probabilistic Models (Ho 2020)"),
        ("DDIM",    "Deterministic sampling; 10-50× faster inference"),
        ("LDM",     "Latent Diffusion; compress to latent space first"),
        ("DALL-E 2","CLIP + diffusion decoder; text-to-image"),
        ("SD",      "Stable Diffusion; open-source LDM; 512×512"),
        ("SDXL",    "Stable Diffusion XL; 1024×1024 native"),
        ("Flux",    "Flow matching; DiT; state-of-art open t2i"),
        ("Sora",    "Spatio-temporal diffusion transformer; video generation"),
    ]
    for m, d in models:
        print(f"    {m:<12} {d}")


# ── 4. Style transfer ─────────────────────────────────────────────────────────
def style_transfer_overview():
    print("\n=== Neural Style Transfer ===")
    print("  Gatys et al. (2015): optimise input image to match")
    print("    Content:  feature activations of a photo")
    print("    Style:    Gram matrix of style image activations")
    print()
    print("  Loss:")
    print("    L = α·L_content + β·L_style")
    print("    L_content = ||F_content - F_generated||²  (deep layer)")
    print("    L_style   = Σ_l ||G_l^style - G_l^gen||²  (Gram matrices)")
    print()

    # Simulate Gram matrix
    rng = np.random.default_rng(0)
    F = rng.standard_normal((32, 10, 10))   # 32 filters, 10×10 spatial
    F_flat = F.reshape(32, -1)              # (32, 100)
    G = F_flat @ F_flat.T / 100            # Gram matrix (32, 32)
    print(f"  Gram matrix: F {F.shape} → G {G.shape}")
    print(f"    Gram diag (style energy per filter): mean={np.diag(G).mean():.3f}")
    print()
    print("  Fast style transfer: train a per-style feed-forward network")
    print("  AdaIN (Adaptive Instance Normalization): arbitrary style in real-time")


# ── 5. Generation metrics ────────────────────────────────────────────────────
def generation_metrics():
    print("\n=== Image Generation Evaluation Metrics ===")
    print("  FID (Fréchet Inception Distance):")
    print("    Lower is better (0 = perfect)")
    print("    Computes distance between real/fake feature distributions")
    print("    FID = ||μ_r - μ_f||² + Tr(Σ_r + Σ_f - 2·(Σ_r·Σ_f)^0.5)")
    print()

    # Toy FID simulation
    rng = np.random.default_rng(0)
    mu_real = np.array([0.0, 0.0]); sig_real = np.eye(2)
    mu_good = np.array([0.1, 0.1]); sig_good = np.array([[1.1, 0.1],[0.1, 1.1]])
    mu_bad  = np.array([2.0, 2.0]); sig_bad  = np.eye(2) * 3

    def toy_fid(m1, s1, m2, s2):
        diff = np.linalg.norm(m1 - m2)**2
        cov_sqrt = np.sqrt(np.abs(np.linalg.eigvals(s1 @ s2))).sum()
        return diff + np.trace(s1 + s2) - 2 * cov_sqrt

    print(f"  Toy FID (good model): {toy_fid(mu_real, sig_real, mu_good, sig_good):.4f}")
    print(f"  Toy FID (bad model):  {toy_fid(mu_real, sig_real, mu_bad,  sig_bad ):.4f}")
    print()
    print("  Other metrics:")
    metrics = [
        ("IS (Inception Score)", "High quality + diversity → high IS"),
        ("LPIPS",                "Perceptual similarity; learned patch distance"),
        ("SSIM",                 "Structural similarity; luminance/contrast/structure"),
        ("PSNR",                 "Peak SNR; 10·log10(MAX²/MSE); higher is better"),
        ("CLIP Score",           "Semantic alignment between image and text prompt"),
        ("Human eval",           "Gold standard; costly; MOS or preference studies"),
    ]
    for m, d in metrics:
        print(f"    {m:<22} {d}")


if __name__ == "__main__":
    vae_image_demo()
    gan_overview()
    diffusion_overview()
    style_transfer_overview()
    generation_metrics()
