"""
Working Example: Diffusion Models
Covers DDPM forward/reverse process, noise schedules,
score matching, DDIM sampling, and major models.
"""
import numpy as np
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_diffusion")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── 1. Forward diffusion process ──────────────────────────────────────────────
def forward_process():
    print("=== DDPM Forward Process ===")
    print("  Ho et al. (2020)")
    print()
    print("  Gradually add Gaussian noise over T steps:")
    print("    q(x_t | x_{t-1}) = N(x_t; √(1-β_t)·x_{t-1}, β_t·I)")
    print()
    print("  Closed form (from x_0 directly):")
    print("    ᾱ_t = Π_{s=1}^t (1-β_s)   (cumulative product)")
    print("    x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε,   ε ~ N(0,I)")
    print()

    T    = 1000
    beta_min, beta_max = 1e-4, 0.02
    betas     = np.linspace(beta_min, beta_max, T)
    alphas    = 1 - betas
    alpha_bars = np.cumprod(alphas)

    # Key schedule stats
    print(f"  Noise schedule: T={T}, β ∈ [{beta_min:.4f}, {beta_max:.4f}]")
    for t_idx in [0, 99, 249, 499, 749, 999]:
        ab  = alpha_bars[t_idx]
        snr = ab / (1 - ab)
        print(f"    t={t_idx+1:>4}: ᾱ={ab:.6f}  signal={np.sqrt(ab):.4f}  noise={np.sqrt(1-ab):.4f}  SNR={snr:.4f}")

    # Simulate forward on synthetic point
    x0  = np.array([1.0, 0.0, -0.5, 0.3])
    rng = np.random.default_rng(0)
    print()
    print(f"  Forward diffusion of x_0 = {x0}:")
    for t_idx in [0, 100, 500, 999]:
        ab  = alpha_bars[t_idx]
        eps = rng.standard_normal(x0.shape)
        xt  = np.sqrt(ab) * x0 + np.sqrt(1-ab) * eps
        print(f"    t={t_idx+1:>4}: x_t = {np.round(xt, 3)}  ||x_t||={np.linalg.norm(xt):.3f}")

    # Plot noise schedule
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    t_vals = np.arange(1, T+1)
    axes[0].plot(t_vals, alpha_bars, label="ᾱ_t (signal²)")
    axes[0].plot(t_vals, 1-alpha_bars, label="1-ᾱ_t (noise²)")
    axes[0].legend(); axes[0].set_xlabel("t"); axes[0].set_title("Linear Schedule")
    # Cosine schedule
    s = 0.008
    alpha_bar_cos = np.cos(((np.arange(T)/T + s)/(1+s)) * np.pi/2)**2
    alpha_bar_cos /= alpha_bar_cos[0]
    axes[1].plot(t_vals, alpha_bar_cos, label="Cosine ᾱ_t")
    axes[1].plot(t_vals, alpha_bars,    label="Linear ᾱ_t")
    axes[1].legend(); axes[1].set_xlabel("t"); axes[1].set_title("Schedule Comparison")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "noise_schedule.png")
    plt.savefig(path, dpi=90); plt.close()
    print(f"\n  Noise schedule plot: {path}")


# ── 2. Reverse process / denoising ────────────────────────────────────────────
def reverse_process():
    print("\n=== Reverse Process (Denoising) ===")
    print("  Goal: learn p_θ(x_{t-1}|x_t) to reverse forward process")
    print()
    print("  Parameterisation (noise prediction):")
    print("    ε_θ(x_t, t)  ← predicted noise")
    print()
    print("  Training objective (simplified ELBO):")
    print("    L_simple = E_{t,x_0,ε}[||ε - ε_θ(x_t, t)||²]")
    print()
    print("  Reverse sampling step (DDPM):")
    print("    x_{t-1} = (1/√α_t) · (x_t - β_t/√(1-ᾱ_t) · ε_θ) + σ_t·z")
    print("    σ_t = √β_t  or  √(β̃_t)  where β̃_t = (1-ᾱ_{t-1})/(1-ᾱ_t)·β_t")
    print()

    # Simulate reverse step
    T = 1000
    betas = np.linspace(1e-4, 0.02, T)
    alphas = 1 - betas
    alpha_bars = np.cumprod(alphas)
    rng = np.random.default_rng(1)

    x_T = rng.standard_normal(4)
    print(f"  Starting from pure noise x_T = {x_T.round(3)}")

    # True noise was 0 (x_0 = 0 for this toy example)
    x_curr = x_T.copy()
    for t in range(T-1, T-6, -1):
        ab  = alpha_bars[t]
        ab1 = alpha_bars[t-1] if t > 0 else 1.0
        a   = alphas[t]; b = betas[t]
        eps_pred = x_curr / np.sqrt(1 - ab)  # trivial denoiser (x_0=0 case)
        mu = (1/np.sqrt(a)) * (x_curr - b/np.sqrt(1-ab) * eps_pred)
        sigma = np.sqrt(b)
        x_curr = mu + sigma * rng.standard_normal(4)

    print(f"  After 5 reverse steps: x ≈ {x_curr.round(3)} (should approach 0)")


# ── 3. DDIM (fast sampling) ───────────────────────────────────────────────────
def ddim_overview():
    print("\n=== DDIM (Denoising Diffusion Implicit Models) ===")
    print("  Song et al. (2020)")
    print()
    print("  Non-Markovian forward process → deterministic reverse ODE")
    print()
    print("  DDIM update (η=0: deterministic):")
    print("    x_{t-1} = √ᾱ_{t-1} · (x_t - √(1-ᾱ_t)·ε_θ) / √ᾱ_t")
    print("             + √(1-ᾱ_{t-1} - σ_t²) · ε_θ  +  σ_t·ε")
    print("    σ_t = η · √((1-ᾱ_{t-1})/(1-ᾱ_t)) · √(1 - ᾱ_t/ᾱ_{t-1})")
    print()
    print("  Speed advantage:")
    print("    DDPM requires T=1000 steps")
    print("    DDIM uses T'=50-100 steps with comparable quality")
    print("    Acceleration: 10-50×")
    print()
    print("  η=0:   deterministic; same latent → same output; invertible")
    print("  η=1:   matches DDPM variance schedule")
    print()
    print("  DDIM enables:")
    print("    Image interpolation in latent space")
    print("    Image editing via inversion + guided denoising")


# ── 4. Score matching and guidance ────────────────────────────────────────────
def score_matching_guidance():
    print("\n=== Score Matching & Classifier-Free Guidance ===")
    print("  Score function: s(x,t) = ∇_x log p_t(x)")
    print("  Equivalent to learning the noise: ε_θ ≈ -σ_t · s_θ(x_t, t)")
    print()
    print("  Classifier guidance (Dhariwal & Nichol 2021):")
    print("    ε_guided = ε_θ(x_t, t) - w · σ_t · ∇_x log p_φ(y|x_t)")
    print("    Requires a noisy classifier p_φ")
    print()
    print("  Classifier-free guidance (Ho & Salimans 2022):")
    print("    ε_guided = ε_θ(x_t, t, ∅) + w · (ε_θ(x_t, t, c) - ε_θ(x_t, t, ∅))")
    print("    w = guidance scale (typically 7.5 for Stable Diffusion)")
    print("    ∅ = null condition (unconditional); c = text/class condition")
    print()

    # Simulate guidance
    rng = np.random.default_rng(0)
    eps_uncond = rng.standard_normal(4)
    eps_cond   = rng.standard_normal(4)
    for w in [1.0, 3.0, 7.5, 15.0]:
        eps_guided = eps_uncond + w * (eps_cond - eps_uncond)
        print(f"    w={w:>4}: ||ε_guided||={np.linalg.norm(eps_guided):.3f}  "
              f"||ε_uncond||={np.linalg.norm(eps_uncond):.3f}")

    print()
    print("  Trade-off: higher w → better quality / prompt adherence, lower diversity")


# ── 5. Major diffusion models ─────────────────────────────────────────────────
def major_models():
    print("\n=== Major Diffusion Models ===")
    models = [
        ("DDPM",             2020, "Ho et al.; U-Net; CIFAR-10 sota at the time"),
        ("Score SDE",        2021, "Song et al.; continuous-time SDE framework"),
        ("ADM/DALL-E 2",     2022, "Class-conditional; beats GAN on ImageNet"),
        ("Stable Diffusion", 2022, "LDM; open-source; 512px; widely used"),
        ("SDXL",             2023, "1024px; improved architecture; two-stage"),
        ("DALL-E 3",         2023, "Recaptioned training data; excellent prompt follow"),
        ("Imagen",           2022, "Google; cascaded diffusion; text-to-image"),
        ("Midjourney v6",    2023, "Commercial; art-style; high quality"),
        ("Flux",             2024, "Flow matching; DiT architecture; open weights"),
        ("Sora",             2024, "Video; spatio-temporal patches; OpenAI"),
        ("CogVideoX",        2024, "Open video diffusion; 3D-full attention"),
    ]
    print(f"  {'Model':<22} {'Year'} {'Notes'}")
    print(f"  {'─'*22} {'─'*4} {'─'*50}")
    for m, y, d in models:
        print(f"  {m:<22} {y}  {d}")

    print()
    print("  Architecture progression:")
    print("    U-Net (DDPM/SD) → U-ViT → DiT → MMDiT (Flux)")
    print("    DiT: Diffusion Transformer; scales with compute")
    print("    MMDiT: Multimodal DiT; joint attention over text + image tokens")


if __name__ == "__main__":
    forward_process()
    reverse_process()
    ddim_overview()
    score_matching_guidance()
    major_models()
