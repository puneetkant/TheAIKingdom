"""
Working Example: Text-to-Image Generation
Covers diffusion models, DALL-E 3, Stable Diffusion, ControlNet,
and evaluation metrics.
"""
import numpy as np
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_t2i")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── 1. Landscape ──────────────────────────────────────────────────────────────
def t2i_landscape():
    print("=== Text-to-Image Generation ===")
    print()
    print("  Two dominant paradigms:")
    print("  1. Diffusion models (current SOTA)")
    print("  2. Autoregressive (DALL-E 1, VQVAE; partly GPT-4o)")
    print()
    models = [
        ("DALL-E 3",         "OpenAI; GPT-4 recaptioning; prompt adherence SOTA"),
        ("Stable Diffusion", "Stability AI; open weights; hugely popular"),
        ("SD XL",            "Stable Diffusion XL; 2-stage; 1024x1024"),
        ("SD 3.5",           "MMDiT architecture; improved text rendering"),
        ("Midjourney v6",    "Aesthetic quality leader; closed"),
        ("Imagen 3",         "Google DeepMind; T5 text encoder; high fidelity"),
        ("Flux.1",           "Black Forest Labs; rectified flow; open weights"),
        ("Playground v3",    "MLLM captioner; strong prompt following"),
    ]
    for m, d in models:
        print(f"  {m:<24} {d}")


# ── 2. Diffusion process ──────────────────────────────────────────────────────
def diffusion_process():
    print("\n=== Diffusion Model Mechanics ===")
    print()
    print("  Forward process (add noise):")
    print("    q(x_t | x_{t-1}) = N(x_t; sqrt(1-β_t)*x_{t-1}, β_t*I)")
    print("    At T=1000 steps, image becomes pure Gaussian noise")
    print()
    print("  Reverse process (denoise):")
    print("    p_θ(x_{t-1} | x_t, c) = N(μ_θ(x_t, t, c), Σ_θ)")
    print("    c = text conditioning (CLIP or T5 embeddings)")
    print()
    print("  Training objective (simplified):")
    print("    L = E[|| ε - ε_θ(x_t, t, c) ||²]")
    print("    Predict noise ε added at step t; condition on text c")
    print()

    # Simulate denoising trajectory
    rng = np.random.default_rng(0)
    T = 20
    x = rng.normal(0, 1, 4)  # pure noise (4D toy image)
    noise_levels = np.linspace(1.0, 0.0, T+1)
    print("  Simulated denoising trajectory (SNR increasing):")
    for t in [0, 5, 10, 15, 19]:
        snr = (1 - noise_levels[t])**2 / (noise_levels[t]**2 + 1e-6)
        print(f"    t={T-t:3d}: noise_level={noise_levels[t]:.2f}  SNR={snr:.2f}")

    print()
    print("  Latent diffusion (Stable Diffusion):")
    print("    Encode image → latent z with VAE (8x spatial compression)")
    print("    Diffuse in latent space (much cheaper!)")
    print("    UNet or DiT denoiser in latent space")
    print("    Decode z → image with VAE decoder")

    print()
    print("  Classifier-free guidance (CFG):")
    print("    ε_cond = ε_θ(x_t, c)")
    print("    ε_uncond = ε_θ(x_t, ∅)")
    print("    ε_final = ε_uncond + w * (ε_cond - ε_uncond)")
    print("    w=7.5 → strong adherence; w=1 → diversity; w=0 → unconditional")


# ── 3. Stable Diffusion components ───────────────────────────────────────────
def sd_components():
    print("\n=== Stable Diffusion Architecture ===")
    print()
    components = [
        ("Text encoder",  "CLIP ViT-L or OpenCLIP H/G; tokenise → embed (77 tokens)"),
        ("VAE",           "Encoder 512→64 latent; Decoder 64→512; KL loss"),
        ("U-Net",         "Latent denoiser; skip connections; cross-attn to text"),
        ("Scheduler",     "DDPM / DDIM / DPM-Solver; 1000→50→20 steps"),
    ]
    for c, d in components:
        print(f"  {c:<16} {d}")
    print()
    print("  Conditioning mechanisms:")
    conds = [
        ("Cross-attention", "Standard text conditioning via Q=latent, K/V=text"),
        ("ControlNet",      "Additional encoder; conditioning image (depth/canny/pose)"),
        ("IP-Adapter",      "Reference image style via decoupled cross-attention"),
        ("LoRA",            "Low-rank adaptors; personalisation (DreamBooth)"),
    ]
    for c, d in conds:
        print(f"  {c:<18} {d}")

    print()
    print("  SDXL improvements over SD 1.5:")
    improvements = [
        "3x larger UNet (2.6B vs 860M params)",
        "Second (refiner) model for high-frequency detail",
        "OpenCLIP G text encoder + CLIP L (dual encoders)",
        "256 vs 77 token text length",
        "Native 1024x1024 resolution",
    ]
    for imp in improvements:
        print(f"  • {imp}")


# ── 4. Evaluation ─────────────────────────────────────────────────────────────
def t2i_evaluation():
    print("\n=== Text-to-Image Evaluation ===")
    print()
    metrics = [
        ("FID",         "Frechet Inception Distance; image quality vs reference"),
        ("CLIP Score",  "Prompt-image alignment; cosine similarity in CLIP space"),
        ("IS",          "Inception Score; quality + diversity (less used now)"),
        ("DINO FID",    "DINO features; better than IS for semantic quality"),
        ("HPS v2",      "Human preference score; trained on human votes"),
        ("PickScore",   "CLIP-based; trained on user preferences"),
        ("ImageReward", "Learned reward from human feedback"),
        ("T2I-CompBench","Compositional understanding; attribute binding; relations"),
    ]
    print(f"  {'Metric':<16} {'Description'}")
    for m, d in metrics:
        print(f"  {m:<16} {d}")

    print()
    print("  Common issues:")
    issues = [
        ("Text rendering",    "Models struggle with legible text in images"),
        ("Attribute binding", "'Red cube and blue sphere' → attribute confusion"),
        ("Counting",          "More than 3-4 of any object causes errors"),
        ("Hands/fingers",     "Historically poor; Flux.1 significantly improved"),
        ("Prompt adherence",  "Complex scenes with many conditions"),
    ]
    for i, d in issues:
        print(f"  {i:<22} {d}")


if __name__ == "__main__":
    t2i_landscape()
    diffusion_process()
    sd_components()
    t2i_evaluation()
