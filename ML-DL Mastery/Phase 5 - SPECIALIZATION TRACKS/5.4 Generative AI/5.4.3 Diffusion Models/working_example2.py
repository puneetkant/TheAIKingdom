"""
Working Example 2: Diffusion Models — forward/reverse process simulation
=========================================================================
Demonstrates forward noising schedule and denoising score matching concept.

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

def linear_beta_schedule(T=200, beta_start=1e-4, beta_end=0.02):
    return np.linspace(beta_start, beta_end, T)

def forward_process(x0, t, alphas_cumprod):
    """q(x_t | x_0) = sqrt(alpha_bar_t)*x0 + sqrt(1-alpha_bar_t)*eps"""
    alpha_bar = alphas_cumprod[t]
    eps = np.random.randn(*x0.shape)
    return np.sqrt(alpha_bar) * x0 + np.sqrt(1 - alpha_bar) * eps, eps

def demo():
    print("=== Diffusion Forward Process ===")
    betas = linear_beta_schedule(T=200)
    alphas = 1 - betas
    alphas_cumprod = np.cumprod(alphas)

    print(f"  T=200 | beta range: {betas[0]:.5f} -> {betas[-1]:.5f}")
    print(f"  sqrt(alpha_bar_T) = {np.sqrt(alphas_cumprod[-1]):.4f} (signal -> 0)")

    # Use a digit image as x0
    X = load_digits().data / 16.0
    x0 = X[0] * 2 - 1  # scale to [-1, 1]

    timesteps = [0, 10, 50, 100, 150, 199]
    fig, axes = plt.subplots(2, len(timesteps), figsize=(12, 4))
    for j, t in enumerate(timesteps):
        xt, eps = forward_process(x0, t, alphas_cumprod)
        axes[0, j].imshow(x0.reshape(8, 8), cmap="gray"); axes[0, j].axis("off")
        axes[0, j].set_title("x0" if j == 0 else "")
        axes[1, j].imshow(xt.reshape(8, 8), cmap="gray"); axes[1, j].axis("off")
        axes[1, j].set_title(f"t={t}")
    axes[0, 0].set_ylabel("x0"); axes[1, 0].set_ylabel("xt")
    plt.suptitle("DDPM Forward Process q(xt|x0)")
    plt.tight_layout(); plt.savefig(OUTPUT / "diffusion_forward.png"); plt.close()

    # SNR curve
    snr = alphas_cumprod / (1 - alphas_cumprod)
    plt.figure(figsize=(6, 3))
    plt.semilogy(snr); plt.xlabel("t"); plt.ylabel("SNR"); plt.title("Signal-to-Noise Ratio vs t")
    plt.tight_layout(); plt.savefig(OUTPUT / "diffusion_snr.png"); plt.close()
    print("  Saved diffusion_forward.png, diffusion_snr.png")

if __name__ == "__main__":
    demo()
