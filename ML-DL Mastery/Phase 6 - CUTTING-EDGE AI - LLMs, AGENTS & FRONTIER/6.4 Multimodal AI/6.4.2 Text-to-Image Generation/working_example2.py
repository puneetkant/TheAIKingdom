"""
Working Example 2: Text-to-Image Generation (Diffusion)
Simulates the diffusion forward process on a synthetic 8x8 "image" (numpy array)
and visualises the denoising schedule.
Run: python working_example2.py
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


def linear_schedule(T, beta_start=1e-4, beta_end=0.02):
    return np.linspace(beta_start, beta_end, T)


def cosine_schedule(T, s=0.008):
    t = np.linspace(0, T, T + 1)
    f = np.cos((t / T + s) / (1 + s) * np.pi / 2) ** 2
    alpha_bar = f / f[0]
    betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
    return np.clip(betas, 1e-5, 0.999)


def forward_process(x0, t, betas):
    """q(x_t | x_0) = sqrt(alpha_bar_t) * x0 + sqrt(1-alpha_bar_t) * eps"""
    alpha_bar = np.cumprod(1 - betas)
    ab_t = alpha_bar[t]
    rng = np.random.default_rng(t)
    eps = rng.standard_normal(x0.shape)
    return np.sqrt(ab_t) * x0 + np.sqrt(1 - ab_t) * eps


def demo():
    print("=== Text-to-Image Generation: Diffusion Forward Process ===")
    T = 1000

    # Synthetic 8x8 "image"
    rng = np.random.default_rng(42)
    x0 = rng.standard_normal((8, 8))

    betas_linear = linear_schedule(T)
    betas_cosine = cosine_schedule(T)

    print(f"  Image shape: {x0.shape}")
    print(f"  T={T}, β_start={betas_linear[0]:.5f}, β_end={betas_linear[-1]:.4f}")

    alpha_bar_l = np.cumprod(1 - betas_linear)
    alpha_bar_c = np.cumprod(1 - betas_cosine)

    timesteps = [0, 100, 250, 500, 750, 999]
    noisy_images = [forward_process(x0, t, betas_linear) for t in timesteps]

    fig, axes = plt.subplots(2, len(timesteps), figsize=(14, 6))

    for j, (t, img) in enumerate(zip(timesteps, noisy_images)):
        ax = axes[0][j]
        ax.imshow(img, cmap="gray", vmin=-3, vmax=3)
        ax.set(title=f"t={t}", xticks=[], yticks=[])

    # Noise schedule comparison
    t_range = np.arange(T)
    axes[1][0].plot(t_range, 1 - alpha_bar_l, label="Linear", color="steelblue")
    axes[1][0].plot(t_range, 1 - alpha_bar_c, label="Cosine", color="tomato")
    axes[1][0].set(xlabel="Timestep t", ylabel="Signal noise ratio 1-ᾱ_t",
                   title="Noise Schedule Comparison")
    axes[1][0].legend()
    axes[1][0].grid(True, alpha=0.3)

    # β schedules
    axes[1][1].plot(t_range, betas_linear, label="Linear β", color="steelblue")
    axes[1][1].plot(t_range, betas_cosine, label="Cosine β", color="tomato")
    axes[1][1].set(xlabel="Timestep", ylabel="β_t", title="Beta Schedules")
    axes[1][1].legend()
    axes[1][1].grid(True, alpha=0.3)

    # SNR (signal-to-noise ratio)
    snr_l = alpha_bar_l / (1 - alpha_bar_l + 1e-10)
    snr_c = alpha_bar_c / (1 - alpha_bar_c + 1e-10)
    axes[1][2].semilogy(t_range, snr_l + 1e-6, label="Linear", color="steelblue")
    axes[1][2].semilogy(t_range, snr_c + 1e-6, label="Cosine", color="tomato")
    axes[1][2].set(xlabel="Timestep", ylabel="SNR (log scale)", title="Signal-to-Noise Ratio")
    axes[1][2].legend()
    axes[1][2].grid(True, alpha=0.3)

    # Variance of noisy images
    variances = [np.var(forward_process(x0, t, betas_linear)) for t in range(0, T, 50)]
    axes[1][3].plot(range(0, T, 50), variances, "o-", color="mediumseagreen", ms=4)
    axes[1][3].set(xlabel="Timestep", ylabel="Variance of x_t", title="Noisy Image Variance")
    axes[1][3].grid(True, alpha=0.3)

    # Hide empty axes
    for j in range(4, len(timesteps)):
        axes[1][j].axis("off")

    plt.tight_layout()
    plt.savefig(OUTPUT / "diffusion_forward.png", dpi=100)
    plt.close()
    print("  Saved diffusion_forward.png")


if __name__ == "__main__":
    demo()
