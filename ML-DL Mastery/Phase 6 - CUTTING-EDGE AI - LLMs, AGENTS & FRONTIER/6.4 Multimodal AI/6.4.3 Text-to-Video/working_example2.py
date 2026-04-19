"""
Working Example 2: Text-to-Video Generation
Temporal consistency metric between synthetic video frames,
frame interpolation concept, and motion analysis.
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


def generate_synthetic_video(n_frames=16, h=8, w=8, noise_std=0.1):
    """Generate synthetic video frames with smooth motion + noise."""
    rng = np.random.default_rng(42)
    base = rng.standard_normal((h, w))
    frames = []
    for t in range(n_frames):
        # Smooth temporal evolution + noise
        motion = np.roll(base, shift=t // 4, axis=1)
        frame = motion + rng.normal(0, noise_std, (h, w))
        frames.append(frame)
    return np.array(frames)  # (T, H, W)


def temporal_consistency(frames):
    """Mean squared error between consecutive frames."""
    return [np.mean((frames[i+1] - frames[i])**2) for i in range(len(frames)-1)]


def ssim_proxy(a, b):
    """Simplified SSIM proxy (not full SSIM — for demo)."""
    mu_a, mu_b = a.mean(), b.mean()
    sigma_a, sigma_b = a.std(), b.std()
    sigma_ab = np.mean((a - mu_a) * (b - mu_b))
    c1, c2 = 0.01**2, 0.03**2
    ssim = ((2*mu_a*mu_b + c1) * (2*sigma_ab + c2)) / \
           ((mu_a**2 + mu_b**2 + c1) * (sigma_a**2 + sigma_b**2 + c2))
    return ssim


def linear_interpolate(f0, f1, alpha):
    """Linear frame interpolation."""
    return (1 - alpha) * f0 + alpha * f1


def demo():
    print("=== Text-to-Video: Temporal Consistency ===")
    frames = generate_synthetic_video(n_frames=16)
    print(f"  Video shape: {frames.shape} (frames, H, W)")

    tc = temporal_consistency(frames)
    ssims = [ssim_proxy(frames[i], frames[i+1]) for i in range(len(frames)-1)]

    print(f"  Mean temporal MSE: {np.mean(tc):.4f}")
    print(f"  Mean SSIM proxy:   {np.mean(ssims):.4f}")

    # High-noise video for comparison
    rng = np.random.default_rng(99)
    noisy_frames = np.array([rng.standard_normal((8, 8)) for _ in range(16)])
    tc_noisy = temporal_consistency(noisy_frames)
    ssims_noisy = [ssim_proxy(noisy_frames[i], noisy_frames[i+1]) for i in range(15)]

    # Frame interpolation
    f0, f1 = frames[0], frames[4]
    alphas = np.linspace(0, 1, 9)
    interp_frames = [linear_interpolate(f0, f1, a) for a in alphas]
    interp_mse = [np.mean((interp_frames[i+1] - interp_frames[i])**2) for i in range(8)]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # Sample frames
    for j, idx in enumerate([0, 4, 8, 12]):
        if j < 4:
            ax = axes[0][j // 2] if j < 2 else axes[1][j // 2 - 1]
            # Actually let's do a different layout

    # Temporal consistency: smooth vs noisy
    axes[0][0].plot(range(len(tc)), tc, "o-", color="steelblue", lw=2, ms=5, label="Smooth")
    axes[0][0].plot(range(len(tc_noisy)), tc_noisy, "s--", color="tomato", lw=2, ms=5, label="Noisy")
    axes[0][0].set(xlabel="Frame Pair", ylabel="Temporal MSE",
                   title="Temporal Consistency (lower is better)")
    axes[0][0].legend()
    axes[0][0].grid(True, alpha=0.3)

    # SSIM comparison
    axes[0][1].plot(range(len(ssims)), ssims, "o-", color="steelblue", lw=2, label="Smooth")
    axes[0][1].plot(range(len(ssims_noisy)), ssims_noisy, "s--", color="tomato", lw=2, label="Noisy")
    axes[0][1].set(xlabel="Frame Pair", ylabel="SSIM Proxy",
                   title="Frame-to-Frame SSIM (higher is better)")
    axes[0][1].legend()
    axes[0][1].grid(True, alpha=0.3)

    # Frame interpolation
    axes[1][0].plot(alphas, [np.mean(f) for f in interp_frames], "o-", color="mediumseagreen", lw=2)
    axes[1][0].set(xlabel="Interpolation alpha", ylabel="Mean Pixel Value",
                   title="Linear Frame Interpolation")
    axes[1][0].grid(True, alpha=0.3)

    # Sample frame grid
    sample_frames = [frames[i] for i in [0, 4, 8, 12]]
    combined = np.concatenate(sample_frames, axis=1)
    axes[1][1].imshow(combined, cmap="gray", vmin=-2, vmax=2)
    axes[1][1].set(title="Sample Frames: t=0, 4, 8, 12", xticks=[], yticks=[])

    plt.tight_layout()
    plt.savefig(OUTPUT / "text_to_video.png", dpi=100)
    plt.close()
    print("  Saved text_to_video.png")


if __name__ == "__main__":
    demo()
