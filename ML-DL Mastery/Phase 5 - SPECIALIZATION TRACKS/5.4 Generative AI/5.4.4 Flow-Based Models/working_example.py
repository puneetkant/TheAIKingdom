"""
Working Example: Flow-Based Generative Models
Covers normalising flows, change-of-variables formula,
coupling layers, continuous flows, and major models.
"""
import numpy as np
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_flows")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -- 1. Normalising flows theory -----------------------------------------------
def normalising_flows_theory():
    print("=== Normalising Flows ===")
    print()
    print("  Exact density estimation via invertible transformations")
    print()
    print("  Change of variables:")
    print("    z ~ p_z(z)  (simple prior, e.g. N(0,I))")
    print("    x = f(z)    (invertible, differentiable transformation)")
    print()
    print("    log p_x(x) = log p_z(z) - log |det J_f(z)|")
    print("    where J_f = df/dz  (Jacobian)")
    print()
    print("  Training: maximise log-likelihood exactly (no ELBO approximation)")
    print("    L = E_{x~p_data}[log p_x(x)]")
    print()
    print("  Sampling:")
    print("    z ~ N(0, I)  ->  x = f(z)")
    print()
    print("  Compared to other generative models:")
    print("    VAE:       approximate posterior; latent compression; fast gen")
    print("    GAN:       no density estimate; best quality; unstable")
    print("    Diffusion: exact density (ELBO); slow gen; current state-of-art")
    print("    Flow:      exact density; fast gen; constrained architectures")


# -- 2. 1D flow demo -----------------------------------------------------------
def flow_1d_demo():
    print("\n=== 1D Flow Demo ===")
    rng = np.random.default_rng(0)

    # Target: mixture of Gaussians
    def sample_target(n):
        c = rng.integers(2, size=n)
        return c * rng.normal(2, 0.5, n) + (1-c) * rng.normal(-2, 0.5, n)

    # Simple affine flow: x = exp(s)·z + t  (s, t learnable)
    s = 0.5; t = 1.0  # initialise

    # log p(x) = log p_z(z) - log |exp(s)| = log N(z;0,1) - s
    def log_prob(x, s, t):
        z = (x - t) * np.exp(-s)  # inverse: z = (x-t)/exp(s)
        log_pz = -0.5 * z**2 - 0.5 * np.log(2*np.pi)
        return log_pz - s

    X = sample_target(500)
    lp_before = log_prob(X, 0, 0).mean()   # identity transform
    lp_after  = log_prob(X, s, t).mean()

    print(f"  Target: mixture of N(-2, 0.5) and N(2, 0.5)")
    print(f"  Flow: x = exp(s)·z + t,  s={s}, t={t}")
    print(f"  Log-likelihood (identity): {lp_before:.4f}")
    print(f"  Log-likelihood (affine):   {lp_after:.4f}")

    # Simple 2D flow: coupling layer
    print()
    print("  2D affine coupling layer:")
    z2 = rng.standard_normal((10, 2))
    # First dim passes through; second dim scaled/shifted by first
    z1, z2_ = z2[:,0], z2[:,1]
    s_fn = np.tanh(z1 * 0.5)   # scale from z1
    t_fn = z1 * 0.3             # shift from z1
    x2   = np.exp(s_fn) * z2_ + t_fn
    log_det = s_fn.sum()

    print(f"    Input z: {z2[:3].round(3)}")
    x_out = np.stack([z1, x2], axis=1)
    print(f"    Output x: {x_out[:3].round(3)}")
    print(f"    Log |det J|: {log_det:.4f}")


# -- 3. RealNVP coupling layers ------------------------------------------------
def realnvp_overview():
    print("\n=== RealNVP (Real-valued Non-Volume Preserving) ===")
    print("  Dinh et al. (2017)")
    print()
    print("  Affine coupling layer:")
    print("    Given x split into [x_{1:d}, x_{d+1:D}]:")
    print("    z_{1:d}    = x_{1:d}  (identity)")
    print("    z_{d+1:D}  = x_{d+1:D} ⊙ exp(s(x_{1:d})) + t(x_{1:d})")
    print("    s(·), t(·) = arbitrary neural networks (NN only evaluated forward)")
    print()
    print("  Inverse (trivially tractable):")
    print("    x_{1:d}    = z_{1:d}")
    print("    x_{d+1:D}  = (z_{d+1:D} - t(z_{1:d})) ⊙ exp(-s(z_{1:d}))")
    print()
    print("  Jacobian is triangular -> log det = Sigma s(x_{1:d})")
    print("    (O(D) rather than O(D³) for full Jacobian)")
    print()
    print("  Multi-scale architecture:")
    print("    Factor out half the dimensions after each coupling layer")
    print("    Reduces spatial resolution; captures multi-scale features")

    # Toy coupling layer implementation
    rng = np.random.default_rng(0)
    D = 6; d = 3
    x = rng.standard_normal(D)
    x1, x2 = x[:d], x[d:]

    # Simple affine s and t networks (linear for demo)
    W_s = rng.standard_normal((d, d)) * 0.1
    W_t = rng.standard_normal((d, d)) * 0.1
    s   = np.tanh(x1 @ W_s)
    t   = x1 @ W_t
    z2  = x2 * np.exp(s) + t
    z   = np.concatenate([x1, z2])
    log_det = s.sum()

    # Invert
    x1r = z[:d]
    s_r = np.tanh(x1r @ W_s)
    t_r = x1r @ W_t
    x2r = (z[d:] - t_r) * np.exp(-s_r)

    print()
    print(f"  Toy coupling layer (D={D}, d={d}):")
    print(f"    x  = {x.round(3)}")
    print(f"    z  = {z.round(3)}")
    print(f"    x' = {np.concatenate([x1r, x2r]).round(3)}")
    print(f"    Reconstruction error: {np.abs(x - np.concatenate([x1r, x2r])).max():.2e}")
    print(f"    log det: {log_det:.4f}")


# -- 4. Glow (generative flow) -------------------------------------------------
def glow_overview():
    print("\n=== Glow ===")
    print("  Kingma & Dhariwal (2018)")
    print()
    print("  One step of Glow = three components:")
    print("    1. ActNorm:  Affine normalisation (like BatchNorm but per-activation)")
    print("    2. 1×1 Invertible Conv: learnable channel permutation W (LU decomp)")
    print("    3. Affine Coupling: RealNVP-style s, t networks")
    print()
    print("  ActNorm log det = H·W·Sigma_c log|s_c|")
    print("  1×1 Conv log det = H·W·log|det(W)|")
    print()
    print("  Why 1×1 invertible conv?")
    print("    Replaces fixed channel permutation in RealNVP")
    print("    Learnable mixing of channels; more expressive")
    print("    LU decomposition: O(D) log det")
    print()
    print("  Results: high-quality face generation; 256×256")
    print("  Enables exact latent manipulation (smile, age, lighting)")


# -- 5. Continuous normalising flows ------------------------------------------
def continuous_flows():
    print("\n=== Continuous Normalising Flows (CNF) ===")
    print("  Chen et al. (2018) — Neural ODE framework")
    print()
    print("  Instead of discrete steps, define dynamics via ODE:")
    print("    dz(t)/dt = f_theta(z(t), t)   — neural network vector field")
    print("    z(0) = x  ->  z(T) = z_T   (forward ODE solve)")
    print()
    print("  Log-likelihood via instantaneous change of variables:")
    print("    d/dt log p(z(t)) = -tr(df/dz(t))   (trace of Jacobian)")
    print("    -> continuous-time version of coupling log det")
    print()
    print("  Training: minimise -E[log p_x(x)] via adjoint method (backprop through ODE)")
    print()
    print("  Flow Matching (Lipman 2022, Liu 2022):")
    print("    Learn vector field to interpolate p_0 -> p_1")
    print("    Conditional Flow Matching (CFM): simpler targets, no need for ODE solver")
    print("    Straight trajectories -> fewer NFEs (function evaluations)")
    print()
    print("  Models using flow matching:")
    print("    Flux: image generation via flow matching + DiT")
    print("    Voicebox: audio generation")
    print("    Emu3 / CogVideoX: multi-modal")


# -- 6. Autoregressive flows ---------------------------------------------------
def autoregressive_flows():
    print("\n=== Autoregressive Flows ===")
    print("  Masked Autoregressive Flow (MAF) — Papamakarios 2017")
    print("  Inverse Autoregressive Flow (IAF) — Kingma 2016")
    print()
    print("  Autoregressive model:")
    print("    p(x) = Pi_i p(x_i | x_{1:i-1})")
    print("    x_i = (z_i - mu_i(x_{1:i-1})) / exp(alpha_i(x_{1:i-1}))")
    print()
    print("  MAF:")
    print("    Fast density evaluation; slow sampling (sequential per dimension)")
    print("    Good for density estimation tasks")
    print()
    print("    IAF: Fast sampling; slow density evaluation")
    print("    Good for variational inference posteriors")
    print()
    print("  MADE (Masked AutoEncoder for Distribution Estimation):")
    print("    Efficient masked connections for autoregressive structure")
    print("    Enables parallel training of autoregressive model")
    print()
    print("  Spline flows:")
    print("    Monotone rational-quadratic splines as bijection")
    print("    Very expressive per-element; used in NSF (Neural Spline Flow)")


if __name__ == "__main__":
    normalising_flows_theory()
    flow_1d_demo()
    realnvp_overview()
    glow_overview()
    continuous_flows()
    autoregressive_flows()
