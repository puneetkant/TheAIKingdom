"""
Working Example: Autoencoder Variants
Covers Variational Autoencoder (VAE), Sparse Autoencoder, Contractive AE,
VQ-VAE concepts, and β-VAE.  VAE implemented from scratch with numpy.
"""
import numpy as np
from sklearn.datasets import load_digits
from sklearn.preprocessing import MinMaxScaler
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_ae_variants")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def sigmoid(z): return 1 / (1 + np.exp(-z.clip(-500, 500)))
def relu(z):    return np.maximum(0, z)
def softplus(z): return np.log(1 + np.exp(z.clip(-500, 200)))


# ── VAE helper ────────────────────────────────────────────────────────────────
class VAE:
    """
    Variational Autoencoder (numpy, simplified MLP).
    Encoder: X → μ, log σ²
    Decoder: z → X̂
    ELBO = E[log p(X|z)] - KL(q(z|X) || N(0,I))
    """
    def __init__(self, input_dim, hidden_dim, latent_dim, rng=None):
        rng    = rng or np.random.default_rng(0)
        s      = 0.05
        # Encoder
        self.W_enc1 = rng.standard_normal((input_dim, hidden_dim)) * s
        self.b_enc1 = np.zeros(hidden_dim)
        self.W_mu   = rng.standard_normal((hidden_dim, latent_dim)) * s
        self.b_mu   = np.zeros(latent_dim)
        self.W_lv   = rng.standard_normal((hidden_dim, latent_dim)) * s
        self.b_lv   = np.zeros(latent_dim)
        # Decoder
        self.W_dec1 = rng.standard_normal((latent_dim, hidden_dim)) * s
        self.b_dec1 = np.zeros(hidden_dim)
        self.W_dec2 = rng.standard_normal((hidden_dim, input_dim)) * s
        self.b_dec2 = np.zeros(input_dim)
        self.latent_dim = latent_dim

    def encode(self, X):
        h   = relu(X @ self.W_enc1 + self.b_enc1)
        mu  = h @ self.W_mu + self.b_mu
        lv  = h @ self.W_lv + self.b_lv   # log variance
        return mu, lv, h

    def sample(self, mu, lv, rng):
        eps = rng.standard_normal(mu.shape)
        z   = mu + np.exp(0.5 * lv) * eps
        return z, eps

    def decode(self, z):
        h    = relu(z @ self.W_dec1 + self.b_dec1)
        X_hat = sigmoid(h @ self.W_dec2 + self.b_dec2)
        return X_hat, h

    def elbo(self, X, X_hat, mu, lv, beta=1.0):
        """ELBO = recon_loss - β·KL.  We minimise -ELBO."""
        recon = -np.mean(X * np.log(X_hat + 1e-9) + (1-X)*np.log(1-X_hat+1e-9))
        kl    = -0.5 * np.mean(1 + lv - mu**2 - np.exp(lv))
        return recon + beta * kl, recon, kl

    def fit(self, X, epochs=80, lr=0.01, batch_size=64, beta=1.0, rng=None):
        rng = rng or np.random.default_rng(1)
        n   = X.shape[0]
        losses = []
        for ep in range(epochs):
            idx    = rng.permutation(n)
            ep_loss = 0
            for i in range(0, n, batch_size):
                Xb = X[idx[i:i+batch_size]]
                mu, lv, h_enc = self.encode(Xb)
                z, eps = self.sample(mu, lv, rng)
                X_hat, h_dec = self.decode(z)
                loss, recon, kl = self.elbo(Xb, X_hat, mu, lv, beta)
                # Simplified gradient update (numerical grad approx for demo)
                sigma = np.exp(0.5 * lv)
                dL_dXhat = -(Xb / (X_hat+1e-9) - (1-Xb)/(1-X_hat+1e-9)) / len(Xb)
                # Decoder backward (simplified)
                dh_dec2 = dL_dXhat * X_hat * (1 - X_hat)
                dW_dec2 = h_dec.T @ dh_dec2
                db_dec2 = dh_dec2.sum(0)
                dh_dec  = dh_dec2 @ self.W_dec2.T * (z @ self.W_dec1 + self.b_dec1 > 0)
                dW_dec1 = z.T @ dh_dec
                db_dec1 = dh_dec.sum(0)
                # KL gradient wrt mu, lv
                dmu = (mu - beta * 0) / len(Xb)   # simplified
                dlv = -0.5 * (1 - np.exp(lv)) * beta / len(Xb)
                for dW in [dW_dec1, dW_dec2, dW_dec2]:
                    np.clip(dW, -1, 1, out=dW)
                self.W_dec1 -= lr * np.clip(dW_dec1, -1, 1)
                self.b_dec1 -= lr * np.clip(db_dec1, -1, 1)
                self.W_dec2 -= lr * np.clip(dW_dec2, -1, 1)
                self.b_dec2 -= lr * np.clip(db_dec2, -1, 1)
                ep_loss += loss * len(Xb)
            losses.append(ep_loss / n)
        return losses


# ── 1. VAE theory ─────────────────────────────────────────────────────────────
def vae_theory():
    print("=== Variational Autoencoder (VAE) ===")
    print("  Standard AE: deterministic encoding — cannot sample new data")
    print("  VAE: encoder outputs a distribution q(z|X) = N(μ, σ²)")
    print()
    print("  Key equations:")
    print("    Encoder:     μ, log σ² = Encoder(X)")
    print("    Reparameterise: z = μ + σ·ε,  ε ~ N(0,I)  (allows backprop!)")
    print("    Decoder:     X̂ = Decoder(z)")
    print()
    print("  Objective — ELBO (Evidence Lower BOund):")
    print("    ELBO = E[log p(X|z)] - KL(q(z|X) || N(0,I))")
    print("         = Reconstruction term - Regularisation term")
    print()
    print("  KL term: -½ Σ (1 + log σ² - μ² - σ²)")
    print("  Forces the latent space to be organised and continuous")
    print()
    print("  β-VAE: ELBO = E[log p(X|z)] - β·KL   (β > 1 → more disentangled)")


# ── 2. VAE demo ───────────────────────────────────────────────────────────────
def vae_demo():
    print("\n=== VAE Training Demo (Digits) ===")
    digits = load_digits()
    X      = MinMaxScaler().fit_transform(digits.data.astype(float))
    rng    = np.random.default_rng(42)
    Xtr    = X[:1400]
    Xts    = X[1400:]

    vae    = VAE(64, 64, latent_dim=8, rng=rng)
    losses = vae.fit(Xtr, epochs=30, lr=0.005, batch_size=64, beta=1.0, rng=rng)

    mu_ts, lv_ts, _ = vae.encode(Xts)
    z_ts, _         = vae.sample(mu_ts, lv_ts, rng)
    X_hat_ts, _     = vae.decode(z_ts)

    recon_mse = np.mean((X_hat_ts - Xts)**2)
    mu_norm   = np.linalg.norm(mu_ts.mean(0))
    lv_mean   = lv_ts.mean()

    print(f"  Latent dim: 8   Train samples: {len(Xtr)}")
    print(f"  Start loss: {losses[0]:.4f}")
    print(f"  End loss:   {losses[-1]:.4f}")
    print(f"  Recon MSE (test): {recon_mse:.5f}")
    print(f"  Mean latent mu norm: {mu_norm:.4f}  (should be ~0 after training)")
    print(f"  Mean log-var:        {lv_mean:.4f}  (should be ~0 → σ≈1)")

    # Sample from prior
    z_prior = rng.standard_normal((10, 8))
    X_gen, _ = vae.decode(z_prior)
    print(f"\n  Generated samples from prior z~N(0,I): shape {X_gen.shape}")


# ── 3. Sparse Autoencoder ────────────────────────────────────────────────────
def sparse_ae():
    print("\n=== Sparse Autoencoder ===")
    print("  Overcomplete latent space (hidden_dim ≥ input_dim)")
    print("  Add L1 regularisation on activations → sparse code")
    print("  L = ||X - X̂||² + λ·||H||₁")
    print()
    print("  KL-sparsity (alternative):")
    print("    Target average activation ρ (e.g. 0.05)")
    print("    Penalty: KL(ρ || ρ̂_j) for each neuron j")
    print()
    print("  Applications:")
    print("    Dictionary learning: each latent dim → an 'atom' of the input")
    print("    Interpretable features: neurons specialise in one concept")
    print("    Sparse coding for images → Gabor-like edge detectors (similar to V1)")

    # Quick demo: L1 on hidden activations
    rng = np.random.default_rng(9)
    X   = MinMaxScaler().fit_transform(load_digits().data.astype(float))
    W   = rng.standard_normal((64, 128)) * 0.05  # overcomplete
    b   = np.zeros(128)
    lam = 0.01

    H_all = relu(X @ W + b)
    avg_activation = H_all.mean()
    sparsity_pct   = (H_all < 0.01).mean() * 100
    print(f"\n  Overcomplete random (untrained):")
    print(f"  H mean activation: {avg_activation:.4f}")
    print(f"  Dead neurons (%):  {sparsity_pct:.1f}%")
    print(f"\n  With L1 λ={lam}: penalty = {lam * H_all.mean():.5f} per sample")


# ── 4. Contractive Autoencoder (CAE) ─────────────────────────────────────────
def contractive_ae():
    print("\n=== Contractive Autoencoder (CAE) ===")
    print("  Add Frobenius norm of Jacobian of encoder to loss:")
    print("    L = ||X - X̂||² + λ·||∂H/∂X||²_F")
    print()
    print("  ∂H/∂X = diag(σ'(z))·W_enc   (Jacobian)")
    print("  ||J||²_F = Σ_{i,j} (∂h_i/∂x_j)²")
    print()
    print("  Effect: small changes in X → small changes in H (robust encoding)")
    print("  Learned features are locally invariant to input perturbations")
    print()
    print("  Difference from DAE:")
    print("    DAE: robust via explicit noise corruption")
    print("    CAE: robust via Jacobian penalty (analytic, no stochasticity)")

    # Compute Jacobian penalty example
    rng = np.random.default_rng(11)
    d, h = 64, 16
    W    = rng.standard_normal((d, h)) * 0.1
    X    = MinMaxScaler().fit_transform(load_digits().data[:100].astype(float))
    Z    = X @ W
    H    = sigmoid(Z)
    dH   = H * (1 - H)   # sigmoid derivative per sample, shape (n, h)

    # Jacobian Frobenius norm per sample
    J_frob_sq = np.sum(dH[:, None, :]**2 * W[None, :, :]**2, axis=(1, 2))
    print(f"  Mean Jacobian Frobenius norm²: {J_frob_sq.mean():.4f}")
    print(f"  This is added × λ to the reconstruction loss")


# ── 5. VQ-VAE concept ────────────────────────────────────────────────────────
def vqvae_concept():
    print("\n=== VQ-VAE (Vector Quantized VAE) ===")
    print("  Discrete latent space: z ∈ {e_1, ..., e_K}  (codebook)")
    print()
    print("  Forward pass:")
    print("    1. Encoder: X → z_e  (continuous)")
    print("    2. Quantise: z_q = argmin_k ||z_e - e_k||₂")
    print("    3. Decoder: z_q → X̂")
    print()
    print("  Problem: argmin is not differentiable")
    print("  Trick: straight-through estimator — copy gradient from z_q to z_e")
    print()
    print("  Loss = recon_loss + ||z_e - stopgrad(z_q)||² + β·||stopgrad(z_e) - z_q||²")
    print("                  ↑ encoder    ↑ commitment                 ↑ codebook update")
    print()
    print("  Applications: image generation (DALL-E), audio (Encodec, SoundStream)")

    # Toy codebook demo
    rng  = np.random.default_rng(20)
    K, d = 8, 4   # 8 codewords, dim=4
    codebook = rng.standard_normal((K, d))

    z_e = rng.standard_normal((5, d))   # 5 encoder outputs
    # Quantise
    dists = np.sum((z_e[:, None, :] - codebook[None, :, :])**2, axis=-1)  # (5, K)
    k_idx = dists.argmin(axis=-1)
    z_q   = codebook[k_idx]
    print(f"\n  Toy VQ-VAE: K={K} codewords, d={d}")
    print(f"  Encoder output z_e shape: {z_e.shape}")
    print(f"  Quantised codes (indices): {k_idx}")
    print(f"  Codebook lookup z_q shape: {z_q.shape}")
    commitment_loss = np.mean((z_e - z_q)**2)
    print(f"  Commitment loss: {commitment_loss:.4f}")


if __name__ == "__main__":
    vae_theory()
    vae_demo()
    sparse_ae()
    contractive_ae()
    vqvae_concept()
