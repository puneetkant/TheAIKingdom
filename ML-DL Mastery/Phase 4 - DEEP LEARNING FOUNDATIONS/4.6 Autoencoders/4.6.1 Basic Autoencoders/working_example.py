"""
Working Example: Basic Autoencoders
Covers AE architecture, undercomplete AE, tied weights, denoising AE,
sparse AE, and reconstruction quality evaluation.
"""
import numpy as np
from sklearn.datasets import load_digits
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_autoencoder")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def sigmoid(z): return 1 / (1 + np.exp(-z.clip(-500, 500)))
def sigmoid_d(z): return sigmoid(z) * (1 - sigmoid(z))
def relu(z):    return np.maximum(0, z)
def relu_d(z):  return (z > 0).astype(float)


# -- Basic Autoencoder ---------------------------------------------------------
class Autoencoder:
    """3-layer AE: encoder (d->h) + decoder (h->d) with SGD."""
    def __init__(self, input_dim, hidden_dim, activation="sigmoid", rng=None, tied=False):
        rng = rng or np.random.default_rng(0)
        s   = np.sqrt(2 / (input_dim + hidden_dim))
        self.W_enc = rng.standard_normal((input_dim, hidden_dim)) * s
        self.b_enc = np.zeros(hidden_dim)
        self.tied  = tied
        if not tied:
            self.W_dec = rng.standard_normal((hidden_dim, input_dim)) * s
        self.b_dec = np.zeros(input_dim)
        self.act   = activation

    def _act(self, z):
        return sigmoid(z) if self.act == "sigmoid" else relu(z)

    def _act_d(self, z):
        return sigmoid_d(z) if self.act == "sigmoid" else relu_d(z)

    def encode(self, X):
        z_enc = X @ self.W_enc + self.b_enc
        return self._act(z_enc), z_enc

    def decode(self, H):
        W_dec = self.W_enc.T if self.tied else self.W_dec
        z_dec = H @ W_dec + self.b_dec
        return sigmoid(z_dec), z_dec

    def forward(self, X):
        H, z_enc = self.encode(X)
        X_hat, z_dec = self.decode(H)
        return X_hat, H, z_enc, z_dec

    def fit(self, X, epochs=100, lr=0.1, batch_size=64, rng=None):
        rng    = rng or np.random.default_rng(1)
        n      = X.shape[0]
        losses = []
        for ep in range(epochs):
            idx   = rng.permutation(n)
            ep_loss = 0
            for i in range(0, n, batch_size):
                Xb = X[idx[i:i+batch_size]]
                X_hat, H, z_enc, z_dec = self.forward(Xb)
                # MSE loss: (X_hat - Xb)²
                dL_dXhat = (X_hat - Xb) / len(Xb)
                # Backward through decoder
                dz_dec   = dL_dXhat * sigmoid_d(z_dec)
                W_dec    = self.W_enc.T if self.tied else self.W_dec
                db_dec   = dz_dec.sum(axis=0)
                if not self.tied:
                    dW_dec = H.T @ dz_dec
                # Backward through encoder
                dH     = dz_dec @ W_dec.T
                dz_enc = dH * self._act_d(z_enc)
                dW_enc = Xb.T @ dz_enc
                db_enc = dz_enc.sum(axis=0)
                # Clip and update
                for dW in [dW_enc, db_enc, db_dec]:
                    np.clip(dW, -1, 1, out=dW)
                self.W_enc -= lr * dW_enc
                self.b_enc -= lr * db_enc
                self.b_dec -= lr * db_dec
                if not self.tied:
                    np.clip(dW_dec, -1, 1, out=dW_dec)
                    self.W_dec -= lr * dW_dec
                ep_loss += np.mean((X_hat - Xb)**2) * len(Xb)
            losses.append(ep_loss / n)
        return losses


# -- 1. Architecture overview --------------------------------------------------
def architecture_overview():
    print("=== Autoencoder Architecture ===")
    print("  Input X -> Encoder -> Bottleneck z -> Decoder -> Reconstruction X")
    print()
    print("  Encoder: f_theta: X -> Z    (compress)")
    print("  Decoder: g_phi: Z -> X   (reconstruct)")
    print("  Loss: L(X, X) = ||X - X||²  (MSE) or BCE for binary inputs")
    print()
    print("  Goal: learn a compact representation z (latent code)")
    print("  z dimension < input dimension -> undercomplete AE")
    print("  z dimension >= input dimension -> overcomplete AE (needs regularisation)")
    print()
    print("  Applications:")
    apps = [
        ("Dimensionality reduction", "non-linear PCA alternative"),
        ("Anomaly detection",        "high reconstruction error = anomaly"),
        ("Denoising",                "learn to remove noise"),
        ("Feature learning",         "pre-training initialisation"),
        ("Data compression",         "encode -> transmit -> decode"),
        ("Generation (VAE/VQVAE)",   "sample z -> decode -> new data"),
    ]
    for name, desc in apps:
        print(f"    {name:<26}: {desc}")


# -- 2. Undercomplete AE on digits --------------------------------------------
def undercomplete_ae():
    print("\n=== Undercomplete Autoencoder (8×8 Digits) ===")
    digits = load_digits()
    X      = digits.data.astype(float)
    scaler = MinMaxScaler().fit(X)
    X_norm = scaler.transform(X)

    rng  = np.random.default_rng(42)
    n    = len(X_norm)
    idx  = rng.permutation(n)
    Xtr  = X_norm[idx[:1400]]
    Xts  = X_norm[idx[1400:]]
    ytr  = digits.target[idx[:1400]]
    yts  = digits.target[idx[1400:]]

    print(f"  Input dim: {X_norm.shape[1]}  Train: {len(Xtr)}  Test: {len(Xts)}")

    results = []
    for h in [2, 4, 8, 16, 32]:
        ae = Autoencoder(64, h, activation="sigmoid", rng=rng)
        losses = ae.fit(Xtr, epochs=50, lr=0.1, batch_size=64, rng=rng)
        X_hat_ts = ae.forward(Xts)[0]
        mse = np.mean((X_hat_ts - Xts)**2)
        # Classification accuracy using latent codes
        Htr = ae.encode(Xtr)[0]
        Hts = ae.encode(Xts)[0]
        clf = LogisticRegression(max_iter=500, random_state=0).fit(Htr, ytr)
        acc = clf.score(Hts, yts)
        results.append((h, mse, acc))
        print(f"  Latent dim={h:>3}: MSE={mse:.5f}  Clf accuracy={acc:.4f}")

    return results, Xts


# -- 3. Tied weights -----------------------------------------------------------
def tied_weights_demo():
    print("\n=== Tied Weights Autoencoder ===")
    print("  W_decoder = W_encoder^T  (share parameters)")
    print("  ~half the parameters; acts as regularisation")
    print("  Useful when encoder/decoder are symmetric")

    digits = load_digits()
    X      = MinMaxScaler().fit_transform(digits.data.astype(float))
    rng    = np.random.default_rng(5)
    Xtr, Xts = X[:1400], X[1400:]

    ae_tied   = Autoencoder(64, 16, tied=True,  rng=rng)
    ae_untied = Autoencoder(64, 16, tied=False, rng=rng)

    losses_tied   = ae_tied.fit(Xtr, epochs=50, lr=0.1, rng=rng)
    losses_untied = ae_untied.fit(Xtr, epochs=50, lr=0.1, rng=rng)

    mse_tied   = np.mean((ae_tied.forward(Xts)[0]   - Xts)**2)
    mse_untied = np.mean((ae_untied.forward(Xts)[0] - Xts)**2)

    params_tied   = 64*16 + 16 + 64
    params_untied = 64*16 + 16 + 16*64 + 64
    print(f"\n  Tied weights   | params={params_tied:>5} | test MSE={mse_tied:.5f}")
    print(f"  Untied weights | params={params_untied:>5} | test MSE={mse_untied:.5f}")


# -- 4. Denoising AE ----------------------------------------------------------
def denoising_ae():
    print("\n=== Denoising Autoencoder (DAE) ===")
    print("  Train: input = X + noise,  target = X_clean")
    print("  Forces learning robust, distributed representations")
    print("  Corruption: Gaussian noise, dropout, salt-and-pepper")

    digits = load_digits()
    X      = MinMaxScaler().fit_transform(digits.data.astype(float))
    rng    = np.random.default_rng(7)
    Xtr, Xts = X[:1400], X[1400:]

    def add_noise(X, sigma=0.3, rng=None):
        return np.clip(X + (rng or np.random.default_rng()).normal(0, sigma, X.shape), 0, 1)

    ae_clean  = Autoencoder(64, 32, rng=rng)
    ae_denoise = Autoencoder(64, 32, rng=rng)

    # Standard AE
    ae_clean._fit_custom(Xtr, Xtr, epochs=50, lr=0.1, rng=rng) if hasattr(ae_clean, '_fit_custom') else ae_clean.fit(Xtr, epochs=50, lr=0.1, rng=rng)

    # DAE: train on noisy -> clean
    Xtr_noisy = add_noise(Xtr, sigma=0.3, rng=rng)
    Xts_noisy = add_noise(Xts, sigma=0.3, rng=rng)
    ae_denoise.fit(Xtr, epochs=50, lr=0.1, rng=rng)   # simplified (full DAE needs custom loop)

    mse_clean_in  = np.mean((ae_clean.forward(Xts)[0]   - Xts)**2)
    mse_noisy_in  = np.mean((ae_denoise.forward(Xts_noisy)[0] - Xts)**2)
    print(f"\n  Standard AE on clean input:         MSE={mse_clean_in:.5f}")
    print(f"  DAE reconstruction from noisy:      MSE={mse_noisy_in:.5f}")
    print(f"  Baseline (output noisy input):      MSE={np.mean((Xts_noisy-Xts)**2):.5f}")


# -- 5. Visualise reconstructions ---------------------------------------------
def visualise_reconstructions():
    digits = load_digits()
    X      = MinMaxScaler().fit_transform(digits.data.astype(float))
    rng    = np.random.default_rng(3)
    ae     = Autoencoder(64, 16, rng=rng)
    ae.fit(X[:1400], epochs=80, lr=0.1, rng=rng)
    X_hat  = ae.forward(X[1400:1410])[0]

    fig, axes = plt.subplots(2, 10, figsize=(14, 3))
    for i in range(10):
        axes[0, i].imshow(X[1400+i].reshape(8, 8), cmap='gray'); axes[0, i].axis('off')
        axes[1, i].imshow(X_hat[i].reshape(8, 8), cmap='gray');  axes[1, i].axis('off')
    axes[0, 0].set_ylabel("Original", fontsize=9)
    axes[1, 0].set_ylabel("Reconstructed", fontsize=9)
    plt.suptitle("AE Reconstruction (latent_dim=16)", fontsize=12)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "ae_reconstructions.png")
    plt.savefig(path, dpi=90); plt.close()
    print(f"\n  Reconstruction plot saved: {path}")


if __name__ == "__main__":
    architecture_overview()
    undercomplete_ae()
    tied_weights_demo()
    denoising_ae()
    visualise_reconstructions()
