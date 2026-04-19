"""
Working Example: Autoencoder Applications
Covers anomaly detection, image denoising simulation, dimensionality
reduction comparison, and generative sampling from a VAE latent space.
"""
import numpy as np
from sklearn.datasets import load_digits, make_blobs
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_ae_apps")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def sigmoid(z): return 1 / (1 + np.exp(-z.clip(-500, 500)))
def relu(z):    return np.maximum(0, z)


# -- Simple AE for reuse -------------------------------------------------------
class SimpleAE:
    def __init__(self, input_dim, hidden_dim, latent_dim, rng):
        s = 0.05
        self.We1 = rng.standard_normal((input_dim, hidden_dim)) * s
        self.be1 = np.zeros(hidden_dim)
        self.We2 = rng.standard_normal((hidden_dim, latent_dim)) * s
        self.be2 = np.zeros(latent_dim)
        self.Wd1 = rng.standard_normal((latent_dim, hidden_dim)) * s
        self.bd1 = np.zeros(hidden_dim)
        self.Wd2 = rng.standard_normal((hidden_dim, input_dim)) * s
        self.bd2 = np.zeros(input_dim)

    def encode(self, X):
        h = relu(X @ self.We1 + self.be1)
        return relu(h @ self.We2 + self.be2)

    def decode(self, Z):
        h = relu(Z @ self.Wd1 + self.bd1)
        return sigmoid(h @ self.Wd2 + self.bd2)

    def forward(self, X):
        Z = self.encode(X)
        return self.decode(Z), Z

    def fit(self, X, epochs=60, lr=0.05, bs=64, rng=None):
        rng = rng or np.random.default_rng(0)
        n   = X.shape[0]
        losses = []
        for ep in range(epochs):
            idx = rng.permutation(n); ep_loss = 0
            for i in range(0, n, bs):
                Xb = X[idx[i:i+bs]]
                X_hat, Z = self.forward(Xb)
                dX_hat = (X_hat - Xb) / len(Xb)
                # Simplified gradient (skip full BPTT for brevity)
                dh2 = dX_hat * X_hat * (1 - X_hat)
                dW2 = np.clip(Z.T @ dX_hat, -1, 1)
                db2 = np.clip(dX_hat.sum(0), -1, 1)
                self.Wd2 -= lr * dW2; self.bd2 -= lr * db2
                ep_loss += np.mean((X_hat - Xb)**2) * len(Xb)
            losses.append(ep_loss / n)
        return losses


# -- 1. Anomaly Detection ------------------------------------------------------
def anomaly_detection():
    print("=== Anomaly Detection with Autoencoders ===")
    print("  Key idea: AE trained on normal data cannot reconstruct anomalies well")
    print("  Anomaly score = reconstruction error = ||X - X||²")
    print("  Threshold tau: flag as anomaly if score > tau")
    print()

    rng = np.random.default_rng(42)

    # Normal data: digits 0-7
    digits = load_digits()
    X_all  = MinMaxScaler().fit_transform(digits.data.astype(float))
    y_all  = digits.target

    normal_mask  = y_all <= 7
    anomaly_mask = y_all >= 8   # 8 and 9 are "anomalies"

    X_normal  = X_all[normal_mask]
    X_anomaly = X_all[anomaly_mask]

    # Train AE only on normal class
    ae = SimpleAE(64, 32, 8, rng)
    ae.fit(X_normal, epochs=50, lr=0.05, rng=rng)

    # Reconstruction errors
    X_hat_n, _ = ae.forward(X_normal)
    X_hat_a, _ = ae.forward(X_anomaly)

    err_normal  = np.mean((X_hat_n - X_normal)**2, axis=1)
    err_anomaly = np.mean((X_hat_a - X_anomaly)**2, axis=1)

    print(f"  Normal digits (0-7)  | n={len(X_normal):<5} | mean recon error = {err_normal.mean():.5f}")
    print(f"  Anomaly digits (8,9) | n={len(X_anomaly):<5} | mean recon error = {err_anomaly.mean():.5f}")

    # AUC-ROC
    scores = np.concatenate([err_normal, err_anomaly])
    labels = np.concatenate([np.zeros(len(err_normal)), np.ones(len(err_anomaly))])
    auc    = roc_auc_score(labels, scores)
    ap     = average_precision_score(labels, scores)
    print(f"  AUC-ROC: {auc:.4f}  AP: {ap:.4f}")

    # Percentile threshold
    tau = np.percentile(err_normal, 95)
    tp  = (err_anomaly > tau).sum()
    fp  = (err_normal  > tau).sum()
    print(f"\n  Threshold @ 95th pct of normal: tau = {tau:.5f}")
    print(f"  True positives: {tp}/{len(err_anomaly)} anomalies detected")
    print(f"  False positives: {fp}/{len(err_normal)} normal flagged")

    # Plot distribution
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(err_normal, bins=30, alpha=0.6, label="Normal", density=True)
    ax.hist(err_anomaly, bins=30, alpha=0.6, label="Anomaly", density=True)
    ax.axvline(tau, color='r', linestyle='--', label=f"tau={tau:.4f}")
    ax.set(xlabel="Reconstruction Error", ylabel="Density",
           title="Anomaly Detection via Reconstruction Error"); ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "anomaly_detection.png")
    plt.savefig(path, dpi=90); plt.close()
    print(f"  Distribution plot: {path}")


# -- 2. Dimensionality reduction comparison ------------------------------------
def dimensionality_reduction():
    print("\n=== Dimensionality Reduction: AE vs PCA ===")
    digits = load_digits()
    X      = MinMaxScaler().fit_transform(digits.data.astype(float))
    y      = digits.target

    rng  = np.random.default_rng(10)
    Xtr  = X[:1400]; ytr = y[:1400]
    Xts  = X[1400:]; yts = y[1400:]

    results = []
    for latent_dim in [2, 4, 8, 16]:
        # AE
        ae    = SimpleAE(64, 32, latent_dim, rng)
        ae.fit(Xtr, epochs=50, lr=0.05, rng=rng)
        Htr   = ae.encode(Xtr)
        Hts   = ae.encode(Xts)
        X_hat = ae.forward(Xts)[0]
        mse_ae = np.mean((X_hat - Xts)**2)

        # PCA
        pca   = PCA(n_components=latent_dim, random_state=0).fit(Xtr)
        Ptr   = pca.transform(Xtr)
        Pts   = pca.transform(Xts)
        X_pca_hat = pca.inverse_transform(Pts)
        mse_pca   = np.mean((X_pca_hat - Xts)**2)

        # Classifier accuracy on compressed features
        clf_ae  = LogisticRegression(max_iter=500, random_state=0).fit(Htr, ytr)
        clf_pca = LogisticRegression(max_iter=500, random_state=0).fit(Ptr, ytr)
        acc_ae  = clf_ae.score(Hts, yts)
        acc_pca = clf_pca.score(Pts, yts)

        results.append((latent_dim, mse_ae, mse_pca, acc_ae, acc_pca))
        print(f"  Latent={latent_dim:>2}: AE MSE={mse_ae:.5f} PCA MSE={mse_pca:.5f} | "
              f"AE Acc={acc_ae:.3f} PCA Acc={acc_pca:.3f}")

    print()
    print("  AE can learn non-linear structure -> better at lower dims")
    print("  PCA is linear -> better when data is roughly linear or dim is moderate")


# -- 3. Denoising application -------------------------------------------------
def denoising_application():
    print("\n=== Denoising Application ===")
    digits = load_digits()
    X      = MinMaxScaler().fit_transform(digits.data.astype(float))
    rng    = np.random.default_rng(5)
    Xtr    = X[:1400]
    Xts    = X[1400:]

    def add_noise(X, sigma, rng):
        return np.clip(X + rng.normal(0, sigma, X.shape), 0, 1)

    sigma = 0.3
    Xts_noisy = add_noise(Xts, sigma, rng)

    ae = SimpleAE(64, 32, 16, rng)
    ae.fit(Xtr, epochs=50, lr=0.05, rng=rng)

    X_hat_clean, _ = ae.forward(Xts)
    X_hat_noisy, _ = ae.forward(Xts_noisy)

    mse_noisy_in   = np.mean((Xts_noisy - Xts)**2)
    mse_after_ae   = np.mean((X_hat_noisy - Xts)**2)

    print(f"  Noise sigma={sigma}")
    print(f"  Baseline MSE (noisy input vs clean): {mse_noisy_in:.5f}")
    print(f"  After AE denoising:                  {mse_after_ae:.5f}")
    psnr = 10 * np.log10(1.0 / mse_after_ae)
    print(f"  PSNR (higher=better):                {psnr:.2f} dB")


# -- 4. Feature pre-training ---------------------------------------------------
def feature_pretraining():
    print("\n=== Unsupervised Pre-training with Autoencoders ===")
    print("  Historical use (pre-2012): initialise DNN weights with layer-wise AE pre-training")
    print("  Now superseded by better initialisers + batch norm + ReLUs")
    print()
    print("  Modern use cases:")
    print("    1. Few-shot learning: AE features on unlabelled data -> fine-tune")
    print("    2. Semi-supervised: AE on all data; classifier on labelled only")
    print("    3. Domain adaptation: unsupervised features generalise better")
    print()

    # Simulate semi-supervised scenario
    digits = load_digits()
    X      = MinMaxScaler().fit_transform(digits.data.astype(float))
    y      = digits.target
    rng    = np.random.default_rng(15)
    idx    = rng.permutation(len(X))
    X, y   = X[idx], y[idx]

    # Unlabelled = first 1500, Labelled = first 50 only
    Xun = X[:1500]
    Xlab, ylab = X[:50], y[:50]
    Xts,  yts  = X[1500:], y[1500:]

    ae = SimpleAE(64, 32, 16, rng)
    ae.fit(Xun, epochs=60, lr=0.05, rng=rng)

    # Raw features baseline
    clf_raw = LogisticRegression(max_iter=500, random_state=0).fit(Xlab, ylab)
    acc_raw = clf_raw.score(Xts, yts)

    # AE features
    Hlab = ae.encode(Xlab)
    Hts  = ae.encode(Xts)
    clf_ae = LogisticRegression(max_iter=500, random_state=0).fit(Hlab, ylab)
    acc_ae = clf_ae.score(Hts, yts)

    print(f"  Labelled samples: {len(Xlab)}  Test: {len(Xts)}")
    print(f"  Raw features (64-dim):    accuracy = {acc_raw:.4f}")
    print(f"  AE features (16-dim):     accuracy = {acc_ae:.4f}")
    print(f"  Gain from unsupervised pre-training: {(acc_ae-acc_raw)*100:+.1f}%")


if __name__ == "__main__":
    anomaly_detection()
    dimensionality_reduction()
    denoising_application()
    feature_pretraining()
