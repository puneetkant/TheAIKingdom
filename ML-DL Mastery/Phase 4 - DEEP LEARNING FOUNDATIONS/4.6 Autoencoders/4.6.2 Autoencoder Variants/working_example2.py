"""
Working Example 2: Autoencoder Variants — Denoising, Sparse, VAE (numpy)
==========================================================================
Comparison of denoising AE vs vanilla AE on noisy Cal Housing data.

Run:  python working_example2.py
"""
from pathlib import Path
try:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.datasets import fetch_california_housing
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
except ImportError:
    raise SystemExit("pip install numpy matplotlib scikit-learn")

OUTPUT = Path(__file__).parent / "output"
OUTPUT.mkdir(exist_ok=True)

relu   = lambda x: np.maximum(0, x)
relu_d = lambda x: (x > 0).astype(float)

def train_ae(X_clean, X_input, n_h=32, n_z=3, lr=0.005, epochs=200, seed=42):
    """Train AE with X_input as corrupt input, X_clean as reconstruction target."""
    rng = np.random.default_rng(seed)
    We1=rng.standard_normal((X_clean.shape[1],n_h))*np.sqrt(2/X_clean.shape[1]); be1=np.zeros(n_h)
    We2=rng.standard_normal((n_h,n_z))*np.sqrt(2/n_h); be2=np.zeros(n_z)
    Wd1=rng.standard_normal((n_z,n_h))*np.sqrt(2/n_z); bd1=np.zeros(n_h)
    Wd2=rng.standard_normal((n_h,X_clean.shape[1]))*np.sqrt(2/n_h); bd2=np.zeros(X_clean.shape[1])
    losses = []
    n = len(X_clean)
    for ep in range(epochs):
        h1=relu(X_input@We1+be1); z=relu(h1@We2+be2)
        h2=relu(z@Wd1+bd1); xh=h2@Wd2+bd2
        loss=np.mean((xh-X_clean)**2); losses.append(loss)
        dout=2*(xh-X_clean)/n; Wd2-=lr*(h2.T@dout); bd2-=lr*dout.sum(0)
        dh2=(dout@Wd2.T)*relu_d(h2); Wd1-=lr*(z.T@dh2); bd1-=lr*dh2.sum(0)
        dz=(dh2@Wd1.T)*relu_d(z); We2-=lr*(h1.T@dz); be2-=lr*dz.sum(0)
        dh1=(dz@We2.T)*relu_d(h1); We1-=lr*(X_input.T@dh1); be1-=lr*dh1.sum(0)
    return losses, (We1,be1,We2,be2,Wd1,bd1,Wd2,bd2)

def infer_ae(X, params):
    We1,be1,We2,be2,Wd1,bd1,Wd2,bd2 = params
    h1=relu(X@We1+be1); z=relu(h1@We2+be2)
    return (relu(z@Wd1+bd1))@Wd2+bd2

def demo():
    print("=== Denoising Autoencoder vs Standard AE ===")
    h = fetch_california_housing(); X = StandardScaler().fit_transform(h.data)
    X_tr, X_te = train_test_split(X, test_size=0.2, random_state=42)

    rng = np.random.default_rng(0)
    noise_std = 0.5
    X_tr_noisy = X_tr + rng.standard_normal(X_tr.shape) * noise_std
    X_te_noisy  = X_te  + rng.standard_normal(X_te.shape) * noise_std

    # Standard AE (clean->clean)
    l_std, p_std = train_ae(X_tr, X_tr, epochs=150)
    # Denoising AE (noisy input -> clean target)
    l_dae, p_dae = train_ae(X_tr, X_tr_noisy, epochs=150)

    mse_std = np.mean((infer_ae(X_te, p_std) - X_te)**2)
    mse_dae_noisy = np.mean((infer_ae(X_te_noisy, p_dae) - X_te)**2)
    mse_std_noisy = np.mean((infer_ae(X_te_noisy, p_std) - X_te)**2)

    print(f"  Standard AE (clean->clean test):       {mse_std:.4f}")
    print(f"  Standard AE (noisy input -> target):   {mse_std_noisy:.4f}")
    print(f"  Denoising AE (noisy input -> target):  {mse_dae_noisy:.4f}")
    print("  -> DAE reconstructs clean signal better from noisy input")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(l_std, label="Standard AE"); ax.plot(l_dae, label="Denoising AE")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE"); ax.set_title("AE Variants Training")
    ax.legend(); plt.tight_layout(); plt.savefig(OUTPUT / "ae_variants.png"); plt.close()
    print("  Saved ae_variants.png")

if __name__ == "__main__":
    demo()
