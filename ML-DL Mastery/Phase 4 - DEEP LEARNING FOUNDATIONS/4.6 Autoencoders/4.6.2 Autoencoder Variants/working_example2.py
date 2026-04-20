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

def demo_sparse_ae():
    """Sparse autoencoder: add L1 penalty on bottleneck activations."""
    print("\n=== Sparse Autoencoder ===")
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split
    h = fetch_california_housing(); X = StandardScaler().fit_transform(h.data)
    X_tr, X_te = train_test_split(X, test_size=0.2, random_state=42)

    rng = np.random.default_rng(99); n_in, n_h, n_z = X_tr.shape[1], 32, 6
    We1=rng.standard_normal((n_in,n_h))*np.sqrt(2/n_in); be1=np.zeros(n_h)
    We2=rng.standard_normal((n_h,n_z))*np.sqrt(2/n_h); be2=np.zeros(n_z)
    Wd1=rng.standard_normal((n_z,n_h))*np.sqrt(2/n_z); bd1=np.zeros(n_h)
    Wd2=rng.standard_normal((n_h,n_in))*np.sqrt(2/n_h); bd2=np.zeros(n_in)

    l1_lambda = 0.01
    losses = []
    n = len(X_tr)
    for ep in range(200):
        h1=relu(X_tr@We1+be1); z=relu(h1@We2+be2)
        h2=relu(z@Wd1+bd1); xh=h2@Wd2+bd2
        recon = np.mean((xh-X_tr)**2)
        sparsity = l1_lambda * np.mean(np.abs(z))
        loss = recon + sparsity; losses.append(loss)
        dout=2*(xh-X_tr)/n; Wd2-=0.005*(h2.T@dout); bd2-=0.005*dout.sum(0)
        dh2=(dout@Wd2.T)*relu_d(h2); Wd1-=0.005*(z.T@dh2); bd1-=0.005*dh2.sum(0)
        dz=(dh2@Wd1.T)*relu_d(z) + l1_lambda*np.sign(z)/n
        We2-=0.005*(h1.T@dz); be2-=0.005*dz.sum(0)
        dh1=(dz@We2.T)*relu_d(h1); We1-=0.005*(X_tr.T@dh1); be1-=0.005*dh1.sum(0)

    # Measure sparsity of bottleneck
    h1_te=relu(X_te@We1+be1); z_te=relu(h1_te@We2+be2)
    sparsity_frac = (np.abs(z_te) < 0.01).mean()
    print(f"  Final loss: {losses[-1]:.4f}  Bottleneck sparsity: {sparsity_frac:.3f} (fraction near-zero)")


def demo_latent_interpolation():
    """Visualise latent space by interpolating between two data points."""
    print("\n=== Latent Space Interpolation ===")
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split
    h = fetch_california_housing(); X = StandardScaler().fit_transform(h.data)
    X_tr, X_te = train_test_split(X, test_size=0.2, random_state=42)

    _, params = train_ae(X_tr, X_tr, n_h=32, n_z=3, epochs=100)
    We1,be1,We2,be2,Wd1,bd1,Wd2,bd2 = params

    def encode(x): return relu(relu(x@We1+be1)@We2+be2)
    def decode(z): return relu(z@Wd1+bd1)@Wd2+bd2

    z0 = encode(X_te[0:1]); z1 = encode(X_te[1:2])
    print(f"  Latent A: {z0.round(3)}")
    print(f"  Latent B: {z1.round(3)}")
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        z_interp = (1 - alpha) * z0 + alpha * z1
        x_dec = decode(z_interp)
        print(f"    alpha={alpha:.2f}: decoded feature0={x_dec[0,0]:.4f}  feature1={x_dec[0,1]:.4f}")


if __name__ == "__main__":
    demo()
    demo_sparse_ae()
    demo_latent_interpolation()
