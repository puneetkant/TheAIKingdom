"""
Working Example 2: Autoencoder Applications — anomaly detection, dimensionality reduction
===========================================================================================
Uses reconstruction error for anomaly detection on California Housing.

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
    from sklearn.metrics import roc_auc_score
except ImportError:
    raise SystemExit("pip install numpy matplotlib scikit-learn")

OUTPUT = Path(__file__).parent / "output"
OUTPUT.mkdir(exist_ok=True)

relu   = lambda x: np.maximum(0, x)
relu_d = lambda x: (x > 0).astype(float)

def train_ae_simple(X, n_z=4, lr=0.005, epochs=200, seed=42):
    rng = np.random.default_rng(seed); n_in = X.shape[1]; n_h = 32
    We1=rng.standard_normal((n_in,n_h))*np.sqrt(2/n_in); be1=np.zeros(n_h)
    We2=rng.standard_normal((n_h,n_z))*np.sqrt(2/n_h); be2=np.zeros(n_z)
    Wd1=rng.standard_normal((n_z,n_h))*np.sqrt(2/n_z); bd1=np.zeros(n_h)
    Wd2=rng.standard_normal((n_h,n_in))*np.sqrt(2/n_h); bd2=np.zeros(n_in)
    n = len(X)
    for ep in range(epochs):
        h1=relu(X@We1+be1); z=relu(h1@We2+be2); h2=relu(z@Wd1+bd1); xh=h2@Wd2+bd2
        dout=2*(xh-X)/n; Wd2-=lr*(h2.T@dout); bd2-=lr*dout.sum(0)
        dh2=(dout@Wd2.T)*relu_d(h2); Wd1-=lr*(z.T@dh2); bd1-=lr*dh2.sum(0)
        dz=(dh2@Wd1.T)*relu_d(z); We2-=lr*(h1.T@dz); be2-=lr*dz.sum(0)
        dh1=(dz@We2.T)*relu_d(h1); We1-=lr*(X.T@dh1); be1-=lr*dh1.sum(0)
    return (We1,be1,We2,be2,Wd1,bd1,Wd2,bd2)

def reconstruct(X, params):
    We1,be1,We2,be2,Wd1,bd1,Wd2,bd2 = params
    h1=relu(X@We1+be1); z=relu(h1@We2+be2); return (relu(z@Wd1+bd1))@Wd2+bd2

def demo():
    print("=== Autoencoder Anomaly Detection ===")
    h = fetch_california_housing(); X = StandardScaler().fit_transform(h.data)
    # "Normal" = house price below 75th percentile (train on normal only)
    thr = np.percentile(h.target, 75)
    X_normal = X[h.target < thr]; X_anomaly = X[h.target >= thr]
    print(f"  Normal: {len(X_normal)}  Anomaly: {len(X_anomaly)}")

    # Train AE only on normal data
    params = train_ae_simple(X_normal, n_z=4, epochs=150)

    # Reconstruction error per sample
    x_hat_n = reconstruct(X_normal, params); x_hat_a = reconstruct(X_anomaly, params)
    err_n = np.mean((x_hat_n - X_normal)**2, axis=1)
    err_a = np.mean((x_hat_a - X_anomaly)**2, axis=1)

    # ROC-AUC: anomaly should have higher error
    n = min(len(err_n), len(err_a))
    scores = np.concatenate([err_n[:n], err_a[:n]])
    labels = np.concatenate([np.zeros(n), np.ones(n)])
    auc = roc_auc_score(labels, scores)
    print(f"  Normal mean error: {err_n.mean():.4f}")
    print(f"  Anomaly mean error: {err_a.mean():.4f}")
    print(f"  ROC-AUC (anomaly detection): {auc:.4f}")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(err_n, bins=40, alpha=0.5, label="Normal", density=True)
    ax.hist(err_a, bins=40, alpha=0.5, label="Anomaly", density=True)
    ax.set_xlabel("Reconstruction Error"); ax.set_title("AE Anomaly Detection")
    ax.legend(); plt.tight_layout(); plt.savefig(OUTPUT / "ae_anomaly.png"); plt.close()
    print("  Saved ae_anomaly.png")

if __name__ == "__main__":
    demo()
