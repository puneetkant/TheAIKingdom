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

def demo_dimensionality_reduction_ae():
    """Compare AE latent space with PCA for dimensionality reduction."""
    print("\n=== AE vs PCA for Dimensionality Reduction ===")
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import load_wine
    from sklearn.metrics import accuracy_score

    wine = load_wine()
    X_all = StandardScaler().fit_transform(wine.data)
    y_all = wine.target
    X_tr, X_te, y_tr, y_te = train_test_split(X_all, y_all, test_size=0.2, random_state=42)

    # PCA 4-component
    pca = PCA(n_components=4); Xp_tr = pca.fit_transform(X_tr); Xp_te = pca.transform(X_te)
    lr_pca = LogisticRegression(max_iter=500, random_state=42).fit(Xp_tr, y_tr)
    acc_pca = accuracy_score(y_te, lr_pca.predict(Xp_te))

    # AE 4-dim bottleneck
    params = train_ae_simple(X_tr, n_z=4, epochs=200)
    We1,be1,We2,be2,_,_,_,_ = params
    Xa_tr = relu(relu(X_tr@We1+be1)@We2+be2)
    Xa_te = relu(relu(X_te@We1+be1)@We2+be2)
    lr_ae = LogisticRegression(max_iter=500, random_state=42).fit(Xa_tr, y_tr)
    acc_ae = accuracy_score(y_te, lr_ae.predict(Xa_te))

    print(f"  PCA (4 components) downstream accuracy:  {acc_pca:.4f}")
    print(f"  AE  (4-dim latent) downstream accuracy:  {acc_ae:.4f}")


def demo_reconstruction_feature_analysis():
    """Examine which features the AE reconstructs well vs poorly."""
    print("\n=== Feature-wise Reconstruction Error ===")
    h = fetch_california_housing(); X = StandardScaler().fit_transform(h.data)
    X_tr, X_te = train_test_split(X, test_size=0.2, random_state=42)
    params = train_ae_simple(X_tr, n_z=4, epochs=200)
    x_hat = reconstruct(X_te, params)
    feat_errors = np.mean((x_hat - X_te)**2, axis=0)
    feature_names = h.feature_names
    print(f"  {'Feature':15s}  {'MSE':>8s}")
    for name, err in sorted(zip(feature_names, feat_errors), key=lambda x: x[1], reverse=True):
        print(f"  {name:15s}  {err:8.4f}")


if __name__ == "__main__":
    demo()
    demo_dimensionality_reduction_ae()
    demo_reconstruction_feature_analysis()
