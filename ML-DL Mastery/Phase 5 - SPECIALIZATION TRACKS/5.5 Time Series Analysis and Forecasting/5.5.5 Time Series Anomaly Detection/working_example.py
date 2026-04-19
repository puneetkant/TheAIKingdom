"""
Working Example: Time Series Anomaly Detection
Covers statistical, distance-based, ML, and DL approaches
for detecting anomalies in time series.
"""
import numpy as np
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_ts_anomaly")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -- Helper: synthetic anomalous series ---------------------------------------
def gen_anomalous_series(T=300, seed=0):
    rng = np.random.default_rng(seed)
    t   = np.arange(T)
    y   = (5 * np.sin(2*np.pi*t/50) + rng.normal(0, 0.5, T))
    labels = np.zeros(T, dtype=int)
    # Inject point anomalies
    for idx in [60, 150, 240]:
        y[idx] += rng.choice([-6, 6])
        labels[idx] = 1
    # Inject contextual anomaly (level shift)
    y[180:200] += 4
    labels[180:200] = 1
    return y, labels


# -- 1. Anomaly types ----------------------------------------------------------
def anomaly_types():
    print("=== Time Series Anomaly Detection ===")
    print()
    print("  Anomaly types:")
    types = [
        ("Point",       "Single outlier far from surrounding values"),
        ("Contextual",  "Normal value globally, but anomalous in context (e.g. level shift)"),
        ("Collective",  "Group of points anomalous together (e.g. missing seasonality)"),
        ("Seasonal",    "Missing or distorted seasonal pattern"),
        ("Trend",       "Unexpected trend change or break"),
    ]
    for t, d in types:
        print(f"  {t:<12} {d}")
    print()
    print("  Applications:")
    apps = [
        "Network intrusion detection",
        "IoT sensor fault detection",
        "Financial fraud / market anomalies",
        "Manufacturing equipment failure prediction",
        "Cloud infrastructure monitoring",
    ]
    for a in apps:
        print(f"  • {a}")


# -- 2. Statistical methods ----------------------------------------------------
def statistical_methods():
    print("\n=== Statistical Anomaly Detection Methods ===")
    y, labels = gen_anomalous_series()
    T = len(y)

    # 1. Z-score
    def z_score(y, threshold=3.0):
        mu, sigma = y.mean(), y.std()
        z = np.abs((y - mu) / sigma)
        return z > threshold, z

    flags_z, z = z_score(y)
    tp = (flags_z & labels).sum(); fp = (flags_z & ~labels).sum()
    fn = (~flags_z & labels).sum()
    prec = tp/(tp+fp+1e-9); rec = tp/(tp+fn+1e-9)
    print(f"  Z-score (|z|>3): flagged={flags_z.sum()}  TP={tp} FP={fp} FN={fn}"
          f"  P={prec:.2f} R={rec:.2f}")

    # 2. Rolling Z-score
    window = 20
    def rolling_z(y, window=20, threshold=3.0):
        flags = np.zeros(T, dtype=bool)
        zs    = np.zeros(T)
        for t in range(window, T):
            window_vals = y[t-window:t]
            mu, sigma   = window_vals.mean(), window_vals.std()
            if sigma > 0:
                zs[t] = abs((y[t] - mu) / sigma)
                flags[t] = zs[t] > threshold
        return flags, zs

    flags_rz, _ = rolling_z(y)
    tp = (flags_rz & labels).sum(); fp = (flags_rz & ~labels).sum()
    fn = (~flags_rz & labels).sum()
    prec = tp/(tp+fp+1e-9); rec = tp/(tp+fn+1e-9)
    print(f"  Rolling Z-score (w={window}):   flagged={flags_rz.sum()}  TP={tp} FP={fp} FN={fn}"
          f"  P={prec:.2f} R={rec:.2f}")

    # 3. IQR method
    Q1, Q3 = np.percentile(y, 25), np.percentile(y, 75)
    IQR     = Q3 - Q1
    flags_iqr = (y < Q1 - 1.5*IQR) | (y > Q3 + 1.5*IQR)
    tp = (flags_iqr & labels).sum(); fp = (flags_iqr & ~labels).sum()
    fn = (~flags_iqr & labels).sum()
    prec = tp/(tp+fp+1e-9); rec = tp/(tp+fn+1e-9)
    print(f"  IQR (±1.5×IQR):         flagged={flags_iqr.sum()}  TP={tp} FP={fp} FN={fn}"
          f"  P={prec:.2f} R={rec:.2f}")

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(y, alpha=0.7, label="Series")
    axes[0].scatter(np.where(flags_rz)[0], y[flags_rz], c='r', s=40, zorder=5, label="Detected")
    axes[0].scatter(np.where(labels)[0],   y[labels.astype(bool)], c='g', s=20,
                    marker='^', zorder=4, label="True")
    axes[0].legend(); axes[0].set_title("Rolling Z-Score Detection")
    axes[1].scatter(range(T), z, s=5, alpha=0.4)
    axes[1].axhline(3, color='r', linestyle='--', label="Threshold=3")
    axes[1].legend(); axes[1].set_ylabel("z-score")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "statistical_detection.png")
    plt.savefig(path, dpi=90); plt.close()
    print(f"  Plot: {path}")


# -- 3. ML methods -------------------------------------------------------------
def ml_methods():
    print("\n=== ML-Based Anomaly Detection ===")
    y, labels = gen_anomalous_series()
    T = len(y)

    print("  Isolation Forest:")
    print("    Build random trees; anomalies have shorter average path length")
    print("    Score = 2^{-E[h(x)] / c(n)}")
    try:
        from sklearn.ensemble import IsolationForest
        window = 10
        X = np.array([y[i:i+window] for i in range(T-window)])
        lbl = labels[window:]
        clf = IsolationForest(contamination=0.1, random_state=0)
        scores = -clf.fit(X).decision_function(X)
        threshold = np.percentile(scores, 90)
        flags = scores > threshold
        tp = (flags & lbl.astype(bool)).sum()
        fp = (flags & ~lbl.astype(bool)).sum()
        fn = (~flags & lbl.astype(bool)).sum()
        prec = tp/(tp+fp+1e-9); rec = tp/(tp+fn+1e-9)
        print(f"    TP={tp} FP={fp} FN={fn}  P={prec:.2f} R={rec:.2f}")
    except ImportError:
        print("    (sklearn not available — code pattern):")
        print("    from sklearn.ensemble import IsolationForest")
        print("    clf = IsolationForest(contamination=0.05)")
        print("    clf.fit(X_train); scores = clf.predict(X_test)")

    print()
    print("  Local Outlier Factor (LOF):")
    print("    LOF_k(p) = E[lrd_k(o)] / lrd_k(p)  for o in N_k(p)")
    print("    LOF >> 1: outlier;  LOF ~= 1: inlier")

    print()
    print("  One-Class SVM:")
    print("    Finds hyperplane separating inliers from origin")
    print("    Kernel trick for non-linear boundaries")
    print()
    print("  Comparison:")
    comparisons = [
        ("Isolation Forest", "Fast; scalable; global anomalies"),
        ("LOF",              "Density-based; local anomalies; slow on large data"),
        ("One-Class SVM",    "Good for low-dim; slow with many data points"),
        ("DBSCAN",           "Clusters + noise points; shape-free"),
    ]
    for m, d in comparisons:
        print(f"  {m:<18} {d}")


# -- 4. Deep learning methods --------------------------------------------------
def dl_methods():
    print("\n=== Deep Learning Anomaly Detection ===")
    print()
    print("  Autoencoder-based:")
    print("    Train on normal data -> high reconstruction error = anomaly")
    print("    Threshold on MSE or Mahalanobis distance in latent space")
    print()

    # Toy autoencoder anomaly detection
    rng    = np.random.default_rng(0)
    T      = 200; L = 10
    normal = rng.normal(0, 1, (T, L))   # normal windows
    anomal = rng.normal(5, 1, (20, L))  # anomalous windows

    # Simulate latent space: PCA-like compression to dim=2
    U, S, Vt = np.linalg.svd(normal, full_matrices=False)
    W = Vt[:2].T   # project to 2D
    z_norm  = normal  @ W
    z_anom  = anomal  @ W

    # Reconstruct
    def reconstruct(z, W):  return z @ W.T

    rec_norm = reconstruct(z_norm, W)
    rec_anom = reconstruct(z_anom, W)
    err_norm = ((normal - rec_norm)**2).mean(axis=1)
    err_anom = ((anomal - rec_anom)**2).mean(axis=1)

    print(f"  Autoencoder (SVD approximation, dim=2):")
    print(f"    Normal recon MSE: {err_norm.mean():.4f} ± {err_norm.std():.4f}")
    print(f"    Anomaly recon MSE: {err_anom.mean():.4f} ± {err_anom.std():.4f}")
    threshold = np.percentile(err_norm, 95)
    flagged = (err_anom > threshold).sum()
    print(f"    Threshold (95th pct): {threshold:.4f}")
    print(f"    Anomalies flagged: {flagged}/{len(anomal)}")

    print()
    print("  Other DL approaches:")
    methods = [
        ("LSTM-VAE",      "VAE with LSTM encoder; anomaly via ELBO"),
        ("Transformers",  "TranAD; anomaly attention score"),
        ("Normalising flows","Exact log-likelihood; threshold on -log p"),
        ("Diffusion",     "DDPM score; high denoising error = anomaly"),
        ("TimesNet",      "Forecasting error as anomaly score"),
    ]
    for m, d in methods:
        print(f"  {m:<18} {d}")


if __name__ == "__main__":
    anomaly_types()
    statistical_methods()
    ml_methods()
    dl_methods()
