"""
Working Example 2: Anomaly Detection — Isolation Forest, One-Class SVM, LOF
============================================================================
Unsupervised anomaly detection on Cal Housing (price extremes as anomalies).

Run:  python working_example2.py
"""
from pathlib import Path
try:
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.datasets import fetch_california_housing
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.svm import OneClassSVM
    from sklearn.metrics import precision_score, recall_score
except ImportError:
    raise SystemExit("pip install numpy matplotlib scikit-learn")

OUTPUT = Path(__file__).parent / "output"
OUTPUT.mkdir(exist_ok=True)

def load_data():
    h = fetch_california_housing()
    X = StandardScaler().fit_transform(h.data)
    # Create pseudo ground-truth anomalies: top/bottom 5% prices
    y_true = np.zeros(len(h.target), dtype=int)
    threshold_high = np.percentile(h.target, 95)
    threshold_low  = np.percentile(h.target, 5)
    y_true[(h.target >= threshold_high) | (h.target <= threshold_low)] = 1
    print(f"  Total anomalies (top/bot 5%): {y_true.sum()} / {len(y_true)}")
    return X, y_true

def demo_isolation_forest(X, y_true):
    print("\n=== Isolation Forest ===")
    for contamination in [0.05, 0.1, 0.15]:
        iso = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
        pred = iso.fit_predict(X)
        anomalies = (pred == -1).astype(int)
        prec = precision_score(y_true, anomalies, zero_division=0)
        rec  = recall_score(y_true, anomalies, zero_division=0)
        print(f"  contamination={contamination}: precision={prec:.4f}  recall={rec:.4f}")

def demo_lof(X, y_true):
    print("\n=== Local Outlier Factor ===")
    for k in [10, 20, 50]:
        lof = LocalOutlierFactor(n_neighbors=k, contamination=0.1)
        pred = lof.fit_predict(X)
        anomalies = (pred == -1).astype(int)
        prec = precision_score(y_true, anomalies, zero_division=0)
        rec  = recall_score(y_true, anomalies, zero_division=0)
        print(f"  n_neighbors={k}: precision={prec:.4f}  recall={rec:.4f}")

def demo_one_class_svm(X, y_true):
    print("\n=== One-Class SVM ===")
    oc = OneClassSVM(nu=0.05, kernel="rbf", gamma="scale")
    pred = oc.fit_predict(X[:5000])   # subset for speed
    anomalies = (pred == -1).astype(int)
    prec = precision_score(y_true[:5000], anomalies, zero_division=0)
    rec  = recall_score(y_true[:5000], anomalies, zero_division=0)
    print(f"  OC-SVM (nu=0.05): precision={prec:.4f}  recall={rec:.4f}")

def demo_elliptic_envelope(X, y_true):
    """Elliptic Envelope assumes Gaussian data; fits a robust covariance ellipse."""
    print("\n=== Elliptic Envelope ===")
    from sklearn.covariance import EllipticEnvelope
    for contamination in [0.05, 0.10]:
        ee = EllipticEnvelope(contamination=contamination, random_state=42)
        pred = ee.fit_predict(X)
        anomalies = (pred == -1).astype(int)
        prec = precision_score(y_true, anomalies, zero_division=0)
        rec  = recall_score(y_true, anomalies, zero_division=0)
        print(f"  contamination={contamination}: precision={prec:.4f}  recall={rec:.4f}")


def demo_anomaly_scores(X, y_true):
    """Compare anomaly score distributions between normal and anomalous points."""
    print("\n=== Anomaly Score Distribution ===")
    iso = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
    iso.fit(X)
    scores = iso.decision_function(X)   # higher = more normal
    normal_scores   = scores[y_true == 0]
    anomaly_scores  = scores[y_true == 1]
    print(f"  Normal   scores: mean={normal_scores.mean():.4f}  std={normal_scores.std():.4f}")
    print(f"  Anomaly  scores: mean={anomaly_scores.mean():.4f}  std={anomaly_scores.std():.4f}")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(normal_scores,  bins=60, alpha=0.6, label="Normal",  density=True)
    ax.hist(anomaly_scores, bins=60, alpha=0.6, label="Anomaly", density=True)
    ax.set_xlabel("Isolation Forest score"); ax.set_title("Score distributions")
    ax.legend()
    fig.savefig(OUTPUT / "anomaly_scores.png", dpi=120, bbox_inches="tight")
    plt.close(fig); print("  Saved: anomaly_scores.png")


def demo_rolling_anomaly():
    """Sliding-window Z-score anomaly detection on a synthetic time series."""
    print("\n=== Rolling Z-Score (time series anomaly) ===")
    np.random.seed(0)
    n = 200
    t = np.arange(n)
    signal = np.sin(2 * np.pi * t / 20) + np.random.normal(0, 0.2, n)
    # inject spikes
    spike_idx = [40, 80, 120, 160]
    signal[spike_idx] += np.array([4, -4, 5, -5])
    window = 20
    flags = []
    for i in range(n):
        start = max(0, i - window)
        w = signal[start:i] if i > 0 else signal[:1]
        z = abs(signal[i] - w.mean()) / (w.std() + 1e-9)
        flags.append(z > 3.0)
    detected = [i for i, f in enumerate(flags) if f]
    print(f"  Injected spikes at: {spike_idx}")
    print(f"  Detected anomalies: {detected}")
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(t, signal, lw=1, label="signal")
    ax.scatter(detected, signal[detected], color="red", zorder=5, label="detected")
    ax.set_title("Rolling Z-Score Anomaly Detection")
    ax.legend()
    fig.savefig(OUTPUT / "anomaly_timeseries.png", dpi=120, bbox_inches="tight")
    plt.close(fig); print("  Saved: anomaly_timeseries.png")


if __name__ == "__main__":
    X, y_true = load_data()
    demo_isolation_forest(X, y_true)
    demo_lof(X, y_true)
    demo_one_class_svm(X, y_true)
    demo_elliptic_envelope(X, y_true)
    demo_anomaly_scores(X, y_true)
    demo_rolling_anomaly()
