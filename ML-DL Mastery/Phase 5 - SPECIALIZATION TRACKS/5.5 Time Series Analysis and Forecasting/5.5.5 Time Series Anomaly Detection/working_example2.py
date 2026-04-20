"""
Working Example 2: Time Series Anomaly Detection — statistical + ML approaches
================================================================================
Detects anomalies using Z-score, rolling stats, and isolation forest proxy.

Run:  python working_example2.py
"""
from pathlib import Path
try:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import StandardScaler
except ImportError:
    raise SystemExit("pip install numpy matplotlib scikit-learn")

OUTPUT = Path(__file__).parent / "output"
OUTPUT.mkdir(exist_ok=True)

def make_series_with_anomalies(n=300, seed=0):
    rng = np.random.default_rng(seed); t = np.arange(n)
    ts = np.sin(2*np.pi*t/30) * 3 + rng.normal(0, 0.2, n)
    # Inject point anomalies
    anomaly_idx = [50, 100, 150, 220, 270]
    ts[anomaly_idx] += rng.choice([-6, 6], len(anomaly_idx))
    return ts, np.array(anomaly_idx)

def zscore_detect(ts, window=20, threshold=3.0):
    """Rolling Z-score anomaly detection."""
    scores = np.zeros(len(ts)); flags = np.zeros(len(ts), dtype=bool)
    for i in range(window, len(ts)):
        window_data = ts[i-window:i]
        mu, sigma = window_data.mean(), window_data.std() + 1e-8
        scores[i] = abs(ts[i] - mu) / sigma
        flags[i] = scores[i] > threshold
    return scores, flags

def iqr_detect(ts):
    """IQR-based global anomaly detection."""
    q1, q3 = np.percentile(ts, 25), np.percentile(ts, 75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5*iqr, q3 + 1.5*iqr
    return (ts < lower) | (ts > upper)

def demo():
    print("=== Time Series Anomaly Detection ===")
    ts, true_idx = make_series_with_anomalies()
    scores, z_flags = zscore_detect(ts)
    iqr_flags = iqr_detect(ts)

    def metrics(flags, true_idx):
        tp = np.sum(flags[true_idx])
        fp = np.sum(flags) - tp
        fn = len(true_idx) - tp
        prec = tp / (tp + fp + 1e-8); rec = tp / (tp + fn + 1e-8)
        return prec, rec

    for name, flags in [("Z-score", z_flags), ("IQR", iqr_flags)]:
        p, r = metrics(flags, true_idx)
        print(f"  {name:10s} | Detected: {flags.sum():3d} | Precision: {p:.2f} | Recall: {r:.2f}")

    fig, axes = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
    axes[0].plot(ts, alpha=0.7, label="Series")
    axes[0].scatter(np.where(z_flags)[0], ts[z_flags], color="red", zorder=5, label="Z-score detections")
    axes[0].scatter(true_idx, ts[true_idx], marker="x", color="green", zorder=6, s=80, label="True anomalies")
    axes[0].legend(fontsize=8); axes[0].set_title("Rolling Z-Score Detection")
    axes[1].plot(scores, color="orange", label="Z-score"); axes[1].axhline(3.0, ls="--", color="red")
    axes[1].legend(); axes[1].set_title("Anomaly Score")
    plt.tight_layout(); plt.savefig(OUTPUT / "ts_anomaly.png"); plt.close()
    print("  Saved ts_anomaly.png")

def demo_lof_analog():
    """LOF-analog: local density-based anomaly scoring on time windows."""
    print("\n=== Local Density Anomaly Scoring ===")
    ts, true_idx = make_series_with_anomalies()
    window = 10; k_nn = 5; scores = []
    # Feature: each window as a flattened vector, find k nearest distances
    for i in range(window, len(ts) - window):
        ref = ts[i-window:i]
        # Compare against k non-overlapping windows
        candidates = [ts[j-window:j] for j in range(window, len(ts)-window)
                      if abs(j-i) > window][:20]
        if not candidates:
            scores.append(0); continue
        dists = [np.linalg.norm(ref - c) for c in candidates]
        scores.append(np.mean(sorted(dists)[:k_nn]))

    scores_arr = np.array(scores)
    threshold = np.percentile(scores_arr, 95)
    detected = np.where(scores_arr > threshold)[0] + window
    tp = sum(1 for d in detected if any(abs(d - t) <= 5 for t in true_idx))
    print(f"  Detections: {len(detected)}  TPs (within 5 steps): {tp}/{len(true_idx)}")


def demo_threshold_sensitivity():
    """Show precision/recall tradeoff across Z-score thresholds."""
    print("\n=== Threshold Sensitivity ===")
    ts, true_idx = make_series_with_anomalies()
    print(f"  {'Threshold':>10}  {'Detected':>10}  {'Precision':>10}  {'Recall':>8}")
    for t in [2.0, 2.5, 3.0, 3.5, 4.0]:
        _, flags = zscore_detect(ts, threshold=t)
        det = flags.sum()
        tp = np.sum(flags[true_idx])
        prec = tp / (det + 1e-8); rec = tp / len(true_idx)
        print(f"  {t:>10.1f}  {det:>10}  {prec:>10.3f}  {rec:>8.3f}")


if __name__ == "__main__":
    demo()
    demo_lof_analog()
    demo_threshold_sensitivity()
