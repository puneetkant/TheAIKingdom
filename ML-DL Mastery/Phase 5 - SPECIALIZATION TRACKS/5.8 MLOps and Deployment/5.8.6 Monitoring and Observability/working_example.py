"""
Working Example: Monitoring and Observability for ML
Covers data drift detection, model performance monitoring,
alerting, and observability dashboards.
"""
import numpy as np
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_monitoring")
os.makedirs(OUTPUT_DIR, exist_ok=True)

matplotlib_ok = False
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib_ok = True
except ImportError:
    pass


# -- 1. Monitoring concepts ----------------------------------------------------
def monitoring_concepts():
    print("=== ML Monitoring and Observability ===")
    print()
    print("  The four golden signals (from Google SRE):")
    signals = [
        ("Latency",      "p50/p95/p99 inference time; tail latency SLA"),
        ("Traffic",      "Requests per second; peak vs off-peak patterns"),
        ("Errors",       "Error rate; 4xx vs 5xx; malformed inputs"),
        ("Saturation",   "GPU/CPU usage; queue depth; memory pressure"),
    ]
    for s, d in signals:
        print(f"  {s:<14} {d}")
    print()
    print("  ML-specific monitoring (beyond SRE):")
    ml_signals = [
        ("Data drift",       "Input distribution shifts over time"),
        ("Concept drift",    "P(Y|X) changes; model accuracy degrades"),
        ("Label drift",      "Output distribution shifts; proxy for concept drift"),
        ("Model staleness",  "Model trained months ago on outdated data"),
        ("Prediction bias",  "Output skewed toward one class unexpectedly"),
        ("Feature skew",     "Training/serving feature mismatch"),
    ]
    for s, d in ml_signals:
        print(f"  {s:<20} {d}")


# -- 2. Data drift detection ---------------------------------------------------
def drift_detection():
    print("\n=== Data Drift Detection ===")
    print()
    rng = np.random.default_rng(0)

    # Reference distribution (training)
    ref = rng.normal(loc=0.0, scale=1.0, size=1000)
    # Current distribution (production — shifted)
    cur = rng.normal(loc=0.8, scale=1.3, size=1000)

    print("  Reference:  mean={:.3f}  std={:.3f}".format(ref.mean(), ref.std()))
    print("  Current:    mean={:.3f}  std={:.3f}".format(cur.mean(), cur.std()))
    print()

    # -- KS Test --------------------------------------------------------------
    try:
        from scipy.stats import ks_2samp
        stat, p_val = ks_2samp(ref, cur)
        print(f"  Kolmogorov-Smirnov test:")
        print(f"    KS statistic:  {stat:.4f}")
        print(f"    p-value:       {p_val:.6f}")
        print(f"    Drift detected (p<0.05): {p_val < 0.05}")
    except ImportError:
        # Manual approximate KS (simplistic)
        bins = np.linspace(-4, 4, 50)
        ref_cdf = np.searchsorted(np.sort(ref), bins) / len(ref)
        cur_cdf = np.searchsorted(np.sort(cur), bins) / len(cur)
        ks_stat = np.max(np.abs(ref_cdf - cur_cdf))
        print(f"  Approx KS stat: {ks_stat:.4f}")

    # -- PSI (Population Stability Index) -------------------------------------
    def psi(expected, actual, n_bins=10):
        breaks = np.percentile(expected, np.linspace(0, 100, n_bins+1))
        breaks[0] -= 1e-9; breaks[-1] += 1e-9
        e_pct = np.histogram(expected, breaks)[0] / len(expected)
        a_pct = np.histogram(actual,   breaks)[0] / len(actual)
        e_pct = np.clip(e_pct, 1e-6, None)
        a_pct = np.clip(a_pct, 1e-6, None)
        return np.sum((a_pct - e_pct) * np.log(a_pct / e_pct))

    psi_val = psi(ref, cur)
    print()
    print(f"  PSI (Population Stability Index):")
    print(f"    PSI = {psi_val:.4f}")
    print(f"    Interpretation: <0.1 stable | 0.1–0.2 warning | >0.2 significant shift")
    print(f"    Status: {'[!] Warning' if psi_val > 0.1 else 'OK'}")

    # -- CUSUM (online drift detection) ---------------------------------------
    print()
    print("  CUSUM (Cumulative Sum) — online drift detection:")
    mu_0 = 0.0; k = 0.5
    cusum = 0.0
    h = 5.0  # threshold
    drift_detected = False
    drift_at = None
    stream = np.concatenate([rng.normal(0, 1, 300), rng.normal(1.5, 1, 200)])
    for i, x in enumerate(stream):
        cusum = max(0, cusum + (x - mu_0 - k))
        if cusum > h and not drift_detected:
            drift_at = i
            drift_detected = True
    print(f"    Stream: 300 samples N(0,1) then 200 samples N(1.5,1)")
    print(f"    Drift detected at step: {drift_at}  (true change at step 300)")

    # Plot
    if matplotlib_ok:
        fig, axes = plt.subplots(2, 1, figsize=(8, 5))
        axes[0].hist(ref, bins=40, alpha=0.6, label="Reference (train)")
        axes[0].hist(cur, bins=40, alpha=0.6, label="Current (prod)")
        axes[0].legend(); axes[0].set_title("Feature Distribution Drift")

        cusum_vals = []
        c = 0.0
        for x in stream:
            c = max(0, c + (x - mu_0 - k))
            cusum_vals.append(c)
        axes[1].plot(cusum_vals, label="CUSUM")
        axes[1].axhline(h, color='r', linestyle='--', label="Threshold")
        if drift_at: axes[1].axvline(drift_at, color='g', linestyle='--', label="Detected")
        axes[1].axvline(300, color='k', linestyle=':', label="True change")
        axes[1].legend(); axes[1].set_title("CUSUM Drift Detection")
        plt.tight_layout()
        path = os.path.join(OUTPUT_DIR, "drift_detection.png")
        plt.savefig(path, dpi=80); plt.close()
        print(f"\n  Plot saved: {path}")


# -- 3. Monitoring tools -------------------------------------------------------
def monitoring_tools():
    print("\n=== Monitoring Tools and Dashboards ===")
    print()
    tools = [
        ("Evidently AI",   "Open-source; data/model drift reports; HTML & JSON"),
        ("Prometheus",     "Metrics scraping; time-series DB; ML metrics"),
        ("Grafana",        "Visualisation; dashboards; alert rules on Prometheus"),
        ("Seldon Alibi",   "Drift detection + outlier detection + explanations"),
        ("NannyML",        "Performance estimation without labels; CBPE"),
        ("WhyLabs",        "Managed; Whylogs for feature profiles"),
        ("Arize AI",       "Managed; embedding drift; LLM monitoring"),
        ("Fiddler",        "Explainability + monitoring; enterprise"),
    ]
    print(f"  {'Tool':<18} {'Notes'}")
    for t, d in tools:
        print(f"  {t:<18} {d}")
    print()
    print("  Key alert rules:")
    alerts = [
        ("p99 latency > 200ms",        "PagerDuty page: serving too slow"),
        ("error rate > 1%",            "PagerDuty: model failing to return results"),
        ("PSI > 0.2 on any feature",   "Slack: significant input drift"),
        ("Accuracy < 0.88 (rolling)",  "Slack: model degrading; trigger retraining"),
        ("GPU util < 20% sustained",   "Scale-down: over-provisioned"),
        ("Request queue depth > 100",  "Scale-up: HPA trigger"),
    ]
    for a, d in alerts:
        print(f"  {a:<32} {d}")


if __name__ == "__main__":
    monitoring_concepts()
    drift_detection()
    monitoring_tools()
