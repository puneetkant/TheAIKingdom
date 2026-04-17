"""
Working Example 2: Monitoring — data drift, prediction drift, alerting, SLA tracking
======================================================================================
Simulates production monitoring with drift detection and alert rules.

Run:  python working_example2.py
"""
from pathlib import Path
try:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit("pip install numpy matplotlib")

OUTPUT = Path(__file__).parent / "output"
OUTPUT.mkdir(exist_ok=True)

def ks_test(reference, production, alpha=0.05):
    """Kolmogorov-Smirnov statistic (simplified)."""
    n1, n2 = len(reference), len(production)
    all_vals = np.sort(np.concatenate([reference, production]))
    cdf1 = np.searchsorted(np.sort(reference), all_vals, side='right') / n1
    cdf2 = np.searchsorted(np.sort(production), all_vals, side='right') / n2
    ks_stat = np.max(np.abs(cdf1 - cdf2))
    critical = 1.36 * np.sqrt((n1 + n2) / (n1 * n2))  # α=0.05 approx
    return ks_stat, ks_stat > critical

def psi(ref, prod, bins=10):
    """Population Stability Index."""
    ref_hist, edges = np.histogram(ref, bins=bins, density=False)
    prod_hist, _ = np.histogram(prod, bins=edges, density=False)
    ref_p = (ref_hist + 1e-8) / ref_hist.sum()
    prod_p = (prod_hist + 1e-8) / prod_hist.sum()
    return np.sum((prod_p - ref_p) * np.log(prod_p / ref_p))

def simulate_production(n_weeks=12, drift_start=7):
    """Return weekly feature means simulating drift after drift_start."""
    means = []
    for w in range(n_weeks):
        shift = (w - drift_start) * 0.3 if w >= drift_start else 0
        data = np.random.randn(200) + shift
        means.append(data)
    return means

class Monitor:
    def __init__(self, reference, psi_threshold=0.2, ks_alpha=0.05):
        self.reference = reference
        self.psi_threshold = psi_threshold
        self.ks_alpha = ks_alpha
        self.alerts = []

    def check(self, week, production):
        ks_stat, ks_drift = ks_test(self.reference, production)
        psi_score = psi(self.reference, production)
        alert = ks_drift or psi_score > self.psi_threshold
        self.alerts.append({"week": week, "ks": ks_stat, "psi": psi_score, "alert": alert})
        return alert

def demo():
    print("=== Monitoring and Observability ===")
    np.random.seed(42)
    reference = np.random.randn(500)
    weekly_data = simulate_production(n_weeks=12, drift_start=7)
    monitor = Monitor(reference)

    for w, prod in enumerate(weekly_data):
        alert = monitor.check(w, prod)
        flag = "🚨 ALERT" if alert else "✓"
        print(f"  Week {w:2d}: KS={monitor.alerts[-1]['ks']:.3f}  PSI={monitor.alerts[-1]['psi']:.3f}  {flag}")

    weeks = [a["week"] for a in monitor.alerts]
    ks_vals = [a["ks"] for a in monitor.alerts]
    psi_vals = [a["psi"] for a in monitor.alerts]
    alert_weeks = [a["week"] for a in monitor.alerts if a["alert"]]

    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    axes[0].plot(weeks, ks_vals, "o-"); axes[0].axhline(0.09, color="red", ls="--", label="threshold")
    axes[0].set_title("KS Statistic"); axes[0].set_xlabel("Week"); axes[0].legend()
    for w in alert_weeks: axes[0].axvspan(w-0.5, w+0.5, alpha=0.2, color="red")
    axes[1].plot(weeks, psi_vals, "s-", color="orange"); axes[1].axhline(0.2, color="red", ls="--", label="threshold")
    axes[1].set_title("PSI (Data Drift)"); axes[1].set_xlabel("Week"); axes[1].legend()
    for w in alert_weeks: axes[1].axvspan(w-0.5, w+0.5, alpha=0.2, color="red")
    plt.tight_layout(); plt.savefig(OUTPUT / "monitoring.png"); plt.close()
    print("  Saved monitoring.png")

if __name__ == "__main__":
    demo()
