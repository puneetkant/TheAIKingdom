"""
Working Example 2: Experiment Tracking — MLflow-style logging with stdlib
===========================================================================
Implements simple run logging (metrics, params, artifacts) using only stdlib.
Mimics MLflow's API concepts.

Run:  python working_example2.py
"""
from pathlib import Path
import json, time, uuid, csv
try:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.datasets import fetch_california_housing
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_squared_error
except ImportError:
    raise SystemExit("pip install numpy matplotlib scikit-learn")

OUTPUT = Path(__file__).parent / "output"
RUN_DIR = OUTPUT / "mlruns"
OUTPUT.mkdir(exist_ok=True); RUN_DIR.mkdir(exist_ok=True)

class SimpleRun:
    """Minimal MLflow-like run context."""
    def __init__(self, experiment="default"):
        self.run_id = str(uuid.uuid4())[:8]
        self.experiment = experiment
        self.params = {}; self.metrics = {}; self.start = time.time()

    def log_param(self, k, v):  self.params[k] = v
    def log_metric(self, k, v): self.metrics.setdefault(k, []).append(v)
    def log_params(self, d):    self.params.update(d)

    def end(self):
        self.duration = round(time.time() - self.start, 3)
        record = {"run_id": self.run_id, "experiment": self.experiment,
                  "params": self.params, "metrics": self.metrics, "duration_s": self.duration}
        path = RUN_DIR / f"{self.run_id}.json"
        path.write_text(json.dumps(record, indent=2))
        print(f"  [run {self.run_id}] {self.params}  ->  {', '.join(f'{k}={v[-1]:.4f}' for k,v in self.metrics.items())}")
        return record

def demo():
    print("=== Experiment Tracking Demo ===")
    h = fetch_california_housing(); X = StandardScaler().fit_transform(h.data)
    X_tr, X_te, y_tr, y_te = train_test_split(X, h.target, test_size=0.2, random_state=42)

    runs = []
    for alpha in [0.001, 0.01, 0.1, 1.0, 10.0]:
        run = SimpleRun(experiment="ridge_housing")
        run.log_params({"alpha": alpha, "model": "Ridge", "n_train": len(X_tr)})
        model = Ridge(alpha=alpha).fit(X_tr, y_tr)
        tr_mse = mean_squared_error(y_tr, model.predict(X_tr))
        te_mse = mean_squared_error(y_te, model.predict(X_te))
        run.log_metric("train_mse", tr_mse); run.log_metric("test_mse", te_mse)
        records = run.end(); runs.append(records)

    # Summary table
    alphas = [r["params"]["alpha"] for r in runs]
    te_mses = [r["metrics"]["test_mse"][0] for r in runs]
    best = runs[int(np.argmin(te_mses))]
    print(f"\n  Best run: alpha={best['params']['alpha']}  test_mse={best['metrics']['test_mse'][0]:.4f}")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogx(alphas, te_mses, "o-")
    ax.set_xlabel("alpha (log)"); ax.set_ylabel("Test MSE"); ax.set_title("Hyperparameter Sweep")
    plt.tight_layout(); plt.savefig(OUTPUT / "experiment_tracking.png"); plt.close()
    print("  Saved experiment_tracking.png")
    print(f"  Run JSONs saved to {RUN_DIR}")

def demo_compare_runs():
    """Load all saved runs and compare them in a leaderboard-style table."""
    print("\n=== Run Leaderboard ===")
    runs = sorted(RUN_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime)
    if not runs:
        print("  No runs found — run demo() first."); return
    records = [json.loads(r.read_text()) for r in runs]
    # Sort by test_mse ascending
    records_sorted = sorted(records, key=lambda r: r["metrics"].get("test_mse", [9999])[0])
    print(f"  {'Rank':>4}  {'Run ID':>10}  {'alpha':>8}  {'Test MSE':>10}  {'Duration':>10}")
    for rank, rec in enumerate(records_sorted, 1):
        alpha = rec["params"].get("alpha", "?")
        mse   = rec["metrics"].get("test_mse", [0])[0]
        dur   = rec.get("duration_s", 0)
        print(f"  {rank:>4}  {rec['run_id']:>10}  {alpha:>8}  {mse:>10.4f}  {dur:>9.3f}s")


def demo_csv_export():
    """Export run history to CSV — common for sharing experiment results."""
    print("\n=== Export Runs to CSV ===")
    runs = sorted(RUN_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime)
    if not runs:
        print("  No runs found."); return
    records = [json.loads(r.read_text()) for r in runs]
    csv_path = OUTPUT / "run_history.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["run_id", "alpha", "train_mse", "test_mse", "duration_s"])
        writer.writeheader()
        for rec in records:
            writer.writerow({
                "run_id": rec["run_id"],
                "alpha":  rec["params"].get("alpha"),
                "train_mse": rec["metrics"].get("train_mse", [None])[0],
                "test_mse":  rec["metrics"].get("test_mse", [None])[0],
                "duration_s": rec.get("duration_s"),
            })
    print(f"  Saved {len(records)} runs to {csv_path}")


if __name__ == "__main__":
    demo()
    demo_compare_runs()
    demo_csv_export()
