"""
Working Example 2: Experiment Tracking — manual experiment logger, metrics, comparison
========================================================================================
Implements a lightweight experiment tracker (no MLflow required).

Run:  python working_example2.py
"""
from pathlib import Path
import json, time
try:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit("pip install numpy matplotlib")

OUTPUT = Path(__file__).parent / "output"
OUTPUT.mkdir(exist_ok=True)

class Experiment:
    def __init__(self, name):
        self.name = name; self.runs = []

    def log_run(self, params, metrics, artifacts=None):
        self.runs.append({
            "run_id": len(self.runs),
            "name": self.name,
            "params": params,
            "metrics": metrics,
            "artifacts": artifacts or [],
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })

    def best_run(self, metric="val_acc", higher_better=True):
        fn = max if higher_better else min
        return fn(self.runs, key=lambda r: r["metrics"].get(metric, -1e9))

    def to_dict(self):
        return {"name": self.name, "runs": self.runs}

def train_lr(X_tr, y_tr, X_val, y_val, lr=0.1, n_iter=60):
    N, F = X_tr.shape
    W = np.zeros(F + 1)
    Xb = np.column_stack([np.ones(N), X_tr])
    Xvb = np.column_stack([np.ones(len(X_val)), X_val])
    train_losses = []
    for _ in range(n_iter):
        p = 1/(1+np.exp(-(Xb@W).clip(-50,50)))
        loss = -np.mean(y_tr*np.log(p+1e-8)+(1-y_tr)*np.log(1-p+1e-8))
        train_losses.append(loss)
        W -= lr * (Xb.T@(p-y_tr))/N
    val_preds = (1/(1+np.exp(-(Xvb@W).clip(-50,50))) >= 0.5).astype(int)
    val_acc = (val_preds == y_val).mean()
    return val_acc, train_losses[-1]

def demo():
    np.random.seed(99)
    N, F = 300, 5
    X = np.random.randn(N, F)
    y = (X[:,0] - 0.5*X[:,1] + 0.3*X[:,2] > 0).astype(int)
    split = int(0.75 * N)
    X_tr, X_val = X[:split], X[split:]
    y_tr, y_val = y[:split], y[split:]
    # Normalise
    mu, sigma = X_tr.mean(0), X_tr.std(0).clip(1e-8)
    X_tr_n, X_val_n = (X_tr-mu)/sigma, (X_val-mu)/sigma

    exp = Experiment("lr-sweep")
    print("=== Experiment Tracking ===")
    for lr in [0.01, 0.05, 0.1, 0.3, 0.5]:
        val_acc, final_loss = train_lr(X_tr_n, y_tr, X_val_n, y_val, lr=lr, n_iter=80)
        exp.log_run(
            params={"lr": lr, "n_iter": 80},
            metrics={"val_acc": round(val_acc, 4), "train_loss": round(final_loss, 4)},
        )
        print(f"  lr={lr:.2f}  val_acc={val_acc:.4f}  loss={final_loss:.4f}")

    best = exp.best_run("val_acc")
    print(f"\n  Best run: lr={best['params']['lr']}  val_acc={best['metrics']['val_acc']}")

    # Save experiment log
    log_path = OUTPUT / "experiment_log.json"
    log_path.write_text(json.dumps(exp.to_dict(), indent=2))
    print(f"  Experiment log saved to {log_path.name}")

    # Plot
    lrs = [r["params"]["lr"] for r in exp.runs]
    accs = [r["metrics"]["val_acc"] for r in exp.runs]
    plt.figure(figsize=(5, 3))
    plt.plot(lrs, accs, "o-"); plt.xscale("log"); plt.xlabel("Learning rate")
    plt.ylabel("Val accuracy"); plt.title("LR Sweep"); plt.tight_layout()
    plt.savefig(OUTPUT / "experiment_tracking.png"); plt.close()
    print("  Saved experiment_tracking.png")

if __name__ == "__main__":
    demo()
