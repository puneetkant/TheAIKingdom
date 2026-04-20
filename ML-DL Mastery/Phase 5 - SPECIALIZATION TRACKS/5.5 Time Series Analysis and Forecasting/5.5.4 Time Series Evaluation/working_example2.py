"""
Working Example 2: Time Series Evaluation — MAE, RMSE, MAPE, SMAPE, backtesting
==================================================================================
Compares multiple forecasting strategies with proper train/val/test splits.

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

def mae(y, yh): return np.mean(np.abs(y - yh))
def rmse(y, yh): return np.sqrt(np.mean((y - yh)**2))
def mape(y, yh): return np.mean(np.abs((y - yh) / (np.abs(y) + 1e-8))) * 100
def smape(y, yh): return 200 * np.mean(np.abs(y - yh) / (np.abs(y) + np.abs(yh) + 1e-8))

def make_series(n=300, seed=0):
    rng = np.random.default_rng(seed); t = np.arange(n)
    return np.sin(2*np.pi*t/25) * 4 + 0.02*t + rng.normal(0, 0.3, n)

def naive_forecast(train, h):
    return np.full(h, train[-1])

def seasonal_naive(train, h, period=25):
    return np.array([train[-(period - (i % period))] for i in range(h)])

def drift_forecast(train, h):
    slope = (train[-1] - train[0]) / (len(train) - 1)
    return train[-1] + slope * np.arange(1, h+1)

def demo():
    print("=== Time Series Evaluation ===")
    ts = make_series()
    split = int(len(ts) * 0.8)
    train, test = ts[:split], ts[split:]
    h = len(test)

    methods = {
        "Naive":    naive_forecast(train, h),
        "Seasonal": seasonal_naive(train, h, period=25),
        "Drift":    drift_forecast(train, h),
    }
    print(f"\n  {'Method':12s} {'MAE':>7s} {'RMSE':>7s} {'MAPE%':>8s} {'sMAPE%':>8s}")
    print("  " + "-"*50)
    for name, pred in methods.items():
        print(f"  {name:12s} {mae(test,pred):7.3f} {rmse(test,pred):7.3f} "
              f"{mape(test,pred):8.2f} {smape(test,pred):8.2f}")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(np.arange(split), train, label="Train", alpha=0.6)
    ax.plot(np.arange(split, split+h), test, label="Test", color="black")
    for name, pred in methods.items():
        ax.plot(np.arange(split, split+h), pred, label=name, ls="--")
    ax.legend(); ax.set_title("Forecasting Methods Comparison")
    plt.tight_layout(); plt.savefig(OUTPUT / "ts_evaluation.png"); plt.close()
    print("\n  Saved ts_evaluation.png")

def demo_walk_forward_validation():
    """Walk-forward (expanding window) validation for time series."""
    print("\n=== Walk-Forward Validation ===")
    ts = make_series()
    min_train = 150; step = 25
    splits = [(min_train + i*step, min_train + (i+1)*step)
              for i in range((len(ts)-min_train)//step)]

    print(f"  {'Fold':>5}  {'Train end':>10}  {'MAE':>8}  {'RMSE':>8}")
    for fold, (tr_end, te_end) in enumerate(splits):
        if te_end > len(ts): break
        train = ts[:tr_end]; test = ts[tr_end:te_end]
        h = len(test)
        pred = seasonal_naive(train, h, period=25)
        print(f"  {fold+1:>5}  {tr_end:>10}  {mae(test, pred):>8.4f}  {rmse(test, pred):>8.4f}")


def demo_prediction_intervals():
    """Bootstrap prediction intervals for naive forecast."""
    print("\n=== Prediction Intervals (Bootstrap) ===")
    ts = make_series()
    split = int(len(ts)*0.8)
    train = ts[:split]; test = ts[split:]
    rng = np.random.default_rng(42)
    n_boot = 200; h = min(10, len(test))
    boot_preds = []
    for _ in range(n_boot):
        residuals = train[1:] - train[:-1]  # first-difference residuals
        sampled = rng.choice(residuals, size=h, replace=True)
        pred = train[-1] + np.cumsum(sampled)
        boot_preds.append(pred)
    boot_arr = np.array(boot_preds)
    lo = np.percentile(boot_arr, 5, axis=0)
    hi = np.percentile(boot_arr, 95, axis=0)
    coverage = np.mean((test[:h] >= lo) & (test[:h] <= hi))
    print(f"  90% PI coverage on {h} steps: {coverage:.3f}  (target: ~0.90)")
    for i in range(min(3, h)):
        print(f"    step {i+1}: true={test[i]:.3f}  [{lo[i]:.3f}, {hi[i]:.3f}]")


if __name__ == "__main__":
    demo()
    demo_walk_forward_validation()
    demo_prediction_intervals()
