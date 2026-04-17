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

if __name__ == "__main__":
    demo()
