"""
Working Example 2: Time Series Fundamentals — stationarity, decomposition, autocorrelation
============================================================================================
Generates synthetic time series, decomposes trend+seasonal+residual, computes ACF.

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

def make_series(n=200, seed=0):
    rng = np.random.default_rng(seed); t = np.arange(n)
    trend = 0.05 * t
    seasonal = 3 * np.sin(2 * np.pi * t / 20)
    noise = rng.normal(0, 0.5, n)
    return trend + seasonal + noise

def moving_average(x, w=20):
    return np.convolve(x, np.ones(w)/w, mode="same")

def acf(x, max_lag=40):
    x = x - x.mean(); n = len(x)
    c0 = np.dot(x, x) / n
    return [np.dot(x[:n-k], x[k:]) / (n * c0) for k in range(max_lag+1)]

def demo():
    print("=== Time Series Fundamentals ===")
    ts = make_series()
    trend_est = moving_average(ts, w=21)
    detrended = ts - trend_est

    # Seasonal estimate (period=20)
    P = 20
    seasonal_est = np.array([detrended[i::P].mean() for i in range(P)] * (len(ts)//P + 1))[:len(ts)]
    residual = ts - trend_est - seasonal_est

    print(f"  Series length: {len(ts)}")
    print(f"  Residual std: {residual.std():.3f}")

    # ADF stationarity proxy (variance ratio)
    half = len(ts) // 2
    ratio = ts[:half].var() / ts[half:].var()
    print(f"  Variance ratio (1st vs 2nd half): {ratio:.3f} (→1 = stationary)")

    rho = acf(ts)
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    axes[0,0].plot(ts); axes[0,0].plot(trend_est, color="red"); axes[0,0].set_title("Series + Trend")
    axes[0,1].plot(seasonal_est[:60]); axes[0,1].set_title("Seasonal component")
    axes[1,0].plot(residual); axes[1,0].set_title("Residual")
    axes[1,1].bar(range(len(rho)), rho); axes[1,1].axhline(0, color="black", lw=0.5)
    axes[1,1].set_title("Autocorrelation (ACF)")
    plt.tight_layout(); plt.savefig(OUTPUT / "ts_fundamentals.png"); plt.close()
    print("  Saved ts_fundamentals.png")

if __name__ == "__main__":
    demo()
