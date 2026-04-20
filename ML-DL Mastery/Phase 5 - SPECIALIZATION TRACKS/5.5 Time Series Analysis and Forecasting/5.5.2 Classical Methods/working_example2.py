"""
Working Example 2: Classical Time Series — ARIMA components and Exponential Smoothing
======================================================================================
Implements AR(1) simulation and simple exponential smoothing from scratch.

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

def ar1_simulate(phi=0.8, sigma=1.0, n=200, seed=0):
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t-1] + rng.normal(0, sigma)
    return x

def simple_exp_smooth(ts, alpha=0.3):
    """SES: s_t = alpha * y_t + (1-alpha) * s_{t-1}."""
    s = np.zeros_like(ts); s[0] = ts[0]
    for t in range(1, len(ts)):
        s[t] = alpha * ts[t] + (1-alpha) * s[t-1]
    return s

def holt_linear(ts, alpha=0.3, beta=0.1):
    """Holt's double exponential smoothing."""
    L = ts[0]; B = ts[1] - ts[0]
    forecasts = []
    for y in ts:
        L_new = alpha * y + (1-alpha) * (L + B)
        B_new = beta * (L_new - L) + (1-beta) * B
        L, B = L_new, B_new
        forecasts.append(L + B)
    return np.array(forecasts)

def demo():
    print("=== Classical Time Series Methods ===")
    ts = ar1_simulate(phi=0.85) + np.sin(np.arange(200) / 10) * 2

    ses = simple_exp_smooth(ts, alpha=0.3)
    holt = holt_linear(ts)
    for label, pred in [("SES", ses), ("Holt", holt)]:
        mae = np.abs(ts[1:] - pred[:-1]).mean()
        print(f"  {label:5s} MAE (1-step): {mae:.3f}")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(ts, label="Observed", alpha=0.6)
    ax.plot(ses, label="SES (alpha=0.3)")
    ax.plot(holt, label="Holt linear")
    ax.legend(); ax.set_title("Classical Smoothing Methods")
    plt.tight_layout(); plt.savefig(OUTPUT / "classical_ts.png"); plt.close()
    print("  Saved classical_ts.png")

def demo_arima_components():
    """Show AR, MA, and ARMA component behaviour on synthetic data."""
    print("\n=== ARIMA Component Analysis ===")
    rng = np.random.default_rng(7)
    n = 200

    # AR(1) process
    phi = 0.7; ar_ts = ar1_simulate(phi=phi, n=n)
    # MA(1) process
    theta_ma = 0.6; eps = rng.normal(0, 1, n)
    ma_ts = np.zeros(n); ma_ts[0] = eps[0]
    for i in range(1, n):
        ma_ts[i] = eps[i] + theta_ma * eps[i-1]

    # Compute lag-1 autocorrelation as a quick check
    def lag_acf(ts, lag=1):
        return np.corrcoef(ts[:-lag], ts[lag:])[0,1]

    print(f"  AR(1) phi={phi}: lag-1 ACF={lag_acf(ar_ts):.4f}  (expected ~{phi})")
    print(f"  MA(1) theta={theta_ma}: lag-1 ACF={lag_acf(ma_ts):.4f}  (expected ~{theta_ma/(1+theta_ma**2):.4f})")
    print(f"  White noise:     lag-1 ACF={lag_acf(eps):.4f}  (expected ~0)")


def demo_forecast_horizon():
    """Show how forecast accuracy degrades with horizon length."""
    print("\n=== Forecast Accuracy vs Horizon ===")
    ts = ar1_simulate(phi=0.85) + np.sin(np.arange(200)/10)*2
    split = 160; train = ts[:split]
    for horizon in [1, 5, 10, 20]:
        test = ts[split:split+horizon]
        naive = np.full(horizon, train[-1])
        ses   = simple_exp_smooth(train, alpha=0.3)
        ses_pred = np.full(horizon, ses[-1])
        mae_naive = np.abs(test - naive).mean()
        mae_ses   = np.abs(test - ses_pred).mean()
        print(f"  h={horizon:3d}: Naive MAE={mae_naive:.4f}  SES MAE={mae_ses:.4f}")


if __name__ == "__main__":
    demo()
    demo_arima_components()
    demo_forecast_horizon()
