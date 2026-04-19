"""
Working Example: Time Series Fundamentals
Covers data types, stationarity, decomposition, autocorrelation,
and common transformations.
"""
import numpy as np
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_ts_fundamentals")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -- 1. Time series types ------------------------------------------------------
def ts_types():
    print("=== Time Series Fundamentals ===")
    print()
    print("  A time series is an ordered sequence of values x_1, x_2, ..., x_T")
    print()
    print("  Types:")
    types = [
        ("Univariate",   "Single variable over time (e.g. daily temperature)"),
        ("Multivariate", "Multiple co-evolving variables (e.g. sensor array)"),
        ("Hierarchical", "Nested series (country -> state -> city sales)"),
        ("Panel",        "Multiple similar series (different products' sales)"),
        ("Irregular",    "Non-uniform timestamps (event logs, transactions)"),
    ]
    for t, d in types:
        print(f"  {t:<14} {d}")
    print()
    print("  Components:")
    comps = [
        ("Trend (T_t)",       "Long-term direction (linear, quadratic, ...)"),
        ("Seasonality (S_t)", "Repeating patterns at fixed period"),
        ("Cycle (C_t)",       "Multi-period fluctuations (business cycle)"),
        ("Residual (R_t)",    "Irregular / noise component"),
    ]
    for c, d in comps:
        print(f"  {c:<22} {d}")
    print()
    print("  Decomposition models:")
    print("    Additive:         x_t = T_t + S_t + R_t")
    print("    Multiplicative:   x_t = T_t × S_t × R_t")
    print("    Log-additive:     log(x_t) = log(T_t) + log(S_t) + log(R_t)")


# -- 2. Generate and decompose a time series -----------------------------------
def generate_and_decompose():
    print("\n=== Series Generation and Decomposition ===")
    rng = np.random.default_rng(42)
    T   = 120  # 10 years, monthly
    t   = np.arange(T)

    trend      = 0.5 * t + 50
    seasonality = 10 * np.sin(2 * np.pi * t / 12)
    noise       = rng.normal(0, 2, T)
    series      = trend + seasonality + noise

    print(f"  Synthetic series: T={T} months")
    print(f"    Trend:       0.5·t + 50")
    print(f"    Seasonality: 10·sin(2pi·t/12) (period=12)")
    print(f"    Noise:       N(0, 2)")
    print()

    # Simple moving average to estimate trend
    window = 12
    ma = np.convolve(series, np.ones(window)/window, mode='valid')
    trend_est = np.concatenate([np.full(window//2, np.nan),
                                ma,
                                np.full(window - window//2 - 1, np.nan)])

    # Seasonal-residual
    detrended = series - trend_est

    # Average seasonal component
    season_est = np.zeros(T)
    for month in range(12):
        indices = range(month, T, 12)
        vals    = [detrended[i] for i in indices if not np.isnan(detrended[i])]
        for idx in indices:
            season_est[idx] = np.nanmean(vals)

    residual = detrended - season_est

    print(f"  Trend estimate (12-pt MA) — first 5: {trend_est[12:17].round(2)}")
    print(f"  Seasonal estimate — first 12: {season_est[:12].round(2)}")
    print(f"  Residual std: {np.nanstd(residual):.3f}  (true: 2.0)")

    # Plot
    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(series, label="Original"); axes[0].legend()
    axes[1].plot(trend_est, label="Trend"); axes[1].legend()
    axes[2].plot(season_est, label="Seasonal"); axes[2].legend()
    axes[3].plot(residual, label="Residual"); axes[3].legend()
    plt.suptitle("Additive Decomposition")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "decomposition.png")
    plt.savefig(path, dpi=90); plt.close()
    print(f"  Decomposition plot: {path}")
    return series


# -- 3. Stationarity -----------------------------------------------------------
def stationarity(series):
    print("\n=== Stationarity ===")
    print("  A series is (weakly) stationary if:")
    print("    E[x_t] = mu             (constant mean)")
    print("    Var[x_t] = sigma²          (constant variance)")
    print("    Cov[x_t, x_{t-k}] = gamma_k  (covariance depends only on lag k)")
    print()

    # Simple test: compare first and second half statistics
    T = len(series); half = T // 2
    m1, s1 = series[:half].mean(), series[:half].std()
    m2, s2 = series[half:].mean(), series[half:].std()
    print(f"  First half:  mean={m1:.2f}  std={s1:.2f}")
    print(f"  Second half: mean={m2:.2f}  std={s2:.2f}")
    print(f"  -> Non-stationary (trend changes mean)")
    print()

    # First-difference
    diff1 = np.diff(series)
    m1d, s1d = diff1[:half].mean(), diff1[:half].std()
    m2d, s2d = diff1[half:].mean(), diff1[half:].std()
    print(f"  First difference Deltax_t = x_t - x_{'{t-1}'}:")
    print(f"  First half:  mean={m1d:.2f}  std={s1d:.2f}")
    print(f"  Second half: mean={m2d:.2f}  std={s2d:.2f}")
    print(f"  -> Closer to stationary")
    print()
    print("  Common transformations to achieve stationarity:")
    transformations = [
        ("Differencing",     "Deltax_t = x_t - x_{t-1}  (removes trend)"),
        ("Seasonal diff",    "Delta_s x_t = x_t - x_{t-s}  (removes seasonality)"),
        ("Log transform",    "log(x_t)  (stabilises multiplicative variance)"),
        ("Box-Cox",          "y_t = (x_t^lambda - 1)/lambda  (generalises log)"),
        ("Z-score norm",     "(x_t - mu)/sigma  (standardise)"),
        ("Power transform",  "sqrt(x_t), x_t^(1/3)"),
    ]
    for tr, d in transformations:
        print(f"  {tr:<18} {d}")

    print()
    print("  ADF test (Augmented Dickey-Fuller):")
    print("    H0: unit root (non-stationary)")
    print("    H1: stationary")
    print("    p < 0.05 -> reject H0 -> stationary")

    try:
        from scipy.stats import ttest_1samp
        # Simplified ADF approximation: check if diff has zero mean
        stat, p = ttest_1samp(diff1, 0)
        print(f"    Simplified test on first diff: t={stat:.3f}  p={p:.4f}")
    except ImportError:
        print("    (scipy not available)")


# -- 4. ACF and PACF -----------------------------------------------------------
def autocorrelation(series):
    print("\n=== ACF and PACF ===")
    print("  ACF (Autocorrelation Function):")
    print("    rho_k = Cov(x_t, x_{t-k}) / Var(x_t)  = gamma_k / gamma_0")
    print()

    T   = len(series)
    mu  = series.mean()
    var = ((series - mu)**2).mean()
    max_lag = 20

    acf = np.array([((series[k:] - mu) * (series[:T-k] - mu)).mean() / var
                    for k in range(max_lag+1)])
    print(f"  ACF lags 0-{max_lag}:")
    for k in range(0, max_lag+1, 4):
        print(f"    k={k:>2}: {acf[k]:+.3f}")

    # Confidence bounds ±1.96/sqrt(T)
    ci = 1.96 / np.sqrt(T)
    sig_lags = [k for k in range(1, max_lag+1) if abs(acf[k]) > ci]
    print(f"  Significant lags (|rho| > {ci:.3f}): {sig_lags}")
    print()
    print("  PACF (Partial ACF):")
    print("    phi_{kk} = correlation between x_t and x_{t-k} after removing")
    print("    linear influence of intermediate lags")
    print()
    print("  Model identification heuristics:")
    print("    AR(p): ACF tails off; PACF cuts off at lag p")
    print("    MA(q): PACF tails off; ACF cuts off at lag q")
    print("    ARMA(p,q): both tail off")
    print("    Random walk: ACF decays very slowly")

    # Plot ACF
    fig, ax = plt.subplots(figsize=(8, 3))
    lags = np.arange(max_lag+1)
    ax.bar(lags, acf, alpha=0.7)
    ax.axhline(ci, color='r', linestyle='--', label=f'±{ci:.3f}')
    ax.axhline(-ci, color='r', linestyle='--')
    ax.set_xlabel("Lag"); ax.set_ylabel("ACF"); ax.set_title("Autocorrelation Function")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "acf.png")
    plt.savefig(path, dpi=90); plt.close()
    print(f"  ACF plot: {path}")


if __name__ == "__main__":
    ts_types()
    series = generate_and_decompose()
    stationarity(series)
    autocorrelation(series)
