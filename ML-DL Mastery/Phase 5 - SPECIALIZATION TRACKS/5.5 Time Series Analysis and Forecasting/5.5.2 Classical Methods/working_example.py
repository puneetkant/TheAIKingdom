"""
Working Example: Classical Time Series Methods
Covers ARIMA, Exponential Smoothing, Holt-Winters, STL decomposition,
and state-space models — all implemented with numpy.
"""
import numpy as np
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_classical_ts")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -- Helper: generate data -----------------------------------------------------
def gen_series(T=120, seed=0):
    rng = np.random.default_rng(seed)
    t   = np.arange(T)
    return (0.4 * t + 50
            + 8 * np.sin(2*np.pi*t/12)
            + rng.normal(0, 2, T))


# -- 1. Exponential smoothing --------------------------------------------------
def exponential_smoothing():
    print("=== Exponential Smoothing Methods ===")
    series = gen_series()
    T      = len(series)

    # Simple exponential smoothing (SES)
    def ses(y, alpha):
        s = np.zeros_like(y)
        s[0] = y[0]
        for t in range(1, len(y)):
            s[t] = alpha * y[t] + (1 - alpha) * s[t-1]
        return s

    alphas = [0.1, 0.3, 0.7]
    print(f"  Simple Exponential Smoothing: ŷ_t = alpha·y_t + (1-alpha)·ŷ_{'{t-1}'}")
    print(f"  alpha close to 1 -> more weight on recent values")
    print()
    for a in alphas:
        smoothed = ses(series, a)
        mse = ((series[1:] - smoothed[:-1])**2).mean()
        print(f"  alpha={a}: one-step-ahead MSE = {mse:.3f}")

    # Double exponential smoothing (Holt)
    def holt(y, alpha, beta, h=1):
        l = np.zeros_like(y); b = np.zeros_like(y)
        l[0], b[0] = y[0], (y[1]-y[0])
        for t in range(1, len(y)):
            l[t] = alpha*y[t] + (1-alpha)*(l[t-1]+b[t-1])
            b[t] = beta*(l[t]-l[t-1]) + (1-beta)*b[t-1]
        forecasts = l[-1] + np.arange(1, h+1) * b[-1]
        return l, b, forecasts

    l, b, fc = holt(series, 0.3, 0.1, h=12)
    print()
    print(f"  Holt (double ES): handles trend")
    print(f"  Level at T={T}: {l[-1]:.2f}  Slope: {b[-1]:.3f}")
    print(f"  12-step forecast: [{fc[0]:.2f} ... {fc[-1]:.2f}]")

    # Holt-Winters (triple ES)
    def holt_winters(y, alpha, beta, gamma, m=12, h=12):
        T = len(y)
        l = np.zeros(T); b = np.zeros(T); s = np.zeros(T+h)
        # Initialise
        l[0] = y[:m].mean()
        b[0] = (y[m:2*m].mean() - y[:m].mean()) / m
        for i in range(m):
            s[i] = y[i] / l[0] if l[0] != 0 else 1.0

        for t in range(1, T):
            l[t] = alpha * (y[t]/s[t-m]) + (1-alpha) * (l[t-1]+b[t-1]) if s[t-m]!=0 else l[t-1]
            b[t] = beta * (l[t]-l[t-1]) + (1-beta) * b[t-1]
            s[t+m-1] = gamma * (y[t]/l[t]) + (1-gamma) * s[t-1] if l[t]!=0 else s[t-1]

        fc = np.array([(l[T-1] + (i+1)*b[T-1]) * s[T-1-m+(i+1)%m] for i in range(h)])
        return l, b, s[:T], fc

    l, b, s, fc = holt_winters(series, 0.3, 0.1, 0.3)
    mse_hw = ((series - l)**2).mean()
    print()
    print(f"  Holt-Winters: handles trend + seasonality (period m=12)")
    print(f"  RMSE vs level: {np.sqrt(mse_hw):.3f}")
    print(f"  12-step forecast: [{fc[0]:.2f} ... {fc[-1]:.2f}]")

    # Plot
    best = ses(series, 0.3)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(series, label="Data", alpha=0.6)
    ax.plot(l, label="HW Level", linewidth=2)
    ax.plot(range(T, T+12), fc, label="HW Forecast", linewidth=2, linestyle='--')
    ax.legend(); ax.set_title("Holt-Winters Forecast")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "holt_winters.png")
    plt.savefig(path, dpi=90); plt.close()
    print(f"\n  Holt-Winters plot: {path}")


# -- 2. AR, MA, ARMA models ----------------------------------------------------
def arma_models():
    print("\n=== AR, MA, ARMA Models ===")
    print()
    print("  AR(p):  x_t = c + phi_1·x_{t-1} + ... + phi_p·x_{t-p} + epsilon_t")
    print("  MA(q):  x_t = c + epsilon_t + theta_1·epsilon_{t-1} + ... + theta_q·epsilon_{t-q}")
    print("  ARMA(p,q): combination")
    print()
    rng = np.random.default_rng(1)

    # Simulate AR(2)
    T = 200
    phi = [0.5, -0.3]
    e   = rng.normal(0, 1, T)
    x   = np.zeros(T)
    for t in range(2, T):
        x[t] = phi[0]*x[t-1] + phi[1]*x[t-2] + e[t]

    print("  Simulated AR(2): phi_1=0.5, phi_2=-0.3")
    print(f"  Mean: {x.mean():.4f} (~=0)  Std: {x.std():.4f}")

    # Yule-Walker estimation
    mu  = x.mean()
    xc  = x - mu
    g0  = (xc**2).mean()
    g1  = (xc[1:]*xc[:-1]).mean()
    g2  = (xc[2:]*xc[:-2]).mean()

    # AR(2) via Yule-Walker
    R   = np.array([[g0, g1], [g1, g0]])
    r   = np.array([g1, g2])
    phi_est = np.linalg.solve(R, r)
    print(f"  Yule-Walker estimates: phi_1={phi_est[0]:.4f}  phi_2={phi_est[1]:.4f}")
    print(f"  True values:           phi_1=0.5000  phi_2=-0.3000")

    # Simulate MA(2)
    theta = [0.5, 0.3]
    e2    = rng.normal(0, 1, T)
    y     = np.zeros(T)
    for t in range(2, T):
        y[t] = e2[t] + theta[0]*e2[t-1] + theta[1]*e2[t-2]
    print()
    print("  Simulated MA(2): theta_1=0.5, theta_2=0.3")
    print(f"  Mean: {y.mean():.4f}  Std: {y.std():.4f}")


# -- 3. ARIMA (integration) ----------------------------------------------------
def arima_overview():
    print("\n=== ARIMA(p, d, q) ===")
    print()
    print("  ARIMA = Autoregressive Integrated Moving Average")
    print("  d = order of differencing to achieve stationarity")
    print()
    print("  ARIMA(1,1,0):")
    print("    Deltax_t = phi_1·Deltax_{t-1} + epsilon_t")
    print()
    print("  SARIMA(p,d,q)(P,D,Q)_m:")
    print("    Adds seasonal AR, I, MA terms with period m")
    print("    e.g. SARIMA(1,1,1)(1,1,1)_{12} for monthly sales data")
    print()
    print("  Model selection via information criteria:")
    print("    AIC = 2k - 2·log(L)       (penalises complexity)")
    print("    BIC = k·log(T) - 2·log(L)  (stronger penalty)")
    print("    AIC tends to choose higher-order; BIC more parsimonious")
    print()

    # Simulate ARIMA(1,1,0)
    rng = np.random.default_rng(2)
    T   = 150
    phi = 0.7
    e   = rng.normal(0, 1, T)
    dx  = np.zeros(T)
    for t in range(1, T):
        dx[t] = phi * dx[t-1] + e[t]
    x = dx.cumsum()  # integrate once

    train, test = x[:120], x[120:]
    print(f"  Simulated ARIMA(1,1,0): phi_1={phi}")
    print(f"  Train: {len(train)}  Test: {len(test)}")

    # Naive forecast (last value)
    fc_naive = np.full(30, train[-1])
    mse_naive = ((test - fc_naive)**2).mean()

    # AR(1) on differences
    dtrain = np.diff(train)
    g0d = (dtrain**2).mean(); g1d = (dtrain[1:]*dtrain[:-1]).mean()
    phi_est = g1d / g0d

    dtrain_last = dtrain[-1]
    fc_diff = np.zeros(30)
    v = dtrain_last
    for i in range(30):
        v = phi_est * v
        fc_diff[i] = train[-1] + (i+1)*v if i == 0 else fc_diff[i-1] + v
    mse_arima = ((test - fc_diff)**2).mean()

    print(f"  phi_1 estimate (Yule-Walker): {phi_est:.4f}  (true: {phi})")
    print(f"  Test MSE — naive: {mse_naive:.2f}   AR(1)-on-diff: {mse_arima:.2f}")


# -- 4. Box-Jenkins methodology ------------------------------------------------
def box_jenkins():
    print("\n=== Box-Jenkins Methodology ===")
    print("  Step-by-step workflow for ARIMA modelling:")
    print()
    steps = [
        ("1. Explore",       "Plot series, ACF/PACF; identify patterns"),
        ("2. Stationarity",  "ADF/KPSS tests; apply differencing d times"),
        ("3. Identify p, q", "ACF -> q; PACF -> p; tentative model"),
        ("4. Estimate",      "MLE of phi, theta; or conditional least squares"),
        ("5. Diagnostics",   "Residual ACF (should be white noise); Ljung-Box test"),
        ("6. Forecast",      "h-step ahead with prediction intervals"),
    ]
    for s, d in steps:
        print(f"  {s:<16} {d}")
    print()
    print("  Tools: statsmodels (Python), auto_arima (pmdarima), forecast (R)")


if __name__ == "__main__":
    exponential_smoothing()
    arma_models()
    arima_overview()
    box_jenkins()
