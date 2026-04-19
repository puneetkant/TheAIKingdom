"""
Working Example: Time Series Evaluation
Covers forecasting metrics, backtesting, cross-validation,
probabilistic forecasts, and model comparison strategies.
"""
import numpy as np
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_ts_eval")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -- 1. Point forecast metrics -------------------------------------------------
def point_metrics():
    print("=== Time Series Evaluation Metrics ===")
    rng  = np.random.default_rng(0)
    T    = 50
    true = np.sin(np.linspace(0, 4*np.pi, T)) + rng.normal(0, 0.1, T)
    pred = true + rng.normal(0, 0.3, T)   # noisy prediction

    def mae(y, yh):   return np.abs(y-yh).mean()
    def rmse(y, yh):  return np.sqrt(((y-yh)**2).mean())
    def mape(y, yh):  return 100*np.abs((y-yh)/np.where(y==0,1e-8,y)).mean()
    def smape(y, yh):
        denom = (np.abs(y) + np.abs(yh)) / 2
        return 100*np.abs(y-yh).mean() / denom.mean()
    def mase(y, yh, y_train, m=1):
        naive_mae = np.abs(np.diff(y_train, n=m)).mean()
        return mae(y,yh) / naive_mae

    train = true + rng.normal(0, 0.1, T)  # synthetic train for MASE denominator

    print(f"  {'Metric':<10} {'Value':>10}  {'Notes'}")
    print(f"  {'-'*10} {'-'*10}  {'-'*45}")
    print(f"  {'MAE':<10} {mae(true,pred):>10.4f}  Mean absolute error")
    print(f"  {'RMSE':<10} {rmse(true,pred):>10.4f}  Root mean squared error (penalises outliers)")
    print(f"  {'MAPE':<10} {mape(true,pred):>10.4f}  % error; ill-defined near zero")
    print(f"  {'sMAPE':<10} {smape(true,pred):>10.4f}  Symmetric MAPE; bounded [0, 200%]")
    print(f"  {'MASE':<10} {mase(true,pred,train):>10.4f}  Scale-free; vs naive (>1=worse than naive)")
    print()
    print("  Recommendations:")
    print("    RMSE: sensitive to large errors; use when errors are costly")
    print("    MAE:  robust to outliers; easy to interpret")
    print("    MASE: scale-free; works across different series")
    print("    M4/M5 competition uses sMAPE + MASE (OWA: weighted average)")


# -- 2. Backtesting and walk-forward validation --------------------------------
def backtesting():
    print("\n=== Backtesting (Walk-Forward Validation) ===")
    print()
    print("  Cannot shuffle time series (temporal ordering must be preserved)")
    print()
    print("  Simple train/test split:")
    print("    Use last H points as test; train on [1 : T-H]")
    print("    Risk: model may not generalise to different regimes")
    print()
    print("  Walk-forward (expanding window):")
    print("    For each fold, train on all history up to t, forecast h steps")
    print("    -> Most realistic; average metrics across folds")
    print()
    print("  Sliding window:")
    print("    Fixed-size training window [t-W : t]")
    print("    Useful when far-past data becomes less relevant")
    print()

    # Simulate walk-forward
    rng = np.random.default_rng(0)
    T = 100; h = 5; min_train = 30
    y = np.cumsum(rng.normal(0, 1, T))

    maes = []
    for t_start in range(min_train, T-h, h):
        train = y[:t_start]
        test  = y[t_start:t_start+h]
        # Naive forecast: repeat last value
        fc    = np.full(h, train[-1])
        maes.append(np.abs(test - fc).mean())

    print(f"  Walk-forward evaluation ({len(maes)} folds, h={h}):")
    print(f"    MAE per fold: {np.round(maes, 3)}")
    print(f"    Mean MAE: {np.mean(maes):.3f}  Std: {np.std(maes):.3f}")


# -- 3. Probabilistic forecasting ---------------------------------------------
def probabilistic_forecasting():
    print("\n=== Probabilistic Forecasting ===")
    print()
    print("  Instead of point estimate, predict full distribution P(y_{t+h} | x_t)")
    print()
    print("  Output forms:")
    forms = [
        ("Quantile",     "ŷ_q for q in {0.1, 0.5, 0.9}; non-crossing property"),
        ("Interval",     "Prediction interval [L, U] at coverage 1-alpha"),
        ("Parametric",   "N(mu, sigma²) or NegBinomial(mu, alpha) per timestep"),
        ("Sample-based", "Monte Carlo draws from predictive distribution"),
    ]
    for f, d in forms:
        print(f"  {f:<14} {d}")
    print()

    # Quantile loss (pinball loss)
    print("  Quantile loss (pinball):")
    print("    L_q(y, ŷ) = q·(y-ŷ)·1[y>=ŷ] + (1-q)·(ŷ-y)·1[y<ŷ]")
    print()

    rng  = np.random.default_rng(0)
    T    = 100
    y    = rng.normal(5, 2, T)
    # Simulate quantile predictions
    q10  = np.full(T, np.percentile(rng.normal(5,2,1000), 10))
    q90  = np.full(T, np.percentile(rng.normal(5,2,1000), 90))
    q50  = np.full(T, np.median(rng.normal(5,2,1000)))

    def pinball(y, yhat, q):
        e = y - yhat
        return np.where(e >= 0, q*e, (q-1)*e).mean()

    print(f"  Pinball loss @ q=0.10: {pinball(y, q10, 0.10):.4f}")
    print(f"  Pinball loss @ q=0.50: {pinball(y, q50, 0.50):.4f}")
    print(f"  Pinball loss @ q=0.90: {pinball(y, q90, 0.90):.4f}")

    # Coverage check
    coverage = np.mean((y >= q10) & (y <= q90))
    print(f"  80% PI coverage: {coverage:.2%} (expected: 80%)")

    print()
    print("  CRPS (Continuous Ranked Probability Score):")
    print("    CRPS = integral[F(z) - 1{z>=y}]² dz  (lower=better)")
    print("    Proper scoring rule: rewards calibration + sharpness")
    print("    Decomposes: Reliability + Resolution")


# -- 4. Model comparison -------------------------------------------------------
def model_comparison():
    print("\n=== Model Comparison and Selection ===")
    print()
    print("  Statistical tests for comparing forecasters:")
    print()
    tests = [
        ("Diebold-Mariano", "Test H0: equal predictive accuracy; paired errors"),
        ("Model Confidence Set", "Set of models not significantly worse than best"),
        ("Giacomini-White", "Conditional predictive ability test"),
    ]
    for t, d in tests:
        print(f"  {t:<22} {d}")
    print()

    # Simulate Diebold-Mariano (simplified)
    rng  = np.random.default_rng(0)
    T    = 50
    y    = rng.normal(0, 1, T)
    e1   = rng.normal(0, 1, T)   # model 1 errors
    e2   = rng.normal(0, 0.8, T) # model 2 errors (better)

    loss1 = e1**2
    loss2 = e2**2
    d_t   = loss1 - loss2
    dm    = d_t.mean() / (d_t.std() / np.sqrt(T))
    print(f"  DM test statistic: {dm:.3f}")
    print(f"  (Large |DM| -> significant difference; DM ~ N(0,1) asymptotically)")
    print()
    print("  Best practices:")
    bps = [
        "Use multiple evaluation windows (not just one test set)",
        "Report mean ± std across folds",
        "Check for statistical significance before claiming improvement",
        "Consider computational cost vs accuracy trade-off",
        "Match evaluation metric to business objective",
        "Beware of overfitting to validation set with hyperparameter tuning",
    ]
    for bp in bps:
        print(f"  • {bp}")


if __name__ == "__main__":
    point_metrics()
    backtesting()
    probabilistic_forecasting()
    model_comparison()
