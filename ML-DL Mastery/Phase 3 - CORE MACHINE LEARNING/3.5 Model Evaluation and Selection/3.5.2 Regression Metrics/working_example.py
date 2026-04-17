"""
Working Example: Regression Metrics
Covers MAE, MSE, RMSE, R², adjusted R², MAPE, SMAPE, RMSLE, Huber loss,
residual analysis, and choosing the right metric.
"""
import numpy as np
from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                              r2_score, mean_absolute_percentage_error,
                              mean_squared_log_error, median_absolute_error,
                              explained_variance_score)
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_reg_metrics")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── 1. Core regression metrics ───────────────────────────────────────────────
def core_metrics():
    print("=== Core Regression Metrics ===")
    rng = np.random.default_rng(0)
    n   = 300
    X   = rng.uniform(0, 10, (n, 3))
    y   = 3*X[:,0] - 2*X[:,1] + X[:,2] + rng.normal(0, 2, n)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)
    model = LinearRegression().fit(X_tr, y_tr)
    y_pred = model.predict(X_te)

    MAE   = mean_absolute_error(y_te, y_pred)
    MSE   = mean_squared_error(y_te, y_pred)
    RMSE  = np.sqrt(MSE)
    R2    = r2_score(y_te, y_pred)
    p     = X_te.shape[1]
    n_te  = len(y_te)
    R2adj = 1 - (1-R2)*(n_te-1)/(n_te-p-1)
    MedAE = median_absolute_error(y_te, y_pred)
    EV    = explained_variance_score(y_te, y_pred)

    print(f"  {'Metric':<32} {'Value'}")
    for name, val in [
        ("MAE (Mean Abs Error)",    MAE),
        ("MSE (Mean Sq Error)",     MSE),
        ("RMSE (Root MSE)",         RMSE),
        ("Median Absolute Error",   MedAE),
        ("R² (Coefficient of Det.)",R2),
        ("Adjusted R²",             R2adj),
        ("Explained Variance Score",EV),
    ]:
        print(f"  {name:<38}: {val:.4f}")

    print(f"\n  Formulas:")
    print(f"    MAE     = Σ|y-ŷ|/n")
    print(f"    MSE     = Σ(y-ŷ)²/n")
    print(f"    RMSE    = √MSE")
    print(f"    R²      = 1 - SS_res/SS_tot  (proportion of variance explained)")
    print(f"    R²_adj  = 1 - (1-R²)(n-1)/(n-p-1)")


# ── 2. Relative metrics (MAPE, SMAPE) ────────────────────────────────────────
def relative_metrics():
    print("\n=== Relative Metrics (scale-independent) ===")
    rng = np.random.default_rng(1)
    y_te   = rng.uniform(1, 100, 100)
    y_pred = y_te * (1 + rng.normal(0, 0.2, 100))

    MAPE  = mean_absolute_percentage_error(y_te, y_pred)
    SMAPE = 100 * np.mean(2*np.abs(y_te - y_pred) / (np.abs(y_te) + np.abs(y_pred) + 1e-9))

    print(f"  MAPE  = {MAPE*100:.2f}%  (Mean Absolute Percentage Error)")
    print(f"  SMAPE = {SMAPE:.2f}%  (Symmetric MAPE, capped at 200%)")
    print(f"\n  MAPE = mean(|y-ŷ|/|y|)  ← undefined when y=0, asymmetric")
    print(f"  SMAPE = mean(2|y-ŷ|/(|y|+|ŷ|)) ← symmetric, bounded")

    # Demonstrate MAPE issue near zero
    y_near_zero = np.array([0.01, 1.0, 10.0, 100.0])
    y_pred_zero = y_near_zero * 1.1
    mape_each   = np.abs(y_near_zero - y_pred_zero) / np.abs(y_near_zero) * 100
    print(f"\n  MAPE issue near zero:")
    for ytrue, ypred, m in zip(y_near_zero, y_pred_zero, mape_each):
        print(f"    y={ytrue:.2f}  ŷ={ypred:.3f}  MAPE={m:.1f}%")


# ── 3. RMSLE (log scale) ─────────────────────────────────────────────────────
def rmsle_demo():
    print("\n=== RMSLE (Root Mean Squared Log Error) ===")
    print("  Used when target spans several orders of magnitude")
    print("  Penalises under-prediction more than over-prediction")
    print("  Formula: √(Σ(log(ŷ+1) - log(y+1))²/n)")

    rng = np.random.default_rng(2)
    y_te   = rng.exponential(1000, 200).clip(1, None)
    y_pred = y_te * (1 + rng.normal(0, 0.3, 200)).clip(0.1)

    RMSLE = np.sqrt(mean_squared_log_error(y_te, y_pred))
    RMSE  = np.sqrt(mean_squared_error(y_te, y_pred))
    print(f"\n  RMSE  (original scale): {RMSE:.2f}")
    print(f"  RMSLE (log scale):      {RMSLE:.4f}")

    # Under vs over prediction
    for factor, label in [(0.5, "under by 50%"), (1.5, "over by 50%"),
                           (0.9, "under by 10%"), (1.1, "over by 10%")]:
        y_p = y_te * factor
        rmsle = np.sqrt(mean_squared_log_error(y_te, y_p))
        print(f"  Predict {label}: RMSLE={rmsle:.4f}")


# ── 4. Residual analysis ─────────────────────────────────────────────────────
def residual_analysis():
    print("\n=== Residual Analysis ===")
    rng = np.random.default_rng(3)
    n   = 200
    x   = rng.uniform(0, 10, n)
    y   = 2*x + rng.normal(0, 1 + 0.3*x, n)   # heteroscedastic
    X   = x.reshape(-1, 1)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)
    model  = LinearRegression().fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    resid  = y_te - y_pred

    print(f"  Residuals: mean={resid.mean():.4f}  std={resid.std():.4f}")
    print(f"  Should be ~0 and homoscedastic")

    # Normality test
    from scipy.stats import shapiro, pearsonr
    stat, p = shapiro(resid)
    print(f"  Shapiro-Wilk normality: stat={stat:.4f}  p={p:.4f}  "
          f"({'normal' if p>0.05 else 'non-normal'})")

    # Durbin-Watson (autocorrelation)
    dw = np.sum(np.diff(resid)**2) / np.sum(resid**2)
    print(f"  Durbin-Watson (2=no autocorrelation): {dw:.4f}")

    # Correlation with predicted (heteroscedasticity check)
    corr, p = pearsonr(y_pred, np.abs(resid))
    print(f"  |Residual| vs ŷ corr={corr:.4f}  (heteroscedasticity if |corr| large)")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(y_pred, resid, alpha=0.5, s=20)
    axes[0].axhline(0, color='r', lw=2)
    axes[0].set(xlabel="Fitted values", ylabel="Residuals", title="Residuals vs Fitted")
    axes[0].grid(True, alpha=0.3)
    axes[1].hist(resid, bins=20, edgecolor='k', alpha=0.7)
    axes[1].set(xlabel="Residual", ylabel="Count", title="Residual Distribution")
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "residual_analysis.png")
    plt.savefig(path, dpi=90); plt.close()
    print(f"\n  Residual plots saved: {path}")


# ── 5. Metric comparison on multiple models ───────────────────────────────────
def model_comparison():
    print("\n=== Regression Metric Comparison Across Models ===")
    rng = np.random.default_rng(4)
    n   = 400
    X   = rng.uniform(0, 5, (n, 4))
    y   = (X[:,0]**2 + np.sin(X[:,1]) + 0.5*X[:,2]
           + rng.normal(0, 0.5, n)).clip(0.01)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)

    models = [
        ("Linear Reg.",    LinearRegression()),
        ("Random Forest",  RandomForestRegressor(n_estimators=50, random_state=0, n_jobs=-1)),
        ("Gradient Boost", GradientBoostingRegressor(n_estimators=50, random_state=0)),
        ("Huber Reg.",     HuberRegressor()),
    ]
    print(f"  {'Model':<18} {'MAE':>8} {'RMSE':>8} {'R²':>8} {'MAPE%':>8} {'RMSLE':>8}")
    for name, m in models:
        m.fit(X_tr, y_tr)
        yp     = m.predict(X_te).clip(0.001)
        mae    = mean_absolute_error(y_te, yp)
        rmse   = np.sqrt(mean_squared_error(y_te, yp))
        r2     = r2_score(y_te, yp)
        mape   = mean_absolute_percentage_error(y_te, yp) * 100
        rmsle  = np.sqrt(mean_squared_log_error(y_te, yp))
        print(f"  {name:<18} {mae:>8.4f} {rmse:>8.4f} {r2:>8.4f} {mape:>8.2f} {rmsle:>8.4f}")


# ── 6. When to use which metric ───────────────────────────────────────────────
def metric_selection_guide():
    print("\n=== Metric Selection Guide ===")
    print("  MAE:   Robust to outliers, interpretable (same units as y)")
    print("         Use when outliers should not dominate the metric")
    print("  RMSE:  Penalises large errors more; same units as y")
    print("         Use when large errors are especially bad (forecasting)")
    print("  R²:    Proportion of variance explained; [0,1] for proper models")
    print("         Use for comparing models on same dataset")
    print("  MAPE:  Scale-independent, easy to communicate as % error")
    print("         Avoid when targets can be zero or near-zero")
    print("  RMSLE: For skewed targets spanning orders of magnitude")
    print("         Use in Kaggle-style pricing or count predictions")
    print("  MedAE: Highly robust to outliers, median of |errors|")
    print("         Use when robustness is critical")


if __name__ == "__main__":
    core_metrics()
    relative_metrics()
    rmsle_demo()
    residual_analysis()
    model_comparison()
    metric_selection_guide()
