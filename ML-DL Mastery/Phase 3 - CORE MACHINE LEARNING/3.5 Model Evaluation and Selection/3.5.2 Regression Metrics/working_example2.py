"""
Working Example 2: Regression Metrics — MAE, MSE, RMSE, R², MAPE
=================================================================
Comparing metrics across models, residual analysis.

Run:  python working_example2.py
"""
from pathlib import Path
try:
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge, Lasso
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                                  r2_score, median_absolute_error)
    from sklearn.pipeline import make_pipeline
except ImportError:
    raise SystemExit("pip install numpy matplotlib scikit-learn")

OUTPUT = Path(__file__).parent / "output"
OUTPUT.mkdir(exist_ok=True)

def mape(y_true, y_pred):
    mask = y_true != 0
    return np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]).mean() * 100

def demo_metrics_comparison():
    print("=== Regression Metrics Comparison ===")
    h = fetch_california_housing()
    X, y = h.data, h.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = [
        ("Ridge",  make_pipeline(StandardScaler(), Ridge(1.0))),
        ("Lasso",  make_pipeline(StandardScaler(), Lasso(0.01))),
        ("RF-100", make_pipeline(StandardScaler(), RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))),
        ("GBM",    make_pipeline(StandardScaler(), GradientBoostingRegressor(n_estimators=100, random_state=42))),
    ]
    hdr = f"  {'Model':8s}  {'MAE':>8}  {'RMSE':>8}  {'R²':>8}  {'MedAE':>8}  {'MAPE%':>8}"
    print(hdr)
    for name, pipe in models:
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        mae  = mean_absolute_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred)**0.5
        r2   = r2_score(y_test, y_pred)
        med  = median_absolute_error(y_test, y_pred)
        mp   = mape(y_test, y_pred)
        print(f"  {name:8s}  {mae:8.4f}  {rmse:8.4f}  {r2:8.4f}  {med:8.4f}  {mp:8.2f}")

def demo_residuals():
    print("\n=== Residual Analysis (Ridge) ===")
    h = fetch_california_housing()
    X, y = h.data, h.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipe = make_pipeline(StandardScaler(), Ridge(1.0))
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    resid = y_test - y_pred

    print(f"  Residual mean: {resid.mean():.4f}  std: {resid.std():.4f}")
    print(f"  % residuals within ±0.5: {(np.abs(resid)<0.5).mean()*100:.1f}%")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.scatter(y_pred, resid, alpha=0.2, s=5)
    ax1.axhline(0, c="r"); ax1.set_xlabel("Predicted"); ax1.set_ylabel("Residual")
    ax1.set_title("Residuals vs Predicted")
    ax2.hist(resid, bins=50); ax2.set_title("Residual Distribution"); ax2.set_xlabel("Residual")
    plt.tight_layout(); plt.savefig(OUTPUT / "residuals_analysis.png"); plt.close()
    print("  Saved residuals_analysis.png")

if __name__ == "__main__":
    demo_metrics_comparison()
    demo_residuals()
