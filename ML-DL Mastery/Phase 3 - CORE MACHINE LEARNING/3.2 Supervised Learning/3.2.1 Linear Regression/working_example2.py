"""
Working Example 2: Linear Regression — OLS, Ridge, Gradient Descent, Cal Housing
==================================================================================
Closed-form OLS, Ridge regularisation, gradient descent from scratch,
feature importance (coefficients), residual analysis.

Run:  python working_example2.py
"""
from pathlib import Path
try:
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.pipeline import make_pipeline
except ImportError:
    raise SystemExit("pip install numpy matplotlib scikit-learn")

OUTPUT = Path(__file__).parent / "output"
OUTPUT.mkdir(exist_ok=True)

def load_data():
    h = fetch_california_housing()
    X_train, X_test, y_train, y_test = train_test_split(h.data, h.target, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, h.feature_names

def demo_ols_sklearn(X_train, X_test, y_train, y_test, names):
    print("=== OLS Linear Regression ===")
    pipe = make_pipeline(StandardScaler(), LinearRegression())
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    print(f"  Test RMSE: {mean_squared_error(y_test, y_pred)**0.5:.4f}")
    print(f"  Test R²:   {r2_score(y_test, y_pred):.4f}")
    coef = pipe.named_steps["linearregression"].coef_
    for name, c in sorted(zip(names, coef), key=lambda x: abs(x[1]), reverse=True):
        print(f"    {name:20s}: {c:+.4f}")
    return pipe

def demo_ridge(X_train, X_test, y_train, y_test):
    print("\n=== Ridge Regression ===")
    for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
        pipe = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        pipe.fit(X_train, y_train)
        rmse = mean_squared_error(y_test, pipe.predict(X_test))**0.5
        print(f"  alpha={alpha:>6}: RMSE={rmse:.4f}")

def demo_gd_from_scratch(X_train, X_test, y_train, y_test):
    print("\n=== Gradient Descent from Scratch ===")
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)
    # Add bias column
    Xtr = np.hstack([np.ones((Xtr.shape[0], 1)), Xtr])
    Xte = np.hstack([np.ones((Xte.shape[0], 1)), Xte])
    w = np.zeros(Xtr.shape[1]); lr = 0.01; losses = []
    for i in range(300):
        pred = Xtr @ w
        err  = pred - y_train
        grad = (2 / len(y_train)) * Xtr.T @ err
        w   -= lr * grad
        losses.append(np.mean(err**2))

    te_rmse = mean_squared_error(y_test, Xte @ w)**0.5
    print(f"  After 300 steps: test RMSE={te_rmse:.4f}")
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(losses); ax.set_xlabel("Iteration"); ax.set_ylabel("MSE"); ax.set_title("GD convergence")
    fig.savefig(OUTPUT / "gd_convergence.png", dpi=120, bbox_inches="tight")
    plt.close(fig); print(f"  Saved: gd_convergence.png")

def demo_residuals(pipe, X_test, y_test):
    print("\n=== Residual Analysis ===")
    y_pred = pipe.predict(X_test)
    resid  = y_test - y_pred
    print(f"  Residual mean: {resid.mean():.4f}  std: {resid.std():.4f}")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(y_pred, resid, s=1, alpha=0.3)
    axes[0].axhline(0, color="r"); axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("Residual")
    axes[0].set_title("Residuals vs Fitted")
    axes[1].hist(resid, bins=50); axes[1].set_title("Residual distribution")
    fig.savefig(OUTPUT / "residuals.png", dpi=120, bbox_inches="tight")
    plt.close(fig); print(f"  Saved: residuals.png")

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, names = load_data()
    pipe = demo_ols_sklearn(X_train, X_test, y_train, y_test, names)
    demo_ridge(X_train, X_test, y_train, y_test)
    demo_gd_from_scratch(X_train, X_test, y_train, y_test)
    demo_residuals(pipe, X_test, y_test)
