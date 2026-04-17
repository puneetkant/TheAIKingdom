"""
Working Example: Linear Regression
Covers OLS, geometric interpretation, analytical solution, gradient descent,
assumptions, diagnostics, Ridge, Lasso, Elastic Net, and polynomial regression.
"""
import numpy as np
from scipy import stats
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_linreg")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── 1. Analytical OLS solution ────────────────────────────────────────────────
def ols_analytical():
    print("=== Ordinary Least Squares (Analytical) ===")
    rng = np.random.default_rng(0)
    n   = 100
    x   = rng.uniform(0, 10, n)
    y   = 2.5 * x + 1.0 + rng.normal(0, 2, n)

    # Design matrix with intercept
    X = np.column_stack([np.ones(n), x])
    # Normal equations: w = (XᵀX)⁻¹Xᵀy
    w = np.linalg.solve(X.T @ X, X.T @ y)
    print(f"  True: w=[1.0, 2.5]")
    print(f"  OLS:  w={w.round(4)}")

    y_hat = X @ w
    ss_res = np.sum((y - y_hat)**2)
    ss_tot = np.sum((y - y.mean())**2)
    r2     = 1 - ss_res/ss_tot
    rmse   = np.sqrt(ss_res / n)
    print(f"  R²={r2:.4f}  RMSE={rmse:.4f}")

    # Confidence intervals via standard errors
    sigma2_hat = ss_res / (n - 2)
    se         = np.sqrt(sigma2_hat * np.diag(np.linalg.inv(X.T @ X)))
    t_crit     = stats.t.ppf(0.975, n-2)
    print(f"\n  95% CI for intercept: [{w[0]-t_crit*se[0]:.4f}, {w[0]+t_crit*se[0]:.4f}]")
    print(f"  95% CI for slope:     [{w[1]-t_crit*se[1]:.4f}, {w[1]+t_crit*se[1]:.4f}]")

    # Compare with scipy
    slope_sp, inter_sp, r_sp, p_sp, se_sp = stats.linregress(x, y)
    print(f"\n  scipy.linregress: slope={slope_sp:.4f}  intercept={inter_sp:.4f}  p={p_sp:.2e}")


# ── 2. Gradient descent for linear regression ────────────────────────────────
def gradient_descent_linreg():
    print("\n=== Gradient Descent for Linear Regression ===")
    rng = np.random.default_rng(1)
    n   = 80
    x   = rng.uniform(0, 5, n)
    y   = 3*x - 1 + rng.normal(0, 1, n)
    X   = np.column_stack([np.ones(n), x])

    w   = np.zeros(2)
    lr  = 0.01
    epochs = 1000

    losses = []
    for e in range(epochs):
        y_pred = X @ w
        resid  = y_pred - y
        grad   = (2/n) * X.T @ resid
        w      -= lr * grad
        loss   = np.mean(resid**2)
        losses.append(loss)

    w_ols = np.linalg.solve(X.T @ X, X.T @ y)
    print(f"  GD (lr={lr}, {epochs} epochs): w={w.round(4)}  MSE={losses[-1]:.4f}")
    print(f"  OLS (analytical):            w={w_ols.round(4)}  MSE={np.mean((X@w_ols - y)**2):.4f}")


# ── 3. Multiple linear regression assumptions ────────────────────────────────
def assumptions_diagnostics():
    print("\n=== Gauss-Markov Assumptions and Diagnostics ===")
    rng = np.random.default_rng(2)
    n   = 150
    X_raw = rng.standard_normal((n, 3))
    w_true = np.array([1.5, -2.0, 0.8])
    y   = X_raw @ w_true + rng.normal(0, 1.0, n)

    X   = np.column_stack([np.ones(n), X_raw])
    w   = np.linalg.solve(X.T @ X, X.T @ y)
    resid = y - X @ w

    # 1. Zero mean residuals
    print(f"  1. E[ε] ≈ 0: {resid.mean():.6f}")

    # 2. Homoscedasticity (Breusch-Pagan proxy)
    y_hat = X @ w
    _, bp_p, _, _ = np.linalg.lstsq(
        np.column_stack([np.ones(n), y_hat]), resid**2, rcond=None)
    # Simple: correlation of |resid| with fitted
    corr = np.corrcoef(y_hat, np.abs(resid))[0,1]
    print(f"  2. Homoscedasticity: corr(ŷ, |ε|)={corr:.4f}  (small → good)")

    # 3. Normality of residuals (Shapiro-Wilk)
    _, sw_p = stats.shapiro(resid)
    print(f"  3. Normality: Shapiro-Wilk p={sw_p:.4f}  ({'pass' if sw_p>0.05 else 'fail'})")

    # 4. No multicollinearity (VIF)
    def vif(X_df):
        vifs = []
        for i in range(1, X_df.shape[1]):
            X_other = np.delete(X_df[:,1:], i-1, axis=1)
            X_other = np.column_stack([np.ones(n), X_other])
            w_vif = np.linalg.solve(X_other.T @ X_other, X_other.T @ X_df[:, i])
            r2    = 1 - np.var(X_df[:,i] - X_other @ w_vif) / np.var(X_df[:,i])
            vifs.append(1/(1-r2) if r2 < 1 else np.inf)
        return vifs
    print(f"  4. VIF: {[round(v,3) for v in vif(X)]}  (>10 → collinearity)")


# ── 4. Ridge, Lasso, Elastic Net ────────────────────────────────────────────
def regularised_regression():
    print("\n=== Regularised Regression ===")
    from sklearn.linear_model import Ridge, Lasso, ElasticNet
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

    rng = np.random.default_rng(3)
    n, p = 80, 30
    w_true = np.zeros(p); w_true[:5] = [3,-2,1.5,-1,0.8]
    X = rng.standard_normal((n, p))
    y = X @ w_true + rng.normal(0, 1, n)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)
    scaler = StandardScaler().fit(X_tr)
    X_tr_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_te)

    print(f"  {'Model':<20} {'Test RMSE':<14} {'Non-zero coefs'}")
    for name, model in [
        ("OLS (np.lstsq)",    None),
        ("Ridge (α=1)",       Ridge(alpha=1)),
        ("Lasso (α=0.1)",     Lasso(alpha=0.1, max_iter=5000)),
        ("ElasticNet",        ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000)),
    ]:
        if model is None:
            coef, _, _, _ = np.linalg.lstsq(X_tr_s, y_tr, rcond=None)
            y_pred = X_te_s @ coef
        else:
            model.fit(X_tr_s, y_tr)
            coef   = model.coef_
            y_pred = model.predict(X_te_s)
        rmse = np.sqrt(np.mean((y_pred - y_te)**2))
        nnz  = (np.abs(coef) > 1e-6).sum()
        print(f"  {name:<20} {rmse:<14.4f} {nnz}/{p}")


# ── 5. Polynomial regression ─────────────────────────────────────────────────
def polynomial_regression():
    print("\n=== Polynomial Regression ===")
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline

    rng  = np.random.default_rng(4)
    x    = rng.uniform(-1, 1, 60)
    y    = 0.5*x**3 - x**2 + 0.2*x + rng.normal(0, 0.1, 60)
    x_t  = np.linspace(-1, 1, 200)
    y_t  = 0.5*x_t**3 - x_t**2 + 0.2*x_t

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, d in zip(axes, [1, 3, 6, 12]):
        model = make_pipeline(PolynomialFeatures(d), Ridge(alpha=1e-6))
        model.fit(x.reshape(-1,1), y)
        y_pred = model.predict(x_t.reshape(-1,1))
        test_rmse = np.sqrt(np.mean((y_pred - y_t)**2))
        ax.scatter(x, y, s=15, alpha=0.6)
        ax.plot(x_t, y_t, 'k--', lw=1, label='True')
        ax.plot(x_t, y_pred, 'r-', lw=2, label=f'd={d}')
        ax.set(title=f"d={d}  RMSE={test_rmse:.3f}", ylim=(-2.5, 2.5))
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "polynomial_regression.png")
    plt.savefig(path, dpi=90); plt.close()
    print(f"  Saved: {path}")

    print(f"\n  {'Degree':<8} {'Train RMSE':<14} {'Test RMSE'}")
    for d in [1, 2, 3, 4, 6, 9, 12]:
        m = make_pipeline(PolynomialFeatures(d), Ridge(alpha=1e-6))
        m.fit(x.reshape(-1,1), y)
        tr = np.sqrt(np.mean((m.predict(x.reshape(-1,1)) - y)**2))
        te = np.sqrt(np.mean((m.predict(x_t.reshape(-1,1)) - y_t)**2))
        print(f"  {d:<8} {tr:<14.4f} {te:.4f}")


if __name__ == "__main__":
    ols_analytical()
    gradient_descent_linreg()
    assumptions_diagnostics()
    regularised_regression()
    polynomial_regression()
