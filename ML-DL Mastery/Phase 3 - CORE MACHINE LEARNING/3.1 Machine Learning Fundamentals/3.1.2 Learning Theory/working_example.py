"""
Working Example: Learning Theory
Covers generalisation bounds, regularisation theory, VC dimension,
Rademacher complexity, double descent, and structural risk minimisation.
"""
import numpy as np
from scipy import stats
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_theory")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── 1. Generalisation bound (finite hypothesis class) ────────────────────────
def generalisation_bound():
    print("=== Generalisation Bound (finite H) ===")
    print("  P(|R(h) - R̂(h)| > ε) ≤ 2|H| exp(-2nε²)")
    print("  Rearranged: R(h) ≤ R̂(h) + √(log(2|H|/δ) / (2n))")
    print()

    delta = 0.05
    for H_size in [10, 100, 1000]:
        print(f"  |H|={H_size}, δ={delta}:")
        for n in [100, 500, 1000, 5000]:
            bound = np.sqrt(np.log(2*H_size/delta) / (2*n))
            print(f"    n={n:5d}: complexity term = {bound:.4f}")
        print()


# ── 2. Regularisation: Ridge vs Lasso ────────────────────────────────────────
def regularisation_comparison():
    print("=== L2 (Ridge) vs L1 (Lasso) Regularisation ===")
    rng = np.random.default_rng(0)
    n, p = 100, 50

    # Sparse true weights (only 5 are non-zero)
    w_true = np.zeros(p)
    w_true[:5] = rng.normal(0, 3, 5)

    X = rng.standard_normal((n, p))
    y = X @ w_true + rng.normal(0, 0.5, n)

    alphas = [0.001, 0.01, 0.1, 1.0, 10.0]
    print(f"  True weights: {(w_true != 0).sum()} non-zero out of {p}")
    print(f"\n  {'α':<8} {'Ridge MSE':<12} {'Ridge ||w||₂':<15} {'Lasso MSE':<12} {'Lasso nonzero'}")

    for alpha in alphas:
        ridge = Ridge(alpha=alpha).fit(X, y)
        lasso = Lasso(alpha=alpha, max_iter=5000).fit(X, y)
        r_mse = np.mean((ridge.coef_ - w_true)**2)
        l_mse = np.mean((lasso.coef_ - w_true)**2)
        r_l2  = np.linalg.norm(ridge.coef_)
        l_nnz = (np.abs(lasso.coef_) > 1e-6).sum()
        print(f"  {alpha:<8} {r_mse:<12.4f} {r_l2:<15.4f} {l_mse:<12.4f} {l_nnz}")


# ── 3. Bias-variance as function of regularisation strength ──────────────────
def regularisation_bias_variance():
    print("\n=== Bias-Variance vs Regularisation Strength (Ridge) ===")
    rng = np.random.default_rng(1)
    n   = 40
    p   = 20
    M   = 300   # experiments

    w_true = np.ones(p) * 2
    X_test = rng.standard_normal((500, p))
    y_test = X_test @ w_true

    alphas = np.logspace(-3, 3, 13)
    print(f"  {'α':<12} {'Bias²':<12} {'Variance':<12} {'MSE'}")
    for alpha in alphas:
        preds = []
        for _ in range(M):
            X_tr = rng.standard_normal((n, p))
            y_tr = X_tr @ w_true + rng.normal(0, 1, n)
            model = Ridge(alpha=alpha).fit(X_tr, y_tr)
            preds.append(model.predict(X_test))

        preds   = np.array(preds)
        bias2   = np.mean((preds.mean(0) - y_test)**2)
        variance = np.mean(preds.var(0))
        mse     = bias2 + variance + 1.0  # + noise var
        print(f"  {alpha:<12.3e} {bias2:<12.4f} {variance:<12.4f} {mse:.4f}")


# ── 4. Structural Risk Minimisation ──────────────────────────────────────────
def structural_risk_minimisation():
    print("\n=== Structural Risk Minimisation ===")
    print("  Choose model: min_h {R̂(h) + Ω(h)}  where Ω is complexity penalty")
    print()
    rng = np.random.default_rng(2)
    x = rng.uniform(0, 1, 60)
    y = np.sin(2*np.pi*x) + rng.normal(0, 0.2, 60)
    X = x.reshape(-1, 1)

    x_v = rng.uniform(0, 1, 200)
    y_v = np.sin(2*np.pi*x_v)
    X_v = x_v.reshape(-1, 1)

    print(f"  {'Degree':<8} {'Train RMSE':<14} {'Val RMSE':<14} {'AIC':<12} {'BIC'}")
    for d in range(1, 14):
        poly  = PolynomialFeatures(d)
        X_p   = poly.fit_transform(X)
        X_pv  = poly.transform(X_v)
        model = Ridge(alpha=1e-6).fit(X_p, y)
        y_hat = model.predict(X_p)
        y_vhat = model.predict(X_pv)
        n     = len(y)
        k     = d + 1   # params
        rss   = np.sum((y - y_hat)**2)
        train_rmse = np.sqrt(rss/n)
        val_rmse   = np.sqrt(np.mean((y_v - y_vhat)**2))
        # AIC / BIC (using Gaussian likelihood)
        aic = 2*k + n*np.log(rss/n)
        bic = k*np.log(n) + n*np.log(rss/n)
        print(f"  {d:<8} {train_rmse:<14.4f} {val_rmse:<14.4f} {aic:<12.1f} {bic:.1f}")


# ── 5. Double descent phenomenon ─────────────────────────────────────────────
def double_descent():
    print("\n=== Double Descent (model complexity curve) ===")
    print("  Classical: bias-var tradeoff → single optimal complexity")
    print("  Modern ML: overparameterised models can generalise (interpolation regime)")
    print()
    rng  = np.random.default_rng(3)
    n    = 50
    noise= 0.3

    x    = rng.uniform(0, 1, n)
    y    = np.sin(4*np.pi*x) + rng.normal(0, noise, n)
    x_t  = np.linspace(0, 1, 200)
    y_t  = np.sin(4*np.pi*x_t)

    print(f"  {'d':<6} {'Train RMSE':<14} {'Test RMSE':<14} {'Regime'}")
    for d in [1,3,5,10,15,20,30,49,50,60]:
        try:
            poly = PolynomialFeatures(min(d, n-1))
            X_p  = poly.fit_transform(x.reshape(-1,1))
            X_tp = poly.transform(x_t.reshape(-1,1))
            # Use minimum-norm solution for overparameterised case
            if X_p.shape[1] > n:
                # Pseudo-inverse (min norm)
                coef, _, _, _ = np.linalg.lstsq(X_p, y, rcond=None)
            else:
                coef = np.linalg.lstsq(X_p, y, rcond=None)[0]
            tr = np.sqrt(np.mean((X_p @ coef - y)**2))
            te = np.sqrt(np.mean((X_tp @ coef - y_t)**2))
            k  = X_p.shape[1]
            regime = "underfit" if d<=3 else ("interp." if k>=n else "classical")
            print(f"  {d:<6} {tr:<14.4f} {te:<14.4f} {regime}  (k={k})")
        except Exception:
            pass


if __name__ == "__main__":
    generalisation_bound()
    regularisation_comparison()
    regularisation_bias_variance()
    structural_risk_minimisation()
    double_descent()
