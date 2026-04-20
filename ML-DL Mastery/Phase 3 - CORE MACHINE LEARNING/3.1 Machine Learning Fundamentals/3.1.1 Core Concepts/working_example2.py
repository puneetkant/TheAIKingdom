"""
Working Example 2: ML Core Concepts — Bias-Variance, Overfitting, Train/Val/Test
=================================================================================
Demonstrates learning curves, bias-variance tradeoff with polynomial regression,
and train/val/test splits using California Housing (sklearn).

Run:  python working_example2.py
"""
from pathlib import Path
try:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split, learning_curve
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.metrics import mean_squared_error
except ImportError:
    raise SystemExit("pip install numpy matplotlib scikit-learn")

OUTPUT = Path(__file__).parent / "output"
OUTPUT.mkdir(exist_ok=True)

def demo_splits():
    print("=== Train / Validation / Test Split ===")
    housing = fetch_california_housing()
    X, y = housing.data, housing.target
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.15/0.85, random_state=42)
    print(f"  Train: {X_train.shape[0]}  Val: {X_val.shape[0]}  Test: {X_test.shape[0]}")
    return X_train, X_val, X_test, y_train, y_val, y_test

def demo_bias_variance(X_train, X_test, y_train, y_test):
    print("\n=== Bias-Variance Tradeoff (Polynomial Degree) ===")
    scaler = StandardScaler().fit(X_train)
    # Use just one feature (MedInc) for polynomial demo
    feat = 0
    Xf_tr = scaler.transform(X_train)[:, feat:feat+1]
    Xf_te = scaler.transform(X_test)[:, feat:feat+1]

    results = []
    for d in [1, 2, 3, 5, 9]:
        pipe = make_pipeline(PolynomialFeatures(d), Ridge(alpha=1e-6))
        pipe.fit(Xf_tr, y_train)
        tr_mse = mean_squared_error(y_train, pipe.predict(Xf_tr))
        te_mse = mean_squared_error(y_test, pipe.predict(Xf_te))
        results.append((d, tr_mse, te_mse))
        print(f"  Degree {d}: train MSE={tr_mse:.4f}  test MSE={te_mse:.4f}")

    degrees = [r[0] for r in results]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(degrees, [r[1] for r in results], "o-", label="Train MSE")
    ax.plot(degrees, [r[2] for r in results], "s-", label="Test MSE")
    ax.set_xlabel("Polynomial degree"); ax.set_ylabel("MSE")
    ax.set_title("Bias-Variance Tradeoff"); ax.legend()
    fig.savefig(OUTPUT / "bias_variance.png", dpi=120, bbox_inches="tight")
    plt.close(fig); print(f"  Saved: bias_variance.png")

def demo_learning_curves(X_train, y_train):
    print("\n=== Learning Curves ===")
    pipe = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    train_sizes, train_scores, val_scores = learning_curve(
        pipe, X_train, y_train, cv=5, scoring="neg_mean_squared_error",
        train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1
    )
    tr_mse = -train_scores.mean(axis=1)
    va_mse = -val_scores.mean(axis=1)
    print(f"  Max train size: {train_sizes[-1]}  Val MSE: {va_mse[-1]:.4f}")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(train_sizes, tr_mse, "o-", label="Train MSE")
    ax.plot(train_sizes, va_mse, "s-", label="Val MSE (CV)")
    ax.set_xlabel("Training set size"); ax.set_ylabel("MSE")
    ax.set_title("Learning Curves (Ridge)"); ax.legend()
    fig.savefig(OUTPUT / "learning_curves.png", dpi=120, bbox_inches="tight")
    plt.close(fig); print(f"  Saved: learning_curves.png")

def demo_bias_variance_sine():
    print("\n=== Bias-Variance Tradeoff on Noisy Sinusoidal Data ===")
    rng = np.random.default_rng(42)
    n = 100
    X_all = np.linspace(0, 2 * np.pi, n).reshape(-1, 1)
    y_all = np.sin(X_all.ravel()) + rng.normal(0, 0.3, n)
    X_tr, X_te, y_tr, y_te = train_test_split(X_all, y_all, test_size=0.3, random_state=42)
    degrees = [1, 3, 5, 10]
    tr_mses, te_mses = [], []
    for d in degrees:
        pipe = make_pipeline(PolynomialFeatures(d), Ridge(alpha=1e-6))
        pipe.fit(X_tr, y_tr)
        tr = mean_squared_error(y_tr, pipe.predict(X_tr))
        te = mean_squared_error(y_te, pipe.predict(X_te))
        tr_mses.append(tr); te_mses.append(te)
        print(f"  Degree {d:2d}: train MSE={tr:.4f}  test MSE={te:.4f}")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(degrees, tr_mses, "o-", label="Train MSE")
    ax.plot(degrees, te_mses, "s-", label="Test MSE")
    ax.set_xlabel("Polynomial degree"); ax.set_ylabel("MSE")
    ax.set_title("Bias-Variance Tradeoff (Sinusoidal)"); ax.legend()
    fig.savefig(OUTPUT / "bias_variance.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: bias_variance.png")


def demo_learning_curves_compare():
    print("\n=== Learning Curves: Ridge vs RandomForest (Cal Housing) ===")
    housing = fetch_california_housing()
    X, y = housing.data, housing.target
    train_sizes = np.linspace(0.1, 1.0, 8)
    models = [
        ("Ridge",        make_pipeline(StandardScaler(), Ridge(1.0))),
        ("RandomForest", RandomForestRegressor(n_estimators=20, random_state=42, n_jobs=-1)),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, (name, model) in zip(axes, models):
        ts, tr_sc, va_sc = learning_curve(
            model, X, y, cv=3, scoring="neg_mean_squared_error",
            train_sizes=train_sizes, n_jobs=-1
        )
        tr_mse = -tr_sc.mean(axis=1); va_mse = -va_sc.mean(axis=1)
        print(f"  {name}: final val MSE={va_mse[-1]:.4f}")
        ax.plot(ts, tr_mse, "o-", label="Train MSE")
        ax.plot(ts, va_mse, "s-", label="Val MSE")
        ax.set_title(f"Learning Curve: {name}")
        ax.set_xlabel("Training set size"); ax.set_ylabel("MSE"); ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT / "learning_curves.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: learning_curves.png")


def demo_regularization():
    print("\n=== Ridge Regularization: L2 Norm of Coefficients vs Alpha ===")
    housing = fetch_california_housing()
    X, y = housing.data, housing.target
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    sc = StandardScaler().fit(X_tr)
    X_tr_s = sc.transform(X_tr); X_te_s = sc.transform(X_te)
    print(f"  {'alpha':>8}  {'coef_L2_norm':>14}  {'test_MSE':>10}")
    for a in [0.001, 0.01, 0.1, 1, 10]:
        model = Ridge(alpha=a).fit(X_tr_s, y_tr)
        l2  = np.linalg.norm(model.coef_)
        mse = mean_squared_error(y_te, model.predict(X_te_s))
        print(f"  {a:8.3f}  {l2:14.6f}  {mse:10.4f}")


if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test = demo_splits()
    demo_bias_variance(X_train, X_test, y_train, y_test)
    demo_learning_curves(X_train, y_train)
    demo_bias_variance_sine()
    demo_learning_curves_compare()
    demo_regularization()
