"""
Working Example 2: Model Interpretation — SHAP-style, permutation importance, partial dependence
=================================================================================================
permutation_importance, PartialDependenceDisplay, manual SHAP-style analysis.

Run:  python working_example2.py
"""
from pathlib import Path
try:
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.inspection import permutation_importance, PartialDependenceDisplay
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import mean_squared_error
    import warnings; warnings.filterwarnings("ignore")
except ImportError:
    raise SystemExit("pip install numpy matplotlib scikit-learn")

OUTPUT = Path(__file__).parent / "output"
OUTPUT.mkdir(exist_ok=True)

def demo_permutation_importance():
    print("=== Permutation Feature Importance ===")
    h = fetch_california_housing()
    X, y = h.data, h.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    result = permutation_importance(rf, X_test, y_test, n_repeats=15, random_state=42)
    order = result.importances_mean.argsort()[::-1]
    print(f"  {'Feature':20s}  {'Mean Imp':>10}  {'Std':>8}")
    for i in order:
        print(f"  {h.feature_names[i]:20s}  {result.importances_mean[i]:>10.4f}  "
              f"{result.importances_std[i]:>8.4f}")

def demo_partial_dependence():
    print("\n=== Partial Dependence Plots ===")
    h = fetch_california_housing()
    X, y = h.data, h.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    gbm = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gbm.fit(X_train, y_train)

    # Top-2 features by permutation importance
    perm = permutation_importance(gbm, X_test, y_test, n_repeats=5, random_state=42)
    top2 = perm.importances_mean.argsort()[-2:][::-1].tolist()
    print(f"  Top-2 features: {[h.feature_names[i] for i in top2]}")

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    PartialDependenceDisplay.from_estimator(gbm, X_train, top2, ax=ax,
                                             feature_names=h.feature_names)
    plt.tight_layout(); plt.savefig(OUTPUT / "partial_dependence.png"); plt.close()
    print("  Saved partial_dependence.png")

def demo_shap_manual():
    """Manual approximation: compare model output with/without each feature."""
    print("\n=== Manual Feature Attribution (leave-one-out) ===")
    h = fetch_california_housing()
    X, y = h.data, h.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipe = make_pipeline(StandardScaler(), GradientBoostingRegressor(n_estimators=50, random_state=42))
    pipe.fit(X_train, y_train)
    base_rmse = mean_squared_error(y_test, pipe.predict(X_test))**0.5

    print(f"  Base RMSE: {base_rmse:.4f}")
    print(f"  {'Feature':20s}  {'RMSE drop':>12}  {'Importance':>12}")
    rng = np.random.default_rng(42)
    for i, name in enumerate(h.feature_names):
        X_perm = X_test.copy()
        X_perm[:, i] = rng.permutation(X_perm[:, i])
        rmse_perm = mean_squared_error(y_test, pipe.predict(X_perm))**0.5
        drop = rmse_perm - base_rmse
        print(f"  {name:20s}  {drop:>12.4f}  {'*' * max(0, int(drop*20))}")

def demo_ice_plots():
    """Individual Conditional Expectation (ICE) plots show per-instance PD."""
    print("\n=== ICE Plots (Individual Conditional Expectation) ===")
    h = fetch_california_housing()
    X, y = h.data, h.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    gbm = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gbm.fit(X_train, y_train)
    # Feature 0 = MedInc (most important)
    perm = permutation_importance(gbm, X_test, y_test, n_repeats=5, random_state=42)
    top_feat = int(perm.importances_mean.argmax())
    print(f"  Plotting ICE for feature: {h.feature_names[top_feat]}")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    PartialDependenceDisplay.from_estimator(
        gbm, X_train, [top_feat], kind="both", ax=axes[0],
        feature_names=h.feature_names, subsample=100, random_state=42,
    )
    PartialDependenceDisplay.from_estimator(
        gbm, X_train, [top_feat], kind="average", ax=axes[1],
        feature_names=h.feature_names,
    )
    axes[0].set_title("ICE + PDP"); axes[1].set_title("PDP only")
    plt.tight_layout(); fig.savefig(OUTPUT / "ice_plots.png", dpi=120, bbox_inches="tight")
    plt.close(fig); print("  Saved: ice_plots.png")


def demo_feature_interaction():
    """2D partial dependence reveals feature interactions."""
    print("\n=== 2D Partial Dependence (interaction) ===")
    h = fetch_california_housing()
    X, y = h.data, h.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    gbm = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gbm.fit(X_train, y_train)
    # MedInc (0) x Latitude (6)
    fig, ax = plt.subplots(figsize=(6, 5))
    PartialDependenceDisplay.from_estimator(
        gbm, X_train, [(0, 6)], ax=ax, feature_names=h.feature_names,
    )
    ax.set_title("2D PDP: MedInc x Latitude")
    fig.savefig(OUTPUT / "pdp_2d.png", dpi=120, bbox_inches="tight")
    plt.close(fig); print("  Saved: pdp_2d.png")


if __name__ == "__main__":
    demo_permutation_importance()
    demo_partial_dependence()
    demo_shap_manual()
    demo_ice_plots()
    demo_feature_interaction()
