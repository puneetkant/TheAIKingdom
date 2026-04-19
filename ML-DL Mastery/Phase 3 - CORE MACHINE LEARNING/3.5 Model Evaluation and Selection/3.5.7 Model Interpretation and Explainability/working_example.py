"""
Working Example: Model Interpretation and Explainability
Covers feature importance, partial dependence plots, SHAP, LIME,
permutation importance, ICE plots, and global vs local explanations.
"""
import numpy as np
from sklearn.datasets import load_breast_cancer, make_regression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_explainability")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -- 1. Linear model coefficients ---------------------------------------------
def linear_coefficients():
    print("=== Linear Model Coefficients ===")
    cancer = load_breast_cancer()
    X, y   = cancer.data, cancer.target
    scaler = StandardScaler()
    X_s    = scaler.fit_transform(X)
    X_tr, X_te, y_tr, y_te = train_test_split(X_s, y, test_size=0.2, random_state=0)
    lr = LogisticRegression(max_iter=1000).fit(X_tr, y_tr)

    coefs  = lr.coef_[0]
    names  = cancer.feature_names
    idx    = np.argsort(np.abs(coefs))[::-1]
    print(f"  Test accuracy: {lr.score(X_te, y_te):.4f}")
    print(f"\n  {'Feature':<35} {'Coefficient':>12}")
    for i in idx[:10]:
        print(f"  {names[i]:<35} {coefs[i]:>12.4f}")
    print(f"\n  (Coefficients are on standardised scale — comparable)")
    print(f"  Positive coef -> predicts class=1 (malignant=0, benign=1)")


# -- 2. Tree feature importance (MDI) -----------------------------------------
def mdi_importance():
    print("\n=== Mean Decrease Impurity (MDI) Feature Importance ===")
    print("  Accumulated impurity reduction from splits on each feature")
    print("  Biased towards high-cardinality features; use with caution")
    cancer = load_breast_cancer()
    X, y   = cancer.data, cancer.target
    rf     = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)
    rf.fit(X, y)

    imp  = rf.feature_importances_
    idx  = imp.argsort()[::-1]
    std  = np.std([t.feature_importances_ for t in rf.estimators_], axis=0)
    print(f"\n  {'Feature':<35} {'Importance':>12} {'Std':>8}")
    for i in idx[:10]:
        print(f"  {cancer.feature_names[i]:<35} {imp[i]:>12.4f} {std[i]:>8.4f}")


# -- 3. Permutation importance -------------------------------------------------
def perm_importance():
    print("\n=== Permutation Importance ===")
    print("  Drop in score when a feature is randomly shuffled on HELD-OUT set")
    print("  Model-agnostic and unbiased; use on test data only")
    cancer = load_breast_cancer()
    X, y   = cancer.data, cancer.target
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)
    rf  = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1).fit(X_tr, y_tr)

    perm = permutation_importance(rf, X_te, y_te, n_repeats=15, random_state=0, n_jobs=-1)
    idx  = perm.importances_mean.argsort()[::-1]
    print(f"\n  {'Feature':<35} {'Mean drop':>10} {'Std':>8}")
    for i in idx[:10]:
        print(f"  {cancer.feature_names[i]:<35} {perm.importances_mean[i]:>10.4f}"
              f" {perm.importances_std[i]:>8.4f}")


# -- 4. Partial Dependence Plots (PDP) ----------------------------------------
def partial_dependence():
    print("\n=== Partial Dependence Plots (PDP) ===")
    print("  Marginal effect of a feature on prediction (averaged over all other features)")
    cancer = load_breast_cancer()
    X, y   = cancer.data, cancer.target
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)
    gb = GradientBoostingClassifier(n_estimators=100, random_state=0).fit(X_tr, y_tr)

    # PDP for top 2 features
    features = [0, 2]  # feature indices
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    PartialDependenceDisplay.from_estimator(
        gb, X_te, features=features,
        feature_names=cancer.feature_names,
        ax=axes, grid_resolution=40
    )
    plt.suptitle("Partial Dependence Plots", fontsize=12)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "pdp.png")
    plt.savefig(path, dpi=90); plt.close()
    print(f"  PDP saved: {path}")
    print(f"  Features plotted: {cancer.feature_names[0]}, {cancer.feature_names[2]}")


# -- 5. SHAP (SHapley Additive exPlanations) ----------------------------------
def shap_explanation():
    print("\n=== SHAP Values ===")
    print("  Based on Shapley values from cooperative game theory")
    print("  phi_i = contribution of feature i to prediction for a single sample")
    print("  Global: |mean SHAP| per feature    Local: SHAP per instance")
    try:
        import shap
        cancer = load_breast_cancer()
        X, y   = cancer.data, cancer.target
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)
        rf = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1).fit(X_tr, y_tr)

        explainer   = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(X_te)
        # For binary: shap_values is list [class0, class1]
        sv_class1   = shap_values[1] if isinstance(shap_values, list) else shap_values

        mean_shap = np.abs(sv_class1).mean(axis=0)
        idx       = mean_shap.argsort()[::-1]
        print(f"\n  Mean |SHAP| (global feature importance):")
        for i in idx[:8]:
            print(f"    {cancer.feature_names[i]:<35}: {mean_shap[i]:.4f}")

        # Local explanation for one sample
        print(f"\n  SHAP for sample 0 (pred={rf.predict_proba(X_te[:1])[:,1][0]:.3f}):")
        for i in idx[:5]:
            print(f"    {cancer.feature_names[i]:<35}: {sv_class1[0, i]:+.4f}")

    except ImportError:
        print("  shap not installed. Install: pip install shap")
        print("  Key properties:")
        print("    Efficiency:       Sigma phi_i = f(x) - E[f(X)]")
        print("    Symmetry:         Equal contributors get equal SHAP")
        print("    Dummy:            Zero-contribution features get phi=0")
        print("    Additivity:       SHAP values add across ensemble models")


# -- 6. LIME (Local Interpretable Model-agnostic Explanations) -----------------
def lime_explanation():
    print("\n=== LIME (Local Interpretable Model-agnostic Explanations) ===")
    print("  Fit a simple (linear) surrogate model around a single prediction")
    print("  Works for any black-box model")
    print()
    try:
        import lime
        import lime.lime_tabular
        cancer = load_breast_cancer()
        X, y   = cancer.data, cancer.target
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)
        rf = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1).fit(X_tr, y_tr)

        explainer = lime.lime_tabular.LimeTabularExplainer(
            X_tr, feature_names=cancer.feature_names,
            class_names=cancer.target_names, discretize_continuous=True
        )
        exp = explainer.explain_instance(X_te[0], rf.predict_proba, num_features=5)
        print(f"  LIME explanation for sample 0 (top 5 features):")
        for feat, weight in exp.as_list():
            print(f"    {feat:<50}: {weight:+.4f}")

    except ImportError:
        print("  lime not installed. Install: pip install lime")
        print("  Algorithm:")
        print("    1. For instance x*, generate perturbed samples N(x*)")
        print("    2. Weight samples by proximity to x* (RBF kernel)")
        print("    3. Fit weighted linear model on perturbed samples")
        print("    4. Coefficients = local feature attributions")


# -- 7. Global vs Local explanations summary -----------------------------------
def explanation_summary():
    print("\n=== Global vs Local Explanations ===")
    print(f"  {'Method':<30} {'Scope':<10} {'Model':<15} {'Notes'}")
    rows = [
        ("Coefficients",          "Global", "Linear",    "Interpretable by design"),
        ("MDI importance",        "Global", "Tree",      "Biased for high-cardinality"),
        ("Permutation importance", "Global","Any",       "Use on test set"),
        ("PDP",                   "Global", "Any",       "Averages over all samples"),
        ("ICE plots",             "Local",  "Any",       "PDP per instance"),
        ("SHAP",                  "Both",   "Any",       "Theoretically principled"),
        ("LIME",                  "Local",  "Any",       "Fast approximation"),
        ("Counterfactuals",       "Local",  "Any",       "'What-if' explanations"),
    ]
    for row in rows:
        print(f"  {row[0]:<30} {row[1]:<10} {row[2]:<15} {row[3]}")


if __name__ == "__main__":
    linear_coefficients()
    mdi_importance()
    perm_importance()
    partial_dependence()
    shap_explanation()
    lime_explanation()
    explanation_summary()
