"""
Working Example 2: Handling Imbalanced Data — class_weight, SMOTE, PR-AUC
===========================================================================
StratifiedKFold, class_weight='balanced', threshold tuning, PR-AUC.

Run:  python working_example2.py
"""
from pathlib import Path
try:
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.metrics import (classification_report, roc_auc_score,
                                  average_precision_score, confusion_matrix,
                                  precision_recall_curve)
    from sklearn.pipeline import make_pipeline
    import warnings; warnings.filterwarnings("ignore")
except ImportError:
    raise SystemExit("pip install numpy matplotlib scikit-learn")

OUTPUT = Path(__file__).parent / "output"
OUTPUT.mkdir(exist_ok=True)

def load_imbalanced():
    """Create imbalanced binary target: top 10% = 1 (rare class)."""
    h = fetch_california_housing()
    X, y_cont = h.data, h.target
    thresh = np.percentile(y_cont, 90)
    y = (y_cont >= thresh).astype(int)
    print(f"  Class distribution: 0={( y==0).sum()} ({(y==0).mean()*100:.1f}%)  "
          f"1={(y==1).sum()} ({(y==1).mean()*100:.1f}%)")
    return X, y

def demo_class_weight():
    print("=== class_weight='balanced' ===")
    X, y = load_imbalanced()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        stratify=y, random_state=42)
    for cw in [None, "balanced"]:
        pipe = make_pipeline(StandardScaler(),
                             LogisticRegression(class_weight=cw, max_iter=1000))
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        probs = pipe.predict_proba(X_test)[:, 1]
        roc   = roc_auc_score(y_test, probs)
        pr    = average_precision_score(y_test, probs)
        tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
        recall = tp / (tp + fn) if (tp + fn) else 0
        print(f"  class_weight={str(cw):10s}: ROC-AUC={roc:.4f}  PR-AUC={pr:.4f}  "
              f"Recall(1)={recall:.3f}  TP={tp}  FN={fn}")

def demo_threshold_tuning():
    print("\n=== Decision Threshold Tuning ===")
    X, y = load_imbalanced()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        stratify=y, random_state=42)
    pipe = make_pipeline(StandardScaler(),
                         LogisticRegression(class_weight="balanced", max_iter=1000))
    pipe.fit(X_train, y_train)
    probs = pipe.predict_proba(X_test)[:, 1]

    print(f"  {'Threshold':>12}  {'Precision':>10}  {'Recall':>8}  {'F1':>8}")
    prec_arr, rec_arr, thresh_arr = precision_recall_curve(y_test, probs)
    for thresh in [0.3, 0.4, 0.5, 0.6, 0.7]:
        preds = (probs >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
        prec  = tp / (tp + fp) if (tp + fp) else 0
        rec   = tp / (tp + fn) if (tp + fn) else 0
        f1    = 2*prec*rec/(prec+rec) if (prec+rec) else 0
        print(f"  {thresh:12.1f}  {prec:10.3f}  {rec:8.3f}  {f1:8.3f}")

def demo_stratified_cv():
    print("\n=== StratifiedKFold CV ===")
    X, y = load_imbalanced()
    pipe = make_pipeline(StandardScaler(),
                         LogisticRegression(class_weight="balanced", max_iter=1000))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipe, X, y, cv=skf, scoring="average_precision")
    print(f"  PR-AUC: {scores.mean():.4f} ± {scores.std():.4f} (5-fold stratified)")

def demo_cost_sensitive():
    """Cost-sensitive learning: vary misclassification cost via sample_weight."""
    print("\n=== Cost-Sensitive Learning ===")
    from sklearn.ensemble import GradientBoostingClassifier
    X, y = load_imbalanced()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        stratify=y, random_state=42)
    # Give minority class 5x weight
    weights = np.where(y_train == 1, 5.0, 1.0)
    gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb.fit(X_train, y_train, sample_weight=weights)
    probs = gb.predict_proba(X_test)[:, 1]
    roc = roc_auc_score(y_test, probs)
    pr  = average_precision_score(y_test, probs)
    print(f"  GBM (cost-sensitive 5x): ROC-AUC={roc:.4f}  PR-AUC={pr:.4f}")


def demo_pr_curve_plot():
    """Plot and compare PR curves for balanced vs unbalanced models."""
    print("\n=== Precision-Recall Curve ===")
    X, y = load_imbalanced()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        stratify=y, random_state=42)
    fig, ax = plt.subplots(figsize=(6, 5))
    for cw, label in [(None, "Unweighted"), ("balanced", "Balanced")]:
        pipe = make_pipeline(StandardScaler(),
                             LogisticRegression(class_weight=cw, max_iter=1000))
        pipe.fit(X_train, y_train)
        probs = pipe.predict_proba(X_test)[:, 1]
        p, r, _ = precision_recall_curve(y_test, probs)
        ap = average_precision_score(y_test, probs)
        ax.plot(r, p, label=f"{label} AP={ap:.3f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve"); ax.legend()
    fig.savefig(OUTPUT / "pr_curve.png", dpi=120, bbox_inches="tight")
    plt.close(fig); print("  Saved: pr_curve.png")


if __name__ == "__main__":
    demo_class_weight()
    demo_threshold_tuning()
    demo_stratified_cv()
    demo_cost_sensitive()
    demo_pr_curve_plot()
