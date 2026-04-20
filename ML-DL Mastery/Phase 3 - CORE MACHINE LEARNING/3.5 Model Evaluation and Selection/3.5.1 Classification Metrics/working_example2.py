"""
Working Example 2: Classification Metrics — Confusion Matrix, ROC, PR Curve
=============================================================================
precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix, multi-class.

Run:  python working_example2.py
"""
from pathlib import Path
try:
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (classification_report, confusion_matrix,
                                  roc_auc_score, average_precision_score,
                                  roc_curve, precision_recall_curve,
                                  ConfusionMatrixDisplay)
    from sklearn.pipeline import make_pipeline
    import warnings; warnings.filterwarnings("ignore")
except ImportError:
    raise SystemExit("pip install numpy matplotlib scikit-learn")

OUTPUT = Path(__file__).parent / "output"
OUTPUT.mkdir(exist_ok=True)

def load_binary():
    h = fetch_california_housing()
    X, y = h.data, (h.target > np.median(h.target)).astype(int)
    return X, y

def demo_classification_report():
    print("=== Classification Report ===")
    X, y = load_binary()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        stratify=y, random_state=42)
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    print(classification_report(y_test, preds, target_names=["Low", "High"]))

def demo_roc_pr():
    print("=== ROC-AUC and PR-AUC ===")
    X, y = load_binary()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        stratify=y, random_state=42)
    for name, model in [("LogReg", LogisticRegression(max_iter=1000)),
                         ("RF-50",  RandomForestClassifier(n_estimators=50, random_state=42))]:
        pipe = make_pipeline(StandardScaler(), model)
        pipe.fit(X_train, y_train)
        probs = pipe.predict_proba(X_test)[:, 1]
        roc = roc_auc_score(y_test, probs)
        pr  = average_precision_score(y_test, probs)
        print(f"  {name:10s}: ROC-AUC={roc:.4f}  PR-AUC={pr:.4f}")

    # Plot ROC
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for name, model in [("LR", LogisticRegression(max_iter=1000)),
                         ("RF", RandomForestClassifier(n_estimators=50, random_state=42))]:
        pipe = make_pipeline(StandardScaler(), model)
        pipe.fit(X_train, y_train)
        probs = pipe.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, probs)
        axes[0].plot(fpr, tpr, label=name)
        prec, rec, _ = precision_recall_curve(y_test, probs)
        axes[1].plot(rec, prec, label=name)

    axes[0].plot([0,1],[0,1],'k--'); axes[0].set_title("ROC Curve")
    axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR"); axes[0].legend()
    axes[1].set_title("PR Curve")
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision"); axes[1].legend()
    plt.tight_layout(); plt.savefig(OUTPUT / "roc_pr_curves.png"); plt.close()
    print("  Saved roc_pr_curves.png")

def demo_multiclass_metrics():
    """Classification metrics for a 4-class problem (macro vs weighted averaging)."""
    print("\n=== Multiclass Metrics ===")
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=2000, n_classes=4, n_features=10,
                                 n_informative=6, random_state=42)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    pipe = make_pipeline(StandardScaler(),
                          RandomForestClassifier(n_estimators=50, random_state=42))
    pipe.fit(X_tr, y_tr)
    preds = pipe.predict(X_te)
    print(classification_report(y_te, preds, digits=3))
    auc_ovr = roc_auc_score(y_te, pipe.predict_proba(X_te), multi_class="ovr", average="macro")
    auc_ovo = roc_auc_score(y_te, pipe.predict_proba(X_te), multi_class="ovo", average="macro")
    print(f"  ROC-AUC OvR={auc_ovr:.4f}  OvO={auc_ovo:.4f}")


def demo_cohen_kappa():
    """Cohen's kappa measures agreement beyond chance."""
    print("\n=== Cohen's Kappa ===")
    from sklearn.metrics import cohen_kappa_score, matthews_corrcoef
    X, y = load_binary()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    for name, model in [("LR", LogisticRegression(max_iter=1000)),
                          ("RF", RandomForestClassifier(n_estimators=50, random_state=42))]:
        pipe = make_pipeline(StandardScaler(), model)
        pipe.fit(X_tr, y_tr)
        preds = pipe.predict(X_te)
        kappa = cohen_kappa_score(y_te, preds)
        mcc   = matthews_corrcoef(y_te, preds)
        print(f"  {name}: kappa={kappa:.4f}  MCC={mcc:.4f}")


if __name__ == "__main__":
    demo_classification_report()
    demo_roc_pr()
    demo_multiclass_metrics()
    demo_cohen_kappa()
