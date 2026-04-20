"""
Working Example 2: Logistic Regression — Binary & Multiclass, ROC, Cal Housing
================================================================================
Binary classification (high/low price), sigmoid from scratch,
ROC curve, precision-recall, regularisation.

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
    from sklearn.metrics import (classification_report, roc_auc_score,
                                  roc_curve, confusion_matrix)
    from sklearn.pipeline import make_pipeline
except ImportError:
    raise SystemExit("pip install numpy matplotlib scikit-learn")

OUTPUT = Path(__file__).parent / "output"
OUTPUT.mkdir(exist_ok=True)

def load_binary():
    h = fetch_california_housing()
    X, y = h.data, (h.target > np.median(h.target)).astype(int)
    return train_test_split(X, y, test_size=0.2, random_state=42)

def demo_sigmoid():
    print("=== Sigmoid & Log-Loss ===")
    z = np.array([-4., -2., 0., 2., 4.])
    sig = 1 / (1 + np.exp(-z))
    for zi, si in zip(z, sig):
        print(f"  z={zi:+.1f}: sigma(z)={si:.4f}")

def demo_logistic(X_train, X_test, y_train, y_test):
    print("\n=== Logistic Regression (Cal Housing binary) ===")
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    pipe.fit(X_train, y_train)
    y_pred  = pipe.predict(X_test)
    y_prob  = pipe.predict_proba(X_test)[:, 1]
    print(classification_report(y_test, y_pred, digits=4))
    auc = roc_auc_score(y_test, y_prob)
    print(f"  ROC-AUC: {auc:.4f}")

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, lw=2, label=f"AUC={auc:.3f}")
    ax.plot([0,1],[0,1],"--",color="grey"); ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.legend(); ax.set_title("ROC Curve — Logistic Regression")
    fig.savefig(OUTPUT / "roc_curve.png", dpi=120, bbox_inches="tight")
    plt.close(fig); print(f"  Saved: roc_curve.png")
    return pipe

def demo_regularisation(X_train, X_test, y_train, y_test):
    print("\n=== Regularisation Strength (C = 1/lambda) ===")
    for C in [0.001, 0.01, 0.1, 1.0, 10.0]:
        pipe = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=1000))
        pipe.fit(X_train, y_train)
        acc = pipe.score(X_test, y_test)
        print(f"  C={C:>6}: accuracy={acc:.4f}")

def demo_multiclass():
    """One-vs-Rest multiclass logistic regression on synthetic 4-class data."""
    print("\n=== Multiclass Logistic Regression (OvR & Multinomial) ===")
    from sklearn.datasets import make_classification
    from sklearn.metrics import accuracy_score
    X, y = make_classification(n_samples=2000, n_features=10, n_classes=4,
                                 n_informative=6, random_state=42)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    for multi_class, solver in [("ovr", "lbfgs"), ("multinomial", "lbfgs")]:
        pipe = make_pipeline(
            StandardScaler(),
            LogisticRegression(multi_class=multi_class, solver=solver,
                               max_iter=1000, random_state=42),
        )
        pipe.fit(X_tr, y_tr)
        acc = pipe.score(X_te, y_te)
        print(f"  {multi_class:12s}: accuracy={acc:.4f}")


def demo_confusion_heatmap(X_train, X_test, y_train, y_test):
    """Confusion matrix heatmap."""
    print("\n=== Confusion Matrix ===")
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    pipe.fit(X_train, y_train)
    cm = confusion_matrix(y_test, pipe.predict(X_test))
    print(f"  TN={cm[0,0]}  FP={cm[0,1]}  FN={cm[1,0]}  TP={cm[1,1]}")
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14)
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["Pred 0","Pred 1"]); ax.set_yticklabels(["True 0","True 1"])
    ax.set_title("Confusion Matrix")
    fig.savefig(OUTPUT / "confusion_matrix.png", dpi=120, bbox_inches="tight")
    plt.close(fig); print("  Saved: confusion_matrix.png")


def demo_coefficient_analysis(X_train, y_train):
    """Show feature coefficients after fitting logistic regression."""
    print("\n=== Feature Coefficients ===")
    h = fetch_california_housing()
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X_train)
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_s, y_train)
    feature_names = h.feature_names
    coefs = lr.coef_[0]
    order = np.argsort(np.abs(coefs))[::-1]
    for i in order:
        print(f"  {feature_names[i]:12s}: coef={coefs[i]:+.4f}")


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_binary()
    demo_sigmoid()
    demo_logistic(X_train, X_test, y_train, y_test)
    demo_regularisation(X_train, X_test, y_train, y_test)
    demo_multiclass()
    demo_confusion_heatmap(X_train, X_test, y_train, y_test)
    demo_coefficient_analysis(X_train, y_train)
