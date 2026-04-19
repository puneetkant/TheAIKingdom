"""
Working Example: Classification Metrics
Covers accuracy, precision, recall, F1, confusion matrix, ROC-AUC, PR-AUC,
multi-class extensions (macro/micro/weighted), and Matthews Correlation Coefficient.
"""
import numpy as np
from sklearn.datasets import make_classification, load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, roc_auc_score,
                              average_precision_score, precision_recall_curve,
                              roc_curve, classification_report,
                              matthews_corrcoef, balanced_accuracy_score,
                              cohen_kappa_score, log_loss)
from sklearn.preprocessing import label_binarize
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_clf_metrics")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -- 1. Core binary classification metrics ------------------------------------
def binary_metrics():
    print("=== Binary Classification Metrics ===")
    rng  = np.random.default_rng(0)
    X, y = make_classification(n_samples=500, n_features=10, n_informative=5,
                                random_state=0)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)
    model = LogisticRegression(max_iter=500).fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:,1]

    cm = confusion_matrix(y_te, y_pred)
    TN, FP, FN, TP = cm.ravel()
    print(f"  Confusion Matrix:")
    print(f"    TN={TN:4d}  FP={FP:4d}")
    print(f"    FN={FN:4d}  TP={TP:4d}")
    print()

    acc   = accuracy_score(y_te, y_pred)
    prec  = precision_score(y_te, y_pred)
    rec   = recall_score(y_te, y_pred)
    f1    = f1_score(y_te, y_pred)
    spec  = TN / (TN + FP)   # specificity (true negative rate)
    npv   = TN / (TN + FN)   # negative predictive value
    auc   = roc_auc_score(y_te, y_prob)
    ap    = average_precision_score(y_te, y_prob)
    mcc   = matthews_corrcoef(y_te, y_pred)
    kappa = cohen_kappa_score(y_te, y_pred)
    bal   = balanced_accuracy_score(y_te, y_pred)
    ll    = log_loss(y_te, y_prob)

    print(f"  {'Metric':<28} {'Value'}")
    for name, val in [
        ("Accuracy",               acc),
        ("Balanced Accuracy",      bal),
        ("Precision (PPV)",        prec),
        ("Recall (Sensitivity/TPR)", rec),
        ("Specificity (TNR)",      spec),
        ("Negative Predictive Value (NPV)", npv),
        ("F1 Score",               f1),
        ("Matthews Corr. Coeff.",  mcc),
        ("Cohen's Kappa",          kappa),
        ("ROC-AUC",                auc),
        ("PR-AUC (Avg Precision)", ap),
        ("Log Loss",               ll),
    ]:
        print(f"  {name:<36}: {val:.4f}")

    print(f"\n  Formulas:")
    print(f"    Precision  = TP/(TP+FP) = {TP}/{TP+FP} = {prec:.4f}")
    print(f"    Recall     = TP/(TP+FN) = {TP}/{TP+FN} = {rec:.4f}")
    print(f"    F1         = 2·(P·R)/(P+R) = {f1:.4f}")
    print(f"    MCC        = (TP·TN - FP·FN) / sqrt[(TP+FP)(TP+FN)(TN+FP)(TN+FN)] = {mcc:.4f}")


# -- 2. ROC and PR curves -----------------------------------------------------
def roc_pr_curves():
    print("\n=== ROC and Precision-Recall Curves ===")
    X, y = make_classification(n_samples=500, n_features=10, n_informative=5,
                                random_state=1, weights=[0.8, 0.2])
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)

    models = {
        "Logistic Reg":     LogisticRegression(max_iter=500),
        "Random Forest":    RandomForestClassifier(n_estimators=50, random_state=0),
        "Random baseline":  None,
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for name, m in models.items():
        if m is None:
            y_prob = np.random.default_rng(0).uniform(0, 1, len(y_te))
        else:
            m.fit(X_tr, y_tr)
            y_prob = m.predict_proba(X_te)[:, 1]
        fpr, tpr, _  = roc_curve(y_te, y_prob)
        prec, rec, _ = precision_recall_curve(y_te, y_prob)
        auc = roc_auc_score(y_te, y_prob)
        ap  = average_precision_score(y_te, y_prob)
        print(f"  {name:<20}: AUC={auc:.4f}  AP={ap:.4f}")
        ax1.plot(fpr, tpr, lw=2, label=f"{name} (AUC={auc:.3f})")
        ax2.plot(rec, prec, lw=2, label=f"{name} (AP={ap:.3f})")

    ax1.plot([0,1],[0,1],'k--'); ax1.set(xlabel="FPR",ylabel="TPR",title="ROC Curve")
    ax2.axhline(y_te.mean(), color='k', linestyle='--', label="Baseline")
    ax2.set(xlabel="Recall", ylabel="Precision", title="PR Curve")
    ax1.legend(fontsize=8); ax2.legend(fontsize=8)
    ax1.grid(True, alpha=0.3); ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "roc_pr_curves.png")
    plt.savefig(path, dpi=90); plt.close()
    print(f"\n  ROC/PR curves saved: {path}")


# -- 3. Multi-class metrics ---------------------------------------------------
def multiclass_metrics():
    print("\n=== Multi-class Classification Metrics ===")
    iris = load_iris()
    X, y = iris.data, iris.target
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)
    model = LogisticRegression(max_iter=500, multi_class="multinomial")
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)

    print(f"  {'Metric':<30} {'Macro':<12} {'Micro':<12} {'Weighted'}")
    for name, fn, kwargs in [
        ("Precision",   precision_score, {}),
        ("Recall",      recall_score,    {}),
        ("F1",          f1_score,        {}),
    ]:
        mac = fn(y_te, y_pred, average="macro", **kwargs, zero_division=0)
        mic = fn(y_te, y_pred, average="micro", **kwargs, zero_division=0)
        wgt = fn(y_te, y_pred, average="weighted", **kwargs, zero_division=0)
        print(f"  {name:<30} {mac:<12.4f} {mic:<12.4f} {wgt:.4f}")

    # Multi-class ROC-AUC (OvR)
    auc = roc_auc_score(y_te, y_prob, multi_class="ovr", average="macro")
    print(f"\n  ROC-AUC (OvR macro): {auc:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_te, y_pred, target_names=iris.target_names, indent=4))


# -- 4. F-beta score ----------------------------------------------------------
def fbeta_score():
    print("\n=== F-beta Score ===")
    print("  F_beta = (1+beta²) · P·R / (beta²·P + R)")
    print("  beta > 1: recall is more important (e.g. medical diagnosis)")
    print("  beta < 1: precision is more important (e.g. spam detection)")
    from sklearn.metrics import fbeta_score

    X, y = make_classification(n_samples=500, n_features=10, random_state=2,
                                weights=[0.85, 0.15])
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)
    model = LogisticRegression(max_iter=500, class_weight="balanced").fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    prec = precision_score(y_te, y_pred, zero_division=0)
    rec  = recall_score(y_te, y_pred, zero_division=0)
    print(f"  Precision={prec:.4f}  Recall={rec:.4f}")
    for beta in [0.25, 0.5, 1.0, 2.0, 4.0]:
        fb = fbeta_score(y_te, y_pred, beta=beta, zero_division=0)
        print(f"  F_{beta}: {fb:.4f}  (beta>1 emphasises recall)")


# -- 5. Threshold analysis ----------------------------------------------------
def threshold_analysis():
    print("\n=== Optimal Threshold Selection ===")
    X, y = make_classification(n_samples=500, n_features=10, random_state=3,
                                weights=[0.8, 0.2])
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)
    model = LogisticRegression(max_iter=500).fit(X_tr, y_tr)
    y_prob = model.predict_proba(X_te)[:,1]

    fpr, tpr, thr_roc = roc_curve(y_te, y_prob)
    prec, rec, thr_pr = precision_recall_curve(y_te, y_prob)

    # Youden's J: maximise TPR - FPR
    j_stat = tpr - fpr
    opt_thr_roc = thr_roc[j_stat.argmax()]
    # F1-optimal
    f1_list = 2*prec*rec / (prec+rec+1e-9)
    opt_thr_pr = thr_pr[f1_list[:-1].argmax()]

    print(f"  Youden's J optimal threshold: {opt_thr_roc:.4f}")
    print(f"  F1-optimal threshold:         {opt_thr_pr:.4f}")
    print(f"  Default threshold:            0.5")

    for name, thr in [("default (0.5)", 0.5),
                      (f"Youden's J ({opt_thr_roc:.2f})", opt_thr_roc),
                      (f"F1-optimal ({opt_thr_pr:.2f})", opt_thr_pr)]:
        yp = (y_prob >= thr).astype(int)
        print(f"\n  {name}:")
        print(f"    P={precision_score(y_te,yp,zero_division=0):.4f}  "
              f"R={recall_score(y_te,yp,zero_division=0):.4f}  "
              f"F1={f1_score(y_te,yp,zero_division=0):.4f}")


if __name__ == "__main__":
    binary_metrics()
    roc_pr_curves()
    multiclass_metrics()
    fbeta_score()
    threshold_analysis()
