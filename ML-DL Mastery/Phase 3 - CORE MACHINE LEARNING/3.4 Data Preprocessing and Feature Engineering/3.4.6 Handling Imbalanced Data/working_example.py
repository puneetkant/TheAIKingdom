"""
Working Example: Handling Imbalanced Data
Covers class imbalance effects, threshold tuning, resampling (oversampling,
undersampling, SMOTE), class weights, and evaluation metrics for imbalance.
"""
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, average_precision_score,
                              precision_recall_curve, f1_score,
                              balanced_accuracy_score)
from sklearn.preprocessing import StandardScaler
import os


# ── 1. The imbalance problem ──────────────────────────────────────────────────
def imbalance_problem():
    print("=== The Class Imbalance Problem ===")
    rng = np.random.default_rng(0)
    # 95% negative, 5% positive (fraud detection scenario)
    n   = 1000
    X   = rng.standard_normal((n, 5))
    y   = (rng.uniform(0, 1, n) < 0.05).astype(int)  # ~5% positive
    print(f"  Class distribution: negative={( y==0).sum()}  positive={(y==1).sum()}")
    print(f"  Imbalance ratio: {(y==0).sum()/(y==1).sum():.1f}:1")

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3,
                                               stratify=y, random_state=0)

    # Naive model: predict always negative
    y_naive = np.zeros(len(y_te), dtype=int)
    naive_acc = (y_naive == y_te).mean()
    print(f"\n  Naive (all-negative) accuracy: {naive_acc:.4f}  ← misleading!")
    print(f"  Recall for minority class:    {0:.4f}  ← model is useless")

    # Logistic Regression (default threshold)
    scaler = StandardScaler().fit(X_tr)
    lr = LogisticRegression(max_iter=500).fit(scaler.transform(X_tr), y_tr)
    y_pred = lr.predict(scaler.transform(X_te))
    print(f"\n  LR (default threshold=0.5):")
    print(f"    Accuracy: {(y_pred==y_te).mean():.4f}")
    print(f"    F1 (minority): {f1_score(y_te, y_pred, pos_label=1):.4f}")
    print(f"    Recall (minority): {(y_pred[y_te==1]==1).mean():.4f}")
    return X, y


# ── 2. Better evaluation metrics for imbalance ───────────────────────────────
def evaluation_metrics(X, y):
    print("\n=== Metrics for Imbalanced Data ===")
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3,
                                               stratify=y, random_state=0)
    scaler = StandardScaler().fit(X_tr)
    lr = LogisticRegression(max_iter=500, class_weight="balanced")
    lr.fit(scaler.transform(X_tr), y_tr)
    proba  = lr.predict_proba(scaler.transform(X_te))[:,1]
    y_pred = lr.predict(scaler.transform(X_te))

    print(f"  ROC-AUC:           {roc_auc_score(y_te, proba):.4f}")
    print(f"  PR-AUC (AP):       {average_precision_score(y_te, proba):.4f}")
    print(f"  F1 (minority):     {f1_score(y_te, y_pred, pos_label=1):.4f}")
    print(f"  Balanced accuracy: {balanced_accuracy_score(y_te, y_pred):.4f}")
    print(f"\n  Confusion matrix:")
    cm = confusion_matrix(y_te, y_pred)
    print(f"    TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"    FN={cm[1,0]}  TP={cm[1,1]}")
    print(f"\n  Precision: {cm[1,1]/(cm[0,1]+cm[1,1]+1e-9):.4f}  "
          f"Recall: {cm[1,1]/(cm[1,0]+cm[1,1]+1e-9):.4f}")


# ── 3. Threshold tuning ───────────────────────────────────────────────────────
def threshold_tuning(X, y):
    print("\n=== Threshold Tuning ===")
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3,
                                               stratify=y, random_state=0)
    scaler = StandardScaler().fit(X_tr)
    lr = LogisticRegression(max_iter=500, class_weight="balanced")
    lr.fit(scaler.transform(X_tr), y_tr)
    proba = lr.predict_proba(scaler.transform(X_te))[:,1]

    print(f"  {'Threshold':<12} {'Precision':<12} {'Recall':<10} {'F1'}")
    for thresh in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        y_pred = (proba >= thresh).astype(int)
        prec   = f1_score(y_te, y_pred, pos_label=1, average=None, zero_division=0)[1] if y_pred.sum() else 0
        rec    = (y_pred[y_te==1]).mean() if (y_te==1).sum() else 0
        f1     = f1_score(y_te, y_pred, pos_label=1, zero_division=0)
        print(f"  {thresh:<12} {prec:<12.4f} {rec:<10.4f} {f1:.4f}")


# ── 4. Class weights ─────────────────────────────────────────────────────────
def class_weights_demo(X, y):
    print("\n=== Class Weights ===")
    print("  Penalise misclassification of minority class more heavily")
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3,
                                               stratify=y, random_state=0)
    scaler = StandardScaler().fit(X_tr)
    X_tr_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_te)

    print(f"  {'class_weight':<20} {'F1-minority':<14} {'Recall-minority'}")
    for cw in [None, "balanced", {0:1, 1:5}, {0:1, 1:10}, {0:1, 1:20}]:
        lr = LogisticRegression(max_iter=500, class_weight=cw).fit(X_tr_s, y_tr)
        yp = lr.predict(X_te_s)
        f1m  = f1_score(y_te, yp, pos_label=1, zero_division=0)
        recm = (yp[y_te==1]==1).mean() if (y_te==1).sum() else 0
        print(f"  {str(cw):<20} {f1m:<14.4f} {recm:.4f}")


# ── 5. Random oversampling and undersampling ─────────────────────────────────
def overunder_sampling(X, y):
    print("\n=== Random Over/Under Sampling ===")
    rng = np.random.default_rng(1)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3,
                                               stratify=y, random_state=0)
    scaler = StandardScaler().fit(X_tr)
    X_tr_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_te)

    def oversample(X, y, ratio=1.0, seed=0):
        rng2  = np.random.default_rng(seed)
        X_min = X[y==1]; X_maj = X[y==0]
        n_dup = int(len(X_maj)*ratio) - len(X_min)
        if n_dup > 0:
            idx   = rng2.choice(len(X_min), n_dup, replace=True)
            X_new = np.vstack([X, X_min[idx]])
            y_new = np.concatenate([y, np.ones(n_dup, int)])
            return X_new, y_new
        return X, y

    def undersample(X, y, ratio=1.0, seed=0):
        rng2  = np.random.default_rng(seed)
        X_min = X[y==1]; X_maj = X[y==0]
        n_keep = int(len(X_min)/ratio)
        idx   = rng2.choice(len(X_maj), min(n_keep, len(X_maj)), replace=False)
        X_new = np.vstack([X_min, X_maj[idx]])
        y_new = np.concatenate([np.ones(len(X_min),int), np.zeros(len(idx),int)])
        return X_new, y_new

    methods = [
        ("No resampling",    X_tr_s, y_tr),
        ("Oversample",       *oversample(X_tr_s, y_tr, ratio=1.0)),
        ("Undersample",      *undersample(X_tr_s, y_tr, ratio=1.0)),
    ]
    print(f"  {'Method':<20} {'Train size':<13} {'F1-minority'}")
    for name, Xr, yr in methods:
        lr   = LogisticRegression(max_iter=500).fit(Xr, yr)
        yp   = lr.predict(X_te_s)
        f1m  = f1_score(y_te, yp, pos_label=1, zero_division=0)
        print(f"  {name:<20} {len(yr):<13} {f1m:.4f}")


# ── 6. SMOTE (conceptual + imblearn if available) ────────────────────────────
def smote_demo(X, y):
    print("\n=== SMOTE (Synthetic Minority Over-sampling) ===")
    print("  Generates synthetic minority samples by interpolating between neighbours")
    print("  New sample: x_new = x_i + λ·(x_j - x_i)  where x_j is a k-NN of x_i")
    print()
    try:
        from imblearn.over_sampling import SMOTE, ADASYN
        from imblearn.combine import SMOTETomek

        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3,
                                                   stratify=y, random_state=0)
        scaler = StandardScaler().fit(X_tr)
        X_tr_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_te)

        print(f"  {'Method':<20} {'Train size':<13} {'F1-minority'}")
        for name, sampler in [("No sampling",  None),
                               ("SMOTE",        SMOTE(random_state=0)),
                               ("ADASYN",       ADASYN(random_state=0)),
                               ("SMOTETomek",   SMOTETomek(random_state=0))]:
            if sampler is None:
                Xr, yr = X_tr_s, y_tr
            else:
                Xr, yr = sampler.fit_resample(X_tr_s, y_tr)
            lr  = LogisticRegression(max_iter=500).fit(Xr, yr)
            yp  = lr.predict(X_te_s)
            f1m = f1_score(y_te, yp, pos_label=1, zero_division=0)
            print(f"  {name:<20} {len(yr):<13} {f1m:.4f}")
    except ImportError:
        print("  imblearn not installed. Install: pip install imbalanced-learn")
        print("  SMOTE concept: for each minority sample x_i:")
        print("    1. Find k nearest minority neighbours")
        print("    2. Pick random neighbour x_j")
        print("    3. Create x_new = x_i + λ(x_j - x_i) where λ∈[0,1]")


# ── 7. Algorithm selection for imbalance ─────────────────────────────────────
def algorithm_comparison(X, y):
    print("\n=== Algorithm Comparison for Imbalanced Data ===")
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3,
                                               stratify=y, random_state=0)
    scaler = StandardScaler().fit(X_tr)
    X_tr_s, X_te_s = scaler.transform(X_tr), scaler.transform(X_te)

    models = [
        ("LR (default)",          LogisticRegression(max_iter=500)),
        ("LR (balanced)",         LogisticRegression(max_iter=500, class_weight="balanced")),
        ("RF (default)",          RandomForestClassifier(n_estimators=50, random_state=0, n_jobs=-1)),
        ("RF (balanced)",         RandomForestClassifier(n_estimators=50, class_weight="balanced_subsample", random_state=0, n_jobs=-1)),
        ("GBM",                   GradientBoostingClassifier(n_estimators=50, random_state=0)),
    ]

    print(f"  {'Model':<28} {'ROC-AUC':<12} {'F1-minor':<12} {'Recall-minor'}")
    for name, m in models:
        m.fit(X_tr_s, y_tr)
        proba = m.predict_proba(X_te_s)[:,1] if hasattr(m,"predict_proba") else None
        yp    = m.predict(X_te_s)
        auc   = roc_auc_score(y_te, proba) if proba is not None else 0
        f1m   = f1_score(y_te, yp, pos_label=1, zero_division=0)
        recm  = (yp[y_te==1]==1).mean() if (y_te==1).sum() else 0
        print(f"  {name:<28} {auc:<12.4f} {f1m:<12.4f} {recm:.4f}")


if __name__ == "__main__":
    X, y = imbalance_problem()
    evaluation_metrics(X, y)
    threshold_tuning(X, y)
    class_weights_demo(X, y)
    overunder_sampling(X, y)
    smote_demo(X, y)
    algorithm_comparison(X, y)
