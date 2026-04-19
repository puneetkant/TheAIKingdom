"""
Working Example: Anomaly Detection
Covers statistical methods (Z-score, IQR), Isolation Forest, Local Outlier Factor,
One-Class SVM, Elliptic Envelope, DBSCAN-based, and evaluation with known labels.
"""
import numpy as np
from scipy import stats
from sklearn.datasets import make_classification, make_blobs
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.metrics import (precision_score, recall_score, f1_score,
                              roc_auc_score, average_precision_score)
from sklearn.preprocessing import StandardScaler
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_anomaly")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -- 1. Statistical methods ----------------------------------------------------
def statistical_anomaly():
    print("=== Statistical Anomaly Detection ===")
    rng = np.random.default_rng(0)
    n   = 200
    x   = rng.normal(50, 10, n)
    # Inject outliers
    x_out = np.concatenate([x, [-10, 110, 200, -50]])
    true_anom = np.array([0]*n + [1]*4)

    # Z-score method
    z     = np.abs(stats.zscore(x_out))
    z_pred = (z > 3).astype(int)
    print(f"  Z-score (|z|>3): detected {z_pred.sum()} anomalies")
    print(f"  Anomaly values:  {x_out[z_pred==1].round(1)}")

    # IQR method
    Q1, Q3 = np.percentile(x_out, 25), np.percentile(x_out, 75)
    IQR     = Q3 - Q1
    lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
    iqr_pred = ((x_out < lower) | (x_out > upper)).astype(int)
    print(f"\n  IQR (±1.5×IQR): detected {iqr_pred.sum()} anomalies")
    print(f"  Bounds: [{lower:.1f}, {upper:.1f}]")
    print(f"  Anomaly values: {x_out[iqr_pred==1].round(1)}")

    # Modified Z-score (MAD-based, robust)
    med   = np.median(x_out)
    mad   = np.median(np.abs(x_out - med))
    mz    = 0.6745 * np.abs(x_out - med) / (mad + 1e-9)
    mz_pred = (mz > 3.5).astype(int)
    print(f"\n  Modified Z-score: detected {mz_pred.sum()} anomalies")


# -- 2. Isolation Forest -------------------------------------------------------
def isolation_forest():
    print("\n=== Isolation Forest ===")
    print("  Anomalies are isolated faster -> shorter average path length in random trees")
    rng = np.random.default_rng(1)
    # Inliers: blob, outliers: random
    X_in  = rng.multivariate_normal([0,0], [[1,0.5],[0.5,1]], 200)
    X_out = rng.uniform(-6, 6, (20, 2))
    X     = np.vstack([X_in, X_out])
    y_true = np.array([0]*200 + [1]*20)   # 1=anomaly

    clf = IsolationForest(contamination=0.1, random_state=0, n_jobs=-1)
    clf.fit(X)
    # sklearn: -1=outlier, 1=inlier
    y_pred = (clf.predict(X) == -1).astype(int)
    scores = -clf.score_samples(X)   # higher = more anomalous

    pr   = precision_score(y_true, y_pred)
    rec  = recall_score(y_true, y_pred)
    auc  = roc_auc_score(y_true, scores)
    print(f"  Precision: {pr:.4f}  Recall: {rec:.4f}  ROC-AUC: {auc:.4f}")
    print(f"  Detected {y_pred.sum()} anomalies (true: 20)")

    # Contamination effect
    print(f"\n  {'contamination':<16} {'Precision':<12} {'Recall':<10} {'F1'}")
    for cont in [0.05, 0.10, 0.15, 0.20]:
        clf = IsolationForest(contamination=cont, random_state=0, n_jobs=-1).fit(X)
        yp  = (clf.predict(X) == -1).astype(int)
        print(f"  {cont:<16} {precision_score(y_true,yp):<12.4f} "
              f"{recall_score(y_true,yp):<10.4f} {f1_score(y_true,yp):.4f}")


# -- 3. Local Outlier Factor ---------------------------------------------------
def lof_demo():
    print("\n=== Local Outlier Factor (LOF) ===")
    print("  Compares local density of point to its k-neighbours' densities")
    print("  LOF >> 1 -> lower density than neighbours -> anomaly")

    rng = np.random.default_rng(2)
    X_in  = rng.multivariate_normal([0,0], [[1,0],[0,1]], 200)
    X_out = rng.uniform(-5, 5, (15, 2))
    X     = np.vstack([X_in, X_out])
    y_true = np.array([0]*200 + [1]*15)

    print(f"\n  {'n_neighbors':<14} {'Precision':<12} {'Recall':<10} {'F1'}")
    for k in [5, 10, 20, 40]:
        lof  = LocalOutlierFactor(n_neighbors=k, contamination=0.07)
        yp   = (lof.fit_predict(X) == -1).astype(int)
        print(f"  {k:<14} {precision_score(y_true,yp):<12.4f} "
              f"{recall_score(y_true,yp):<10.4f} {f1_score(y_true,yp):.4f}")


# -- 4. One-Class SVM ---------------------------------------------------------
def one_class_svm():
    print("\n=== One-Class SVM ===")
    print("  Trains a boundary around inliers in feature space (kernel-induced)")

    rng = np.random.default_rng(3)
    X_tr  = rng.multivariate_normal([0,0], [[1,0.3],[0.3,1]], 300)
    X_te_in  = rng.multivariate_normal([0,0], [[1,0.3],[0.3,1]], 100)
    X_te_out = rng.uniform(-5, 5, (20, 2))
    X_te  = np.vstack([X_te_in, X_te_out])
    y_te  = np.array([0]*100 + [1]*20)

    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr)
    X_te_s = scaler.transform(X_te)

    print(f"\n  {'nu':<8} {'kernel':<10} {'Precision':<12} {'Recall'}")
    for nu in [0.05, 0.1, 0.2]:
        for kernel in ["rbf", "linear"]:
            oc = OneClassSVM(nu=nu, kernel=kernel).fit(X_tr_s)
            yp = (oc.predict(X_te_s) == -1).astype(int)
            print(f"  {nu:<8} {kernel:<10} {precision_score(y_te,yp,zero_division=0):<12.4f} "
                  f"{recall_score(y_te,yp,zero_division=0):.4f}")


# -- 5. Elliptic Envelope -----------------------------------------------------
def elliptic_envelope():
    print("\n=== Elliptic Envelope (Robust Covariance) ===")
    print("  Fits a robust Gaussian to inliers; flags points far from the centre")

    rng = np.random.default_rng(4)
    X_in  = rng.multivariate_normal([0,0], [[2,1],[1,1]], 200)
    X_out = rng.uniform(-8, 8, (20, 2))
    X     = np.vstack([X_in, X_out])
    y_true = np.array([0]*200 + [1]*20)

    for cont in [0.05, 0.10, 0.15]:
        ee  = EllipticEnvelope(contamination=cont, random_state=0).fit(X)
        yp  = (ee.predict(X) == -1).astype(int)
        print(f"  contamination={cont}: Precision={precision_score(y_true,yp):.4f}  "
              f"Recall={recall_score(y_true,yp):.4f}")


# -- 6. Algorithm comparison ---------------------------------------------------
def anomaly_comparison():
    print("\n=== Anomaly Detection Algorithm Comparison ===")
    rng = np.random.default_rng(5)
    X_in  = rng.multivariate_normal([0,0], [[1,0.5],[0.5,1]], 300)
    X_out = rng.uniform(-5, 5, (30, 2))
    X     = np.vstack([X_in, X_out])
    y     = np.array([0]*300 + [1]*30)

    scaler = StandardScaler().fit(X)
    X_s    = scaler.transform(X)
    cont   = 0.09

    detectors = [
        ("Isolation Forest",   IsolationForest(contamination=cont, random_state=0, n_jobs=-1)),
        ("LOF",                LocalOutlierFactor(contamination=cont)),
        ("One-Class SVM",      OneClassSVM(nu=cont)),
        ("Elliptic Envelope",  EllipticEnvelope(contamination=cont, random_state=0)),
    ]

    print(f"  {'Detector':<22} {'Precision':<12} {'Recall':<10} {'F1':<10} {'n_detected'}")
    for name, det in detectors:
        yp = (det.fit_predict(X_s) == -1).astype(int)
        print(f"  {name:<22} {precision_score(y,yp,zero_division=0):<12.4f} "
              f"{recall_score(y,yp,zero_division=0):<10.4f} "
              f"{f1_score(y,yp,zero_division=0):<10.4f} {yp.sum()}")


if __name__ == "__main__":
    statistical_anomaly()
    isolation_forest()
    lof_demo()
    one_class_svm()
    elliptic_envelope()
    anomaly_comparison()
