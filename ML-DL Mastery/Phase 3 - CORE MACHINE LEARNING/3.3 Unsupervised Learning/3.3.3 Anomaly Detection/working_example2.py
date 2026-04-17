"""
Working Example 2: Anomaly Detection — Isolation Forest, One-Class SVM, LOF
============================================================================
Unsupervised anomaly detection on Cal Housing (price extremes as anomalies).

Run:  python working_example2.py
"""
from pathlib import Path
try:
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.datasets import fetch_california_housing
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.svm import OneClassSVM
    from sklearn.metrics import precision_score, recall_score
except ImportError:
    raise SystemExit("pip install numpy matplotlib scikit-learn")

OUTPUT = Path(__file__).parent / "output"
OUTPUT.mkdir(exist_ok=True)

def load_data():
    h = fetch_california_housing()
    X = StandardScaler().fit_transform(h.data)
    # Create pseudo ground-truth anomalies: top/bottom 5% prices
    y_true = np.zeros(len(h.target), dtype=int)
    threshold_high = np.percentile(h.target, 95)
    threshold_low  = np.percentile(h.target, 5)
    y_true[(h.target >= threshold_high) | (h.target <= threshold_low)] = 1
    print(f"  Total anomalies (top/bot 5%): {y_true.sum()} / {len(y_true)}")
    return X, y_true

def demo_isolation_forest(X, y_true):
    print("\n=== Isolation Forest ===")
    for contamination in [0.05, 0.1, 0.15]:
        iso = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
        pred = iso.fit_predict(X)
        anomalies = (pred == -1).astype(int)
        prec = precision_score(y_true, anomalies, zero_division=0)
        rec  = recall_score(y_true, anomalies, zero_division=0)
        print(f"  contamination={contamination}: precision={prec:.4f}  recall={rec:.4f}")

def demo_lof(X, y_true):
    print("\n=== Local Outlier Factor ===")
    for k in [10, 20, 50]:
        lof = LocalOutlierFactor(n_neighbors=k, contamination=0.1)
        pred = lof.fit_predict(X)
        anomalies = (pred == -1).astype(int)
        prec = precision_score(y_true, anomalies, zero_division=0)
        rec  = recall_score(y_true, anomalies, zero_division=0)
        print(f"  n_neighbors={k}: precision={prec:.4f}  recall={rec:.4f}")

def demo_one_class_svm(X, y_true):
    print("\n=== One-Class SVM ===")
    oc = OneClassSVM(nu=0.05, kernel="rbf", gamma="scale")
    pred = oc.fit_predict(X[:5000])   # subset for speed
    anomalies = (pred == -1).astype(int)
    prec = precision_score(y_true[:5000], anomalies, zero_division=0)
    rec  = recall_score(y_true[:5000], anomalies, zero_division=0)
    print(f"  OC-SVM (nu=0.05): precision={prec:.4f}  recall={rec:.4f}")

if __name__ == "__main__":
    X, y_true = load_data()
    demo_isolation_forest(X, y_true)
    demo_lof(X, y_true)
    demo_one_class_svm(X, y_true)
