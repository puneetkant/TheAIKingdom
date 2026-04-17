"""
Working Example 2: Support Vector Machines — SVC, SVR, Kernels, Cal Housing
=============================================================================
Linear SVC, RBF kernel, C vs margin tradeoff, SVR regression.

Run:  python working_example2.py
"""
from pathlib import Path
try:
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC, SVR, LinearSVC
    from sklearn.metrics import mean_squared_error, roc_auc_score, classification_report
    from sklearn.pipeline import make_pipeline
except ImportError:
    raise SystemExit("pip install numpy matplotlib scikit-learn")

OUTPUT = Path(__file__).parent / "output"
OUTPUT.mkdir(exist_ok=True)

def load_binary(n=5000):
    h = fetch_california_housing()
    X, y = h.data[:n], (h.target[:n] > np.median(h.target[:n])).astype(int)
    return train_test_split(X, y, test_size=0.2, random_state=42)

def demo_linear_svc(X_train, X_test, y_train, y_test):
    print("=== Linear SVC (Cal Housing binary) ===")
    for C in [0.01, 0.1, 1.0, 10.0]:
        pipe = make_pipeline(StandardScaler(), LinearSVC(C=C, max_iter=5000))
        pipe.fit(X_train, y_train)
        acc = pipe.score(X_test, y_test)
        print(f"  C={C:>5}: accuracy={acc:.4f}")

def demo_rbf_kernel(X_train, X_test, y_train, y_test):
    print("\n=== RBF Kernel SVC ===")
    # Use subset for speed
    pipe = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, gamma="scale", probability=True))
    pipe.fit(X_train, y_train)
    y_prob = pipe.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    acc = pipe.score(X_test, y_test)
    print(f"  RBF SVC: accuracy={acc:.4f}  AUC={auc:.4f}")

def demo_svr():
    print("\n=== SVR (Support Vector Regression) ===")
    h = fetch_california_housing()
    X, y = h.data[:3000], h.target[:3000]   # smaller for SVR speed
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    for kernel in ["linear", "rbf"]:
        pipe = make_pipeline(StandardScaler(), SVR(kernel=kernel, C=1.0, epsilon=0.1))
        pipe.fit(X_train, y_train)
        rmse = mean_squared_error(y_test, pipe.predict(X_test))**0.5
        print(f"  SVR({kernel}): RMSE={rmse:.4f}")

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_binary()
    demo_linear_svc(X_train, X_test, y_train, y_test)
    demo_rbf_kernel(X_train, X_test, y_train, y_test)
    demo_svr()
