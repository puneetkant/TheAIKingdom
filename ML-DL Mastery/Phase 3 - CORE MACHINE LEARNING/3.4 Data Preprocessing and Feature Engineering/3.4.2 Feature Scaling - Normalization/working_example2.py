"""
Working Example 2: Feature Scaling — StandardScaler, MinMax, RobustScaler
==========================================================================
Effect of scaling on KNN, SVM, Ridge regression performance.

Run:  python working_example2.py
"""
from pathlib import Path
try:
    import numpy as np
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer
    from sklearn.datasets import load_iris
    from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
    from sklearn.svm import LinearSVR
    from sklearn.linear_model import Ridge, LogisticRegression
    from sklearn.metrics import mean_squared_error
    from sklearn.pipeline import make_pipeline
except ImportError:
    raise SystemExit("pip install numpy scikit-learn")

def demo_scaling_comparison():
    print("=== Scaling Comparison (Cal Housing) ===")
    h = fetch_california_housing()
    X, y = h.data, h.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scalers = {
        "None":     None,
        "Standard": StandardScaler(),
        "MinMax":   MinMaxScaler(),
        "Robust":   RobustScaler(),
    }
    models = {
        "Ridge":   Ridge(1.0),
        "KNN(10)": KNeighborsRegressor(10),
    }

    for model_name, model in models.items():
        print(f"\n  Model: {model_name}")
        for scaler_name, scaler in scalers.items():
            if scaler is None:
                import sklearn; from sklearn.base import clone
                m = sklearn.base.clone(model)
                m.fit(X_train, y_train)
                rmse = mean_squared_error(y_test, m.predict(X_test))**0.5
            else:
                pipe = make_pipeline(scaler, model)
                pipe.fit(X_train, y_train)
                rmse = mean_squared_error(y_test, pipe.predict(X_test))**0.5
            print(f"    {scaler_name:10s}: RMSE={rmse:.4f}")

def demo_feature_range():
    print("\n=== Feature Value Ranges (Cal Housing) ===")
    h = fetch_california_housing()
    X = h.data
    print(f"  {'Feature':20s}  {'min':>10}  {'max':>10}  {'std':>10}")
    for name, col in zip(h.feature_names, X.T):
        print(f"  {name:20s}  {col.min():10.2f}  {col.max():10.2f}  {col.std():10.2f}")

def demo_standard_scaler():
    print("\n=== Standard Scaler (Cal Housing): mean/std before and after ===")
    h = fetch_california_housing()
    X = h.data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"  {'Feature':20s}  {'mean_before':>12}  {'std_before':>10}  "
          f"{'mean_after':>10}  {'std_after':>9}")
    for name, raw, sc in zip(h.feature_names, X.T, X_scaled.T):
        print(f"  {name:20s}  {raw.mean():12.4f}  {raw.std():10.4f}  "
              f"{sc.mean():10.4f}  {sc.std():9.4f}")


def demo_robust_scaler():
    print("\n=== Robust vs Standard Scaler with 5% Outliers (Iris + KNN) ===")
    iris = load_iris()
    rng = np.random.default_rng(42)
    X, y = iris.data.copy(), iris.target
    n_out = max(1, int(0.05 * len(X)))
    idx = rng.choice(len(X), n_out, replace=False)
    X[idx] += rng.uniform(10, 20, (n_out, X.shape[1]))
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)
    for name, scaler in [("StandardScaler", StandardScaler()),
                          ("RobustScaler",   RobustScaler())]:
        pipe = make_pipeline(scaler, KNeighborsClassifier(n_neighbors=5))
        pipe.fit(X_tr, y_tr)
        print(f"  {name:16s}: accuracy={pipe.score(X_te, y_te):.4f}")


def demo_minmax_scaler():
    print("\n=== MinMax Scaler: range check and classifier accuracy (Iris) ===")
    iris = load_iris()
    X, y = iris.data, iris.target
    mm = MinMaxScaler()
    X_scaled = mm.fit_transform(X)
    print(f"  {'Feature':25s}  {'min_before':>10}  {'max_before':>10}  "
          f"{'min_after':>9}  {'max_after':>9}")
    for name, raw, sc in zip(iris.feature_names, X.T, X_scaled.T):
        print(f"  {name:25s}  {raw.min():10.4f}  {raw.max():10.4f}  "
              f"{sc.min():9.4f}  {sc.max():9.4f}")
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler_fit = MinMaxScaler().fit(X_tr)
    for label, Xtr, Xte in [
        ("No scaling", X_tr,                       X_te),
        ("MinMax",     scaler_fit.transform(X_tr), scaler_fit.transform(X_te)),
    ]:
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(Xtr, y_tr)
        print(f"  {label:12s}: accuracy={clf.score(Xte, y_te):.4f}")


def demo_normalizer():
    print("\n=== L2 Row Normalizer (Cal Housing): cosine similarities preserved ===")
    h = fetch_california_housing()
    X = h.data[:10]
    X_norm = Normalizer(norm="l2").fit_transform(X)
    norm0 = np.linalg.norm(X[0]); norm1 = np.linalg.norm(X[1])
    cos_raw   = float(X[0] @ X[1]) / (norm0 * norm1)
    cos_normd = float(X_norm[0] @ X_norm[1])
    print(f"  Cosine sim rows 0,1 (raw vectors):    {cos_raw:.6f}")
    print(f"  Dot product of L2-normed rows 0,1:   {cos_normd:.6f}")
    print(f"  Absolute difference:                  {abs(cos_raw - cos_normd):.2e}")
    norms = np.linalg.norm(X_norm, axis=1)
    print(f"  Row L2 norms after normalizing: min={norms.min():.6f}  max={norms.max():.6f}")


def demo_scaling_impact():
    print("\n=== Scaling Impact on LogisticRegression (Iris, feature 0 * 1000) ===")
    iris = load_iris()
    X, y = iris.data.copy(), iris.target
    X[:, 0] *= 1000
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)
    clf_raw = LogisticRegression(max_iter=5000, random_state=42)
    clf_raw.fit(X_tr, y_tr)
    print(f"  {'No scaling':16s}: accuracy={clf_raw.score(X_te, y_te):.4f}")
    sc = StandardScaler().fit(X_tr)
    clf_sc = LogisticRegression(max_iter=1000, random_state=42)
    clf_sc.fit(sc.transform(X_tr), y_tr)
    print(f"  {'StandardScaler':16s}: accuracy={clf_sc.score(sc.transform(X_te), y_te):.4f}")


if __name__ == "__main__":
    demo_feature_range()
    demo_scaling_comparison()
    demo_standard_scaler()
    demo_robust_scaler()
    demo_minmax_scaler()
    demo_normalizer()
    demo_scaling_impact()
