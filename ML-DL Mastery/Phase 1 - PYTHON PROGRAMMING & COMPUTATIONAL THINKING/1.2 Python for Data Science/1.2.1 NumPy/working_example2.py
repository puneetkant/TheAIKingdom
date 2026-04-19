"""
Working Example 2: NumPy — Real-World Array Operations for ML
=============================================================
Downloads California Housing dataset from HuggingFace and demonstrates:
  - Array creation from CSV data, dtype control
  - Broadcasting and vectorized feature engineering
  - Linear algebra: matrix multiply, eigenvalues (PCA-lite)
  - Boolean indexing and fancy indexing for data filtering
  - Performance: NumPy vs pure-Python speed comparison

Run:  python working_example2.py
"""
import csv
import math
import time
import urllib.request
from pathlib import Path

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("NumPy not installed. Run: pip install numpy")
    raise SystemExit(1)

DATA = Path(__file__).parent / "data"
DATA.mkdir(exist_ok=True)


# -- 1. Download and load California Housing -----------------------------------
def download_cal_housing() -> Path:
    dest = DATA / "cal_housing.csv"
    if not dest.exists():
        try:
            urllib.request.urlretrieve(
                "https://huggingface.co/datasets/scikit-learn/california-housing/resolve/main/cal_housing.csv",
                dest
            )
            print(f"Downloaded {dest.name}")
        except Exception as e:
            print(f"Download failed ({e}); using synthetic data")
            np.random.seed(42)
            n = 500
            header = "MedInc,HouseAge,AveRooms,AveBedrms,Population,AveOccup,Latitude,Longitude,MedHouseVal"
            rows = [header]
            for _ in range(n):
                rows.append(",".join(str(round(v, 4)) for v in
                    [np.random.uniform(1,10), np.random.randint(1,50),
                     np.random.uniform(3,10), np.random.uniform(0.5,2),
                     np.random.randint(100,5000), np.random.uniform(1,5),
                     np.random.uniform(32,42), np.random.uniform(-124,-114),
                     np.random.uniform(0.5,5)]))
            dest.write_text("\n".join(rows))
    return dest


def load_to_numpy(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    cols = list(rows[0].keys())
    target_col = "MedHouseVal"
    feature_cols = [c for c in cols if c != target_col]

    X = np.array([[float(r[c]) for c in feature_cols] for r in rows], dtype=np.float64)
    y = np.array([float(r[target_col]) for r in rows], dtype=np.float64)
    return X, y, feature_cols


# -- 2. Vectorized feature engineering -----------------------------------------
def feature_engineering(X: np.ndarray, feature_cols: list[str]) -> tuple[np.ndarray, list[str]]:
    print("\n=== Vectorized Feature Engineering ===")
    # Indices by column name
    idx = {name: i for i, name in enumerate(feature_cols)}

    rooms     = X[:, idx["AveRooms"]]
    bedrms    = X[:, idx["AveBedrms"]]
    pop       = X[:, idx["Population"]]
    occ       = X[:, idx["AveOccup"]]
    income    = X[:, idx["MedInc"]]
    house_age = X[:, idx["HouseAge"]]

    # New features via broadcasting (no for-loops)
    bed_room_ratio  = bedrms / (rooms + 1e-8)            # bedroom/room ratio
    pop_per_room    = pop / (rooms * occ + 1e-8)         # density proxy
    log_income      = np.log1p(income)                   # log transform
    income_age_int  = income * house_age                 # interaction term

    new_features = np.column_stack([bed_room_ratio, pop_per_room, log_income, income_age_int])
    new_names    = ["bed_room_ratio", "pop_per_room", "log_income", "income_age_int"]

    print(f"  Original features : {X.shape[1]}")
    print(f"  New features added: {new_features.shape[1]}")
    print(f"  bed_room_ratio stats: min={bed_room_ratio.min():.3f}  max={bed_room_ratio.max():.3f}  mean={bed_room_ratio.mean():.3f}")
    print(f"  log_income     stats: min={log_income.min():.3f}  max={log_income.max():.3f}")
    return new_features, new_names


# -- 3. Normalisation (z-score) -------------------------------------------------
def standardize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return standardised X, mean, and std vectors."""
    mean = X.mean(axis=0)
    std  = X.std(axis=0) + 1e-8
    return (X - mean) / std, mean, std


# -- 4. Linear algebra: manual linear regression + PCA -------------------------
def demo_linear_algebra(X: np.ndarray, y: np.ndarray, feature_cols: list[str]) -> None:
    print("\n=== Linear Algebra ===")

    # Normal equations: w = (X^T X)^{-1} X^T y
    X_std, _, _ = standardize(X)
    X_bias = np.column_stack([np.ones(len(X_std)), X_std])  # add bias

    XtX = X_bias.T @ X_bias
    Xty = X_bias.T @ y

    try:
        w = np.linalg.solve(XtX, Xty)
        y_pred = X_bias @ w
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot
        rmse = math.sqrt(ss_res / len(y))
        print(f"  OLS R²   : {r2:.4f}")
        print(f"  OLS RMSE : {rmse:.4f}")
    except np.linalg.LinAlgError:
        print("  Singular matrix — skipping OLS")

    # Mini PCA: top 2 eigenvectors of covariance matrix
    cov = np.cov(X_std.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # eigh returns sorted ascending; take last 2
    top_idx = np.argsort(eigenvalues)[::-1][:2]
    explained = eigenvalues[top_idx] / eigenvalues.sum()
    print(f"\n  PCA top-2 explained variance: {explained[0]:.3f}, {explained[1]:.3f}  "
          f"(total={sum(explained):.3f})")


# -- 5. Boolean indexing + fancy indexing --------------------------------------
def demo_indexing(X: np.ndarray, y: np.ndarray) -> None:
    print("\n=== Indexing Patterns ===")

    # Boolean: expensive houses
    mask_expensive = y > np.percentile(y, 75)
    print(f"  Houses above 75th percentile: {mask_expensive.sum()} / {len(y)}")

    # Fancy: select random 10-sample batch
    np.random.seed(7)
    batch_idx = np.random.choice(len(X), size=10, replace=False)
    batch_X = X[batch_idx]
    batch_y = y[batch_idx]
    print(f"  Random batch mean income: {batch_X[:, 0].mean():.3f}")
    print(f"  Random batch mean value : {batch_y.mean():.3f}")

    # np.where
    risk_label = np.where(y > np.median(y), "high", "low")
    unique, counts = np.unique(risk_label, return_counts=True)
    print(f"  Risk labels: {dict(zip(unique, counts))}")


# -- 6. Performance comparison -------------------------------------------------
def demo_performance(X: np.ndarray) -> None:
    print("\n=== Performance: NumPy vs Pure Python ===")
    flat = X[:, 0].tolist()   # first column as plain list

    t0 = time.perf_counter()
    mean_py = sum(flat) / len(flat)
    t_py = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    mean_np = X[:, 0].mean()
    t_np = (time.perf_counter() - t0) * 1000

    print(f"  Pure Python mean: {mean_py:.4f}  ({t_py:.3f} ms)")
    print(f"  NumPy mean      : {mean_np:.4f}  ({t_np:.3f} ms)")
    if t_py > 0:
        print(f"  NumPy speedup   : {t_py/max(t_np,0.001):.1f}x")


if __name__ == "__main__":
    path = download_cal_housing()
    X, y, feature_cols = load_to_numpy(path)
    print(f"Loaded: X.shape={X.shape}, y.shape={y.shape}, dtype={X.dtype}")
    print(f"Features: {feature_cols}")

    feature_engineering(X, feature_cols)
    demo_linear_algebra(X, y, feature_cols)
    demo_indexing(X, y)
    demo_performance(X)
