"""
Working Example 2: Systems of Linear Equations — Gaussian elimination + ML
===========================================================================
Demonstrates solving Ax=b using:
  - np.linalg.solve (LU-backed)
  - np.linalg.lstsq (least-squares for overdetermined systems)
  - Manual Gaussian elimination (educational)
  - Condition number and numerical stability
  - Application: multi-output regression on Cal Housing

Run:  python working_example2.py
"""
import csv, urllib.request
from pathlib import Path
try:
    import numpy as np
except ImportError:
    raise SystemExit("pip install numpy")

DATA = Path(__file__).parent / "data"
DATA.mkdir(exist_ok=True)

def download() -> tuple:
    dest = DATA / "cal_housing.csv"
    if not dest.exists():
        try: urllib.request.urlretrieve(
            "https://huggingface.co/datasets/scikit-learn/california-housing/resolve/main/cal_housing.csv", dest)
        except Exception:
            import random; random.seed(1)
            rows = ["MedInc,HouseAge,AveRooms,MedHouseVal"]
            for i in range(100):
                rows.append(f"{round(random.uniform(1,10),3)},{random.randint(1,52)},{round(random.uniform(3,8),3)},{round(random.uniform(0.5,5),3)}")
            dest.write_text("\n".join(rows))
    with open(dest) as f: rows = list(csv.DictReader(f))
    feat = ["MedInc","HouseAge","AveRooms"]
    X = np.array([[float(r[c]) for c in feat] for r in rows[:150]])
    y = np.array([float(r["MedHouseVal"]) for r in rows[:150]])
    return X, y

def demo_solve():
    print("=== np.linalg.solve (square system) ===")
    # 3×3 example: price model from 3 equations
    A = np.array([[1., 2., 1.], [2., 3., 1.], [1., 1., 2.]])
    b = np.array([14., 20., 12.])
    x = np.linalg.solve(A, b)
    print(f"  x = {x}")
    print(f"  Ax - b = {(A @ x - b).round(10)}")  # should be ~0

def demo_lstsq(X, y):
    print("\n=== np.linalg.lstsq (overdetermined) ===")
    Xb = np.hstack([np.ones((len(X),1)), X])
    w, residuals, rank, sv = np.linalg.lstsq(Xb, y, rcond=None)
    print(f"  Weights: {w.round(4)}")
    print(f"  Rank: {rank}  (full={Xb.shape[1]})")
    print(f"  Singular values: {sv.round(3)}")
    yh = Xb @ w
    r2 = 1 - np.sum((y-yh)**2)/np.sum((y-y.mean())**2)
    print(f"  R² = {r2:.4f}")

def gaussian_elimination(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Educational forward elimination + back substitution."""
    n = len(b)
    Ab = np.hstack([A.astype(float), b.reshape(-1,1)])
    for col in range(n):
        # Partial pivoting
        pivot_row = col + np.argmax(np.abs(Ab[col:, col]))
        Ab[[col, pivot_row]] = Ab[[pivot_row, col]]
        for row in range(col+1, n):
            if Ab[col, col] == 0: continue
            factor = Ab[row, col] / Ab[col, col]
            Ab[row] -= factor * Ab[col]
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (Ab[i, -1] - Ab[i, i+1:n] @ x[i+1:n]) / Ab[i, i]
    return x

def demo_gaussian():
    print("\n=== Manual Gaussian Elimination ===")
    A = np.array([[2., 1., -1.], [-3., -1., 2.], [-2., 1., 2.]])
    b = np.array([8., -11., -3.])
    x_manual = gaussian_elimination(A, b)
    x_numpy  = np.linalg.solve(A, b)
    print(f"  Manual: {x_manual.round(6)}")
    print(f"  NumPy:  {x_numpy.round(6)}")
    print(f"  Match: {np.allclose(x_manual, x_numpy)}")

def demo_condition_number():
    print("\n=== Condition Number & Stability ===")
    for scale in [1, 1e6, 1e12]:
        A = np.array([[1., scale], [1., scale + 1.]])
        cond = np.linalg.cond(A)
        print(f"  scale={scale:.0e}  cond(A)={cond:.3e}  {'ILL-CONDITIONED' if cond > 1e8 else 'ok'}")

if __name__ == "__main__":
    X, y = download()
    demo_solve()
    demo_lstsq(X, y)
    demo_gaussian()
    demo_condition_number()
