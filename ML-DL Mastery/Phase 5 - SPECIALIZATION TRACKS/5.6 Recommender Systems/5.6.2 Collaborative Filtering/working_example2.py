"""
Working Example 2: Collaborative Filtering — matrix factorisation from scratch
================================================================================
Implements SVD-based and SGD matrix factorisation on synthetic rating matrix.

Run:  python working_example2.py
"""
from pathlib import Path
try:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit("pip install numpy matplotlib")

OUTPUT = Path(__file__).parent / "output"
OUTPUT.mkdir(exist_ok=True)

def make_rating_matrix(n_users=20, n_items=15, n_factors=3, seed=0):
    rng = np.random.default_rng(seed)
    U = rng.normal(0, 1, (n_users, n_factors))
    V = rng.normal(0, 1, (n_items, n_factors))
    full = U @ V.T + rng.normal(0, 0.2, (n_users, n_items))
    full = np.clip(full, 1, 5)
    # Mask 30% as unobserved
    mask = rng.random((n_users, n_items)) > 0.3
    R = np.where(mask, full, np.nan)
    return R, full

def sgd_mf(R, k=5, lr=0.01, reg=0.01, epochs=200):
    n, m = R.shape
    rng = np.random.default_rng(42)
    P = rng.normal(0, 0.1, (n, k))  # user factors
    Q = rng.normal(0, 0.1, (m, k))  # item factors
    obs = [(i, j) for i in range(n) for j in range(m) if not np.isnan(R[i, j])]
    losses = []
    for _ in range(epochs):
        np.random.shuffle(obs)
        total = 0
        for i, j in obs:
            err = R[i, j] - P[i] @ Q[j]
            P[i] += lr * (err * Q[j] - reg * P[i])
            Q[j] += lr * (err * P[i] - reg * Q[j])
            total += err**2
        losses.append(total / len(obs))
    return P, Q, losses

def demo():
    print("=== Collaborative Filtering: Matrix Factorisation ===")
    R, full = make_rating_matrix()
    observed = (~np.isnan(R)).sum()
    print(f"  Rating matrix: {R.shape} | Observed: {observed}/{R.size} ({100*observed/R.size:.0f}%)")

    P, Q, losses = sgd_mf(R, k=5)
    R_pred = P @ Q.T

    # RMSE on observed entries
    mask = ~np.isnan(R)
    rmse = np.sqrt(np.mean((R[mask] - R_pred[mask])**2))
    print(f"  Train RMSE: {rmse:.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    im0 = axes[0].imshow(np.where(~np.isnan(R), R, 0), aspect="auto", cmap="Blues")
    axes[0].set_title("Observed Ratings"); plt.colorbar(im0, ax=axes[0])
    im1 = axes[1].imshow(R_pred, aspect="auto", cmap="Blues")
    axes[1].set_title("Predicted Ratings"); plt.colorbar(im1, ax=axes[1])
    axes[2].plot(losses); axes[2].set_title("SGD Loss"); axes[2].set_xlabel("Epoch")
    plt.tight_layout(); plt.savefig(OUTPUT / "collaborative_filtering.png"); plt.close()
    print("  Saved collaborative_filtering.png")

if __name__ == "__main__":
    demo()
