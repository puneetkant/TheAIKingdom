"""
Working Example: Collaborative Filtering
Covers user-user CF, item-item CF, matrix factorisation (SVD / ALS / SGD),
and implicit feedback models.
"""
import numpy as np
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_collab")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -- 1. User-item rating matrix ------------------------------------------------
def build_rating_matrix():
    """Returns (R, user_names, item_names) where NaN = unrated."""
    users = ["Alice", "Bob", "Carol", "David", "Eve"]
    items = ["Inception", "Interstellar", "Dark Knight", "Toy Story", "Finding Nemo", "Avatar"]
    raw = [
        [5, 4, np.nan, 1, np.nan, 3],
        [4, np.nan, 5, np.nan, 2, 4],
        [np.nan, 5, 4, np.nan, np.nan, 5],
        [1, 2, np.nan, 5, 5, np.nan],
        [np.nan, 3, 4, 4, np.nan, 3],
    ]
    return np.array(raw), users, items


# -- 2. Memory-based CF --------------------------------------------------------
def memory_based_cf():
    print("=== Memory-Based Collaborative Filtering ===")
    R, users, items = build_rating_matrix()
    U, I = R.shape
    print(f"  Rating matrix: {U} users × {I} items  "
          f"(sparsity={np.isnan(R).mean():.0%})")

    # User-user CF: cosine similarity on mean-centred ratings
    def user_similarity(R):
        mu     = np.nanmean(R, axis=1, keepdims=True)
        Rc     = np.where(np.isnan(R), 0, R - mu)
        norms  = np.sqrt((Rc**2).sum(axis=1, keepdims=True))
        Rn     = Rc / (norms + 1e-9)
        return Rn @ Rn.T

    sim_u = user_similarity(R)
    print()
    print("  User-user cosine similarity:")
    print(f"  {'':>8} " + "  ".join(f"{u[:5]:>6}" for u in users))
    for i, u in enumerate(users):
        row = "  ".join(f"{sim_u[i,j]:>6.3f}" for j in range(U))
        print(f"  {u:<8} {row}")

    # Predict for Alice × "Interstellar" slot (index [0,1]) — already rated, for verification
    target_u, target_i = 0, 2   # Alice, Dark Knight (unrated)
    k = 2
    nbrs = np.argsort(sim_u[target_u])[::-1]
    rated_nbrs = [n for n in nbrs if n != target_u and not np.isnan(R[n, target_i])][:k]
    mu_u = np.nanmean(R, axis=1)
    num  = sum(sim_u[target_u, n] * (R[n, target_i] - mu_u[n]) for n in rated_nbrs)
    den  = sum(abs(sim_u[target_u, n]) for n in rated_nbrs) + 1e-9
    pred = mu_u[target_u] + num/den
    print()
    print(f"  User-based prediction for {users[target_u]} × '{items[target_i]}':")
    print(f"    Neighbours: {[users[n] for n in rated_nbrs]}")
    print(f"    Predicted rating: {pred:.3f}")

    # Item-item CF
    sim_i = user_similarity(R.T)   # swap axes
    print()
    print("  Item-item CF: most similar to 'Inception':")
    sims = sim_i[0].copy(); sims[0] = -1
    top3 = sims.argsort()[::-1][:3]
    for idx in top3:
        print(f"    {items[idx]:<20} sim={sim_i[0,idx]:.3f}")


# -- 3. Matrix factorisation (SVD) ---------------------------------------------
def matrix_factorisation():
    print("\n=== Matrix Factorisation ===")
    R, users, items = build_rating_matrix()
    U, I  = R.shape
    K     = 2    # latent factors

    # Fill NaN with global mean for SVD
    mu     = np.nanmean(R)
    R_fill = np.where(np.isnan(R), mu, R)

    # Compact SVD
    Uf, Sf, Vft = np.linalg.svd(R_fill, full_matrices=False)
    # Truncate to K factors
    P = Uf[:, :K] * Sf[:K]  # (U, K) user factors
    Q = Vft[:K, :]           # (K, I) item factors

    R_hat = P @ Q
    print(f"  SVD rank-{K} approximation:  P={P.shape}, Q={Q.shape}")

    # RMSE on observed
    mask  = ~np.isnan(R)
    rmse  = np.sqrt(((R_hat[mask] - R[mask])**2).mean())
    print(f"  RMSE on observed: {rmse:.4f}")
    print()
    print("  Predicted rating matrix (rounded):")
    print(f"  {'':>8} " + "  ".join(f"{it[:8]:>8}" for it in items))
    for i, u in enumerate(users):
        row = "  ".join(f"{R_hat[i,j]:>8.2f}" for j in range(I))
        print(f"  {u:<8} {row}")


# -- 4. SGD matrix factorisation -----------------------------------------------
def sgd_mf():
    print("\n=== SGD Matrix Factorisation (Simon Funk style) ===")
    R, users, items = build_rating_matrix()
    U, I = R.shape; K = 3

    rng = np.random.default_rng(42)
    P   = rng.normal(0, 0.1, (U, K))    # user factors
    Q   = rng.normal(0, 0.1, (I, K))    # item factors
    b_u = np.zeros(U)                    # user biases
    b_i = np.zeros(I)                    # item biases
    mu  = np.nanmean(R)

    lr = 0.01; reg = 0.02
    observed = [(u, i, R[u, i]) for u in range(U) for i in range(I)
                if not np.isnan(R[u, i])]

    losses = []
    for epoch in range(200):
        rng.shuffle(observed := list(observed))
        total_loss = 0
        for u, i, r in observed:
            pred = mu + b_u[u] + b_i[i] + P[u] @ Q[i]
            e    = r - pred
            b_u[u] += lr * (e - reg * b_u[u])
            b_i[i] += lr * (e - reg * b_i[i])
            P[u]   += lr * (e * Q[i] - reg * P[u])
            Q[i]   += lr * (e * P[u] - reg * Q[i])
            total_loss += e**2
        rmse = np.sqrt(total_loss / len(observed))
        losses.append(rmse)

    print(f"  SGD MF (K={K}, {200} epochs):  Final RMSE={losses[-1]:.4f}")

    # Show a predicted rating
    u_idx, i_idx = 0, 2  # Alice, Dark Knight
    pred = mu + b_u[u_idx] + b_i[i_idx] + P[u_idx] @ Q[i_idx]
    pred = np.clip(pred, 1, 5)
    print(f"  Predicted {users[u_idx]} × '{items[i_idx]}': {pred:.3f}")

    # Plot convergence
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(losses)
    ax.set_xlabel("Epoch"); ax.set_ylabel("RMSE"); ax.set_title("SGD MF Training")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "sgd_mf_loss.png")
    plt.savefig(path, dpi=90); plt.close()
    print(f"  Plot: {path}")


# -- 5. Implicit feedback ------------------------------------------------------
def implicit_feedback():
    print("\n=== Implicit Feedback CF ===")
    print()
    print("  Explicit: star ratings, thumbs up/down")
    print("  Implicit: clicks, purchases, views, play counts")
    print()
    print("  iALS (Implicit Alternating Least Squares — Hu et al. 2008):")
    print("    r_{ui} in {0,1} (binary preference)")
    print("    c_{ui} = 1 + alpha·click_count  (confidence weight)")
    print("    Minimise: Sigma_{u,i} c_{ui}(r_{ui} - p_u^T q_i)² + lambda(||P||² + ||Q||²)")
    print()
    print("  BPR (Bayesian Personalised Ranking — Rendle et al. 2009):")
    print("    Maximise: Sigma_{(u,i,j)} ln sigma(x_{ui} - x_{uj})  + regularisation")
    print("    x_{ui} = p_u^T q_i  (score)")
    print("    Sampled triplets (u, i+, i-) where i+ was interacted with")
    print()
    print("  EASE (Embarrassingly Shallow AutoEncoder):")
    print("    B = (X^T X + lambdaI)^{-1}  then  B_{ii} = 0  (no self-loops)")
    print("    Ŷ = XB  (closed-form; very competitive)")
    print()
    print("  Libraries:")
    libs = [
        ("implicit",      "Cython/CUDA iALS and BPR; battle-tested"),
        ("LightFM",       "Hybrid CF; embedding-based; supports side features"),
        ("Cornac",        "Research-friendly CF framework"),
        ("RecBole",       "Comprehensive RecSys benchmark library"),
        ("TorchRec",      "Meta; production-grade embedding tables"),
    ]
    for l, d in libs:
        print(f"  {l:<14} {d}")


if __name__ == "__main__":
    memory_based_cf()
    matrix_factorisation()
    sgd_mf()
    implicit_feedback()
