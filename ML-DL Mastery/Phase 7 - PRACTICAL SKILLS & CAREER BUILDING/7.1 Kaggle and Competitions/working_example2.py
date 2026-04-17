"""
Working Example 2: Kaggle and Competitions
Cross-validation pipeline, ensemble stacking, and leaderboard simulation.
Run: python working_example2.py
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


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -30, 30)))


def log_loss(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def accuracy(y_true, y_pred_proba, threshold=0.5):
    return np.mean((y_pred_proba >= threshold) == y_true)


def make_dataset(n=500, d=10, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    X = rng.standard_normal((n, d))
    w = rng.standard_normal(d) * 0.5
    y = (sigmoid(X @ w + rng.normal(0, 0.5, n)) > 0.5).astype(float)
    return X, y


def kfold_cv(X, y, n_folds=5, rng=None):
    """Stratified K-fold cross-validation proxy."""
    if rng is None:
        rng = np.random.default_rng(42)
    n = len(X)
    idx = rng.permutation(n)
    fold_size = n // n_folds
    fold_scores = []

    for fold in range(n_folds):
        val_idx = idx[fold * fold_size: (fold + 1) * fold_size]
        train_idx = np.concatenate([idx[:fold * fold_size], idx[(fold + 1) * fold_size:]])
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        # Proxy model: linear logistic regression (1 gradient step per feature)
        w = np.zeros(X_tr.shape[1])
        for _ in range(100):
            pred = sigmoid(X_tr @ w)
            grad = X_tr.T @ (pred - y_tr) / len(y_tr)
            w -= 0.1 * grad

        val_pred = sigmoid(X_val @ w)
        fold_scores.append(accuracy(y_val, val_pred))

    return fold_scores


def ensemble_stack(base_preds, y, meta_lr=0.1, n_steps=50, rng=None):
    """Meta-learner: weighted average of base model predictions."""
    if rng is None:
        rng = np.random.default_rng(42)
    n_models = len(base_preds)
    weights = np.ones(n_models) / n_models
    losses = []

    for step in range(n_steps):
        stacked_pred = np.array(base_preds).T @ weights
        stacked_pred = sigmoid(stacked_pred)
        loss = log_loss(y, stacked_pred)
        losses.append(loss)
        # Gradient w.r.t. weights
        err = stacked_pred - y
        grad = np.array([np.mean(err * p) for p in base_preds])
        weights -= meta_lr * grad
        weights = np.clip(weights, 0, None)
        weights /= (weights.sum() + 1e-10)

    return weights, losses


def demo():
    print("=== Kaggle and Competitions ===")
    rng = np.random.default_rng(42)
    X, y = make_dataset(n=500, d=12, rng=rng)
    print(f"  Dataset: {X.shape}, positive rate: {y.mean():.2f}")

    # K-Fold CV
    fold_scores = kfold_cv(X, y, n_folds=5, rng=rng)
    print(f"\n  5-Fold CV accuracy: {[f'{s:.3f}' for s in fold_scores]}")
    print(f"  Mean: {np.mean(fold_scores):.3f} ± {np.std(fold_scores):.3f}")

    # Three simulated base models with different biases
    w1 = rng.standard_normal(12) * 0.3
    w2 = rng.standard_normal(12) * 0.5
    w3 = rng.standard_normal(12) * 0.8
    pred1 = sigmoid(X @ w1)
    pred2 = sigmoid(X @ w2)
    pred3 = sigmoid(X @ w3)

    # Ensemble stacking
    weights, stack_losses = ensemble_stack([pred1, pred2, pred3], y, n_steps=100, rng=rng)
    print(f"\n  Ensemble weights: {weights.round(3)}")
    final_pred = sigmoid(np.array([pred1, pred2, pred3]).T @ weights)
    print(f"  Ensemble accuracy: {accuracy(y, final_pred):.3f}")
    print(f"  Ensemble log-loss: {log_loss(y, final_pred):.4f}")

    # Simulate leaderboard (public/private split)
    n_teams = 30
    public_scores = 0.7 + rng.standard_normal(n_teams) * 0.05
    # Private scores: some teams overfit (correlated noise)
    private_noise = 0.7 * rng.standard_normal(n_teams) + 0.3 * rng.standard_normal(n_teams)
    private_scores = public_scores + private_noise * 0.03
    public_rank = np.argsort(public_scores)[::-1]
    private_rank = np.argsort(private_scores)[::-1]

    # Rank correlation (Spearman proxy)
    pub_rank_arr = np.argsort(public_rank)
    priv_rank_arr = np.argsort(private_rank)
    rank_corr = np.corrcoef(pub_rank_arr, priv_rank_arr)[0, 1]
    print(f"\n  Leaderboard shake-up rank correlation: {rank_corr:.3f}")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # CV fold scores
    axes[0][0].bar(range(1, 6), fold_scores, color="steelblue", alpha=0.8)
    axes[0][0].axhline(np.mean(fold_scores), color="red", linestyle="--", label=f"Mean={np.mean(fold_scores):.3f}")
    axes[0][0].set(xlabel="Fold", ylabel="Accuracy", title="5-Fold Cross-Validation")
    axes[0][0].set_ylim(0, 1)
    axes[0][0].legend()
    axes[0][0].grid(True, axis="y", alpha=0.3)

    # Stacking loss curve
    axes[0][1].plot(stack_losses, color="tomato", lw=2)
    axes[0][1].set(xlabel="Meta-learner Step", ylabel="Log-Loss",
                   title="Ensemble Stacking Meta-Learner Loss")
    axes[0][1].grid(True, alpha=0.3)

    # Leaderboard: public vs private
    axes[1][0].scatter(public_scores, private_scores, alpha=0.6, color="mediumseagreen", s=40)
    axes[1][0].plot([min(public_scores), max(public_scores)],
                     [min(public_scores), max(public_scores)], "k--", alpha=0.4)
    axes[1][0].set(xlabel="Public LB Score", ylabel="Private LB Score",
                   title=f"Leaderboard Shake-up (ρ={rank_corr:.2f})")
    axes[1][0].grid(True, alpha=0.3)

    # Ensemble weights
    axes[1][1].bar(["Model 1", "Model 2", "Model 3"], weights,
                    color=["#3498db", "#e74c3c", "#2ecc71"])
    axes[1][1].set(ylabel="Weight", title="Ensemble Stacking Weights")
    axes[1][1].grid(True, axis="y", alpha=0.3)
    for i, w in enumerate(weights):
        axes[1][1].text(i, w + 0.01, f"{w:.3f}", ha="center")

    plt.tight_layout()
    plt.savefig(OUTPUT / "kaggle_competition.png", dpi=100)
    plt.close()
    print("  Saved kaggle_competition.png")


if __name__ == "__main__":
    demo()
