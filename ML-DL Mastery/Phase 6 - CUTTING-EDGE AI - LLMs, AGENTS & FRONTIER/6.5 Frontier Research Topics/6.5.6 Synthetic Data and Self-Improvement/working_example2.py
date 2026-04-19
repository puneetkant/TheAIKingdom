"""
Working Example 2: Synthetic Data and Self-Improvement
Data augmentation pipeline and self-training loop simulation.
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


def augment_tabular(X, noise_std=0.1, flip_prob=0.1, rng=None):
    """Feature noise + random column flipping."""
    if rng is None:
        rng = np.random.default_rng(42)
    X_aug = X + rng.normal(0, noise_std, X.shape)
    flip_mask = rng.random(X.shape) < flip_prob
    X_aug = np.where(flip_mask, -X_aug, X_aug)
    return X_aug


def mixup(X, y, alpha=0.2, rng=None):
    """Mixup augmentation."""
    if rng is None:
        rng = np.random.default_rng(42)
    lam = rng.beta(alpha, alpha)
    idx = rng.permutation(len(X))
    X_mix = lam * X + (1 - lam) * X[idx]
    y_mix = lam * y + (1 - lam) * y[idx]
    return X_mix, y_mix


def self_training_loop(X_labeled, y_labeled, X_unlabeled, n_rounds=5, threshold=0.8, rng=None):
    """Simulate self-training: pseudo-label high-confidence unlabeled examples."""
    if rng is None:
        rng = np.random.default_rng(42)
    n_labeled_over_rounds = [len(X_labeled)]
    acc_over_rounds = [0.65]  # Starting accuracy proxy

    for r in range(n_rounds):
        n_unlabeled = len(X_unlabeled)
        if n_unlabeled == 0:
            break
        # Simulate model confidence (improves each round)
        confidence = rng.beta(2 + r, 3 - min(r, 2), n_unlabeled)
        high_conf_mask = confidence >= threshold
        n_added = high_conf_mask.sum()

        # Add pseudo-labeled examples
        X_labeled = np.vstack([X_labeled, X_unlabeled[high_conf_mask]])
        pseudo_labels = rng.integers(0, 2, n_added)
        y_labeled = np.concatenate([y_labeled, pseudo_labels])
        X_unlabeled = X_unlabeled[~high_conf_mask]

        n_labeled_over_rounds.append(len(X_labeled))
        # Accuracy improves with more labeled data (log-ish)
        acc = min(0.95, 0.65 + 0.06 * np.log1p(r + 1) + rng.normal(0, 0.01))
        acc_over_rounds.append(acc)
        print(f"  Round {r+1}: added {n_added} pseudo-labels, total={len(X_labeled)}, acc~={acc:.3f}")

    return n_labeled_over_rounds, acc_over_rounds


def demo():
    print("=== Synthetic Data and Self-Improvement ===")
    rng = np.random.default_rng(42)

    n, d = 100, 8
    X = rng.standard_normal((n, d))
    y = (X[:, 0] + X[:, 1] > 0).astype(float)

    X_aug = augment_tabular(X, noise_std=0.15, rng=rng)
    X_mix, y_mix = mixup(X, y, alpha=0.3, rng=rng)

    # Dataset statistics
    print(f"\n  Original: {X.shape}, Augmented: {X_aug.shape}")
    print(f"  Mixup: {X_mix.shape}")
    print(f"  Feature mean diff (orig vs aug): {np.abs(X.mean(0) - X_aug.mean(0)).mean():.4f}")

    # Self-training
    X_lab = X[:20]; y_lab = y[:20]
    X_unlab = X[20:]
    print("\n  Self-training loop:")
    n_lab_history, acc_history = self_training_loop(X_lab, y_lab, X_unlab, n_rounds=5, rng=rng)

    # Augmentation diversity: pairwise distance increase
    def mean_pairwise_dist(X):
        diffs = X[:, None] - X[None, :]
        return np.sqrt((diffs**2).sum(-1)).mean()

    orig_dist = mean_pairwise_dist(X[:30])
    aug_dist = mean_pairwise_dist(X_aug[:30])
    print(f"\n  Mean pairwise dist: orig={orig_dist:.3f}, augmented={aug_dist:.3f}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Feature distribution: original vs augmented (dim 0)
    axes[0].hist(X[:, 0], bins=20, alpha=0.6, color="steelblue", label="Original")
    axes[0].hist(X_aug[:, 0], bins=20, alpha=0.6, color="tomato", label="Augmented")
    axes[0].hist(X_mix[:, 0], bins=20, alpha=0.6, color="mediumseagreen", label="Mixup")
    axes[0].set(xlabel="Feature 0", ylabel="Count", title="Augmentation: Feature Distribution")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Self-training: labeled data growth
    axes[1].plot(range(len(n_lab_history)), n_lab_history, "o-", color="steelblue", lw=2)
    axes[1].set(xlabel="Self-Training Round", ylabel="# Labeled Examples",
                title="Self-Training: Labeled Data Growth")
    axes[1].grid(True, alpha=0.3)

    # Accuracy improvement
    axes[2].plot(range(len(acc_history)), acc_history, "s-", color="tomato", lw=2)
    axes[2].set(xlabel="Round", ylabel="Accuracy", title="Self-Training: Accuracy Improvement")
    axes[2].set_ylim(0, 1)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT / "synthetic_data.png", dpi=100)
    plt.close()
    print("  Saved synthetic_data.png")


if __name__ == "__main__":
    demo()
