"""
Working Example 2: Supervised Fine-Tuning (SFT)
Uses logistic regression as a proxy to demonstrate fine-tuning:
few-shot adaptation vs base model, loss curves, accuracy comparison.
Run: python working_example2.py
"""
from pathlib import Path

try:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
except ImportError:
    raise SystemExit("pip install numpy matplotlib scikit-learn")

OUTPUT = Path(__file__).parent / "output"
OUTPUT.mkdir(exist_ok=True)


def simulated_loss(epochs, start, end, noise_std=0.015):
    rng = np.random.default_rng(42)
    t = np.linspace(0, 1, epochs)
    return start * np.exp(-(start - end) * t / start) + end * (1 - np.exp(-3 * t)) \
           + rng.normal(0, noise_std, epochs)


def demo():
    print("=== Supervised Fine-Tuning (SFT) ===")
    rng = np.random.default_rng(0)

    # Synthetic dataset: general task (base) vs fine-tuning task
    X_full, y_full = make_classification(n_samples=2000, n_features=20,
                                         n_informative=10, random_state=42)
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X_full, y_full, test_size=0.2, random_state=42)

    # Base model: trained on large general data
    base = LogisticRegression(max_iter=500, random_state=42)
    base.fit(X_train_full, y_train_full)
    base_acc = accuracy_score(y_test, base.predict(X_test))

    # Fine-tuned model: trained on small domain-specific subset (few-shot proxy)
    few_shot_sizes = [10, 20, 50, 100, 200, 500]
    ft_accs = []
    for n in few_shot_sizes:
        idx = rng.choice(len(X_train_full), n, replace=False)
        ft = LogisticRegression(max_iter=500, random_state=42)
        ft.fit(X_train_full[idx], y_train_full[idx])
        ft_accs.append(accuracy_score(y_test, ft.predict(X_test)))

    print(f"  Base model accuracy (full data): {base_acc:.3f}")
    for n, acc in zip(few_shot_sizes, ft_accs):
        print(f"  Fine-tuned ({n:4d} samples): {acc:.3f}")

    # Simulated training loss curves
    epochs = 50
    base_loss = simulated_loss(epochs, 0.7, 0.18)
    ft_loss = simulated_loss(epochs, 0.65, 0.12, noise_std=0.02)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Accuracy vs fine-tuning samples
    axes[0].axhline(base_acc, color="steelblue", linestyle="--", lw=2, label="Base (full)")
    axes[0].plot(few_shot_sizes, ft_accs, "o-", color="darkorange", lw=2, label="Fine-tuned")
    axes[0].set(xlabel="Fine-tuning Samples", ylabel="Test Accuracy",
                title="SFT: Accuracy vs Data Size")
    axes[0].set_xscale("log")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss curves
    axes[1].plot(range(epochs), base_loss, color="steelblue", lw=2, label="Base model loss")
    axes[1].plot(range(epochs), ft_loss, color="tomato", lw=2, label="Fine-tuned loss")
    axes[1].set(xlabel="Epoch", ylabel="Cross-Entropy Loss",
                title="SFT: Training Loss Curves")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT / "sft_comparison.png", dpi=100)
    plt.close()
    print("  Saved sft_comparison.png")


if __name__ == "__main__":
    demo()
