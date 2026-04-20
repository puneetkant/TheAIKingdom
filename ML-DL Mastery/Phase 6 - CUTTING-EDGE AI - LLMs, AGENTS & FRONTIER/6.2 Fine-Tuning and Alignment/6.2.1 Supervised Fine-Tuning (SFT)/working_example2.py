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


def demo_learning_rate_sensitivity():
    """Show how learning rate affects SFT convergence on a simple proxy."""
    print("\n=== Learning Rate Sensitivity ===")
    epochs = 40
    lrs = {"lr=0.001": (0.65, 0.15), "lr=0.01": (0.65, 0.10), "lr=0.1": (0.65, 0.25)}
    plt.figure(figsize=(6, 4))
    for label, (start, end) in lrs.items():
        loss = simulated_loss(epochs, start, end, noise_std=0.02)
        plt.plot(range(epochs), loss, lw=2, label=label)
        print(f"  {label}: final loss = {loss[-1]:.4f}")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title("SFT: Learning Rate Sensitivity")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT / "sft_lr_sensitivity.png", dpi=100); plt.close()
    print("  Saved sft_lr_sensitivity.png")


def demo_catastrophic_forgetting():
    """Proxy: fine-tuning on small domain hurts performance on general task."""
    print("\n=== Catastrophic Forgetting ===")
    rng = np.random.default_rng(5)
    X_full, y_full = make_classification(n_samples=2000, n_features=20,
                                          n_informative=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y_full, test_size=0.3, random_state=42)
    # Base model on full training set
    base = LogisticRegression(max_iter=500, random_state=42)
    base.fit(X_train, y_train)
    base_acc = accuracy_score(y_test, base.predict(X_test))
    # Fine-tune on tiny biased subset (class 0 heavily overrepresented)
    class0_idx = np.where(y_train == 0)[0][:28]
    class1_idx = np.where(y_train == 1)[0][:2]
    biased_idx = np.concatenate([class0_idx, class1_idx])
    ft = LogisticRegression(max_iter=500, random_state=42, C=0.1)
    ft.fit(X_train[biased_idx], y_train[biased_idx])
    ft_acc = accuracy_score(y_test, ft.predict(X_test))
    print(f"  Base model test accuracy:          {base_acc:.3f}")
    print(f"  After biased fine-tune accuracy:   {ft_acc:.3f}")
    print(f"  Forgetting degradation:            {base_acc - ft_acc:.3f}")
    plt.figure(figsize=(5, 3))
    plt.bar(["Base Model", "Fine-tuned (biased)"], [base_acc, ft_acc],
            color=["steelblue", "tomato"], edgecolor="white")
    plt.ylabel("Test Accuracy"); plt.title("Catastrophic Forgetting Demo")
    plt.ylim(0.4, 1.0); plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT / "catastrophic_forgetting.png", dpi=100); plt.close()
    print("  Saved catastrophic_forgetting.png")


if __name__ == "__main__":
    demo()
    demo_learning_rate_sensitivity()
    demo_catastrophic_forgetting()
