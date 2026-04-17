"""
Working Example 2: Reasoning and Test-Time Compute
Majority voting, best-of-N selection, and accuracy vs compute budget.
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


def simulate_answer(true_answer, p_correct=0.6, rng=None):
    """Sample a model answer: correct with probability p_correct."""
    if rng is None:
        rng = np.random.default_rng()
    if rng.random() < p_correct:
        return true_answer
    else:
        alternatives = [a for a in ["A", "B", "C", "D"] if a != true_answer]
        return rng.choice(alternatives)


def majority_vote(answers):
    """Return the most common answer."""
    counts = {}
    for a in answers:
        counts[a] = counts.get(a, 0) + 1
    return max(counts, key=counts.get)


def best_of_n(answers, scores):
    """Return the answer with the highest reward model score."""
    return answers[np.argmax(scores)]


def accuracy_vs_n(p_correct, n_values, n_problems=200, rng=None):
    """Compute majority vote accuracy for different N values."""
    if rng is None:
        rng = np.random.default_rng(42)
    true_answers = rng.choice(["A", "B", "C", "D"], n_problems)
    accs = []
    for N in n_values:
        correct = 0
        for true in true_answers:
            samples = [simulate_answer(true, p_correct, rng) for _ in range(N)]
            pred = majority_vote(samples)
            if pred == true:
                correct += 1
        accs.append(correct / n_problems)
    return accs


def demo():
    print("=== Reasoning & Test-Time Compute ===")
    rng = np.random.default_rng(42)
    n_values = [1, 2, 4, 8, 16, 32, 64]

    # Accuracy vs N for different model strengths
    p_values = [0.4, 0.55, 0.7, 0.85]
    accs_by_p = [accuracy_vs_n(p, n_values, n_problems=100, rng=rng) for p in p_values]

    for p, accs in zip(p_values, accs_by_p):
        print(f"  p={p}: N=1: {accs[0]:.2f}, N=64: {accs[-1]:.2f}")

    # Best-of-N vs majority vote
    true = "A"
    N_list = [1, 2, 4, 8, 16, 32]
    bon_accs, maj_accs = [], []
    n_trials = 300
    for N in N_list:
        bon_correct, maj_correct = 0, 0
        for _ in range(n_trials):
            answers = [simulate_answer(true, 0.6, rng) for _ in range(N)]
            # Best-of-N: reward = 1 if correct, else random [0,1]
            scores = np.array([1.0 + rng.random() * 0.1 if a == true else rng.random() * 0.9
                                for a in answers])
            bon_pred = best_of_n(answers, scores)
            maj_pred = majority_vote(answers)
            bon_correct += int(bon_pred == true)
            maj_correct += int(maj_pred == true)
        bon_accs.append(bon_correct / n_trials)
        maj_accs.append(maj_correct / n_trials)

    print(f"\n  Best-of-N@32: {bon_accs[-1]:.2f}, Majority-32: {maj_accs[-1]:.2f}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Accuracy vs N by model strength
    for p, accs in zip(p_values, accs_by_p):
        axes[0].plot(n_values, accs, "o-", lw=2, ms=5, label=f"p={p}")
    axes[0].set(xlabel="N (# samples)", ylabel="Accuracy",
                title="Majority Vote Accuracy vs Test-Time Compute")
    axes[0].legend()
    axes[0].set_xscale("log", base=2)
    axes[0].grid(True, alpha=0.3)

    # Best-of-N vs Majority vote
    axes[1].plot(N_list, bon_accs, "o-", color="steelblue", lw=2, label="Best-of-N")
    axes[1].plot(N_list, maj_accs, "s--", color="tomato", lw=2, label="Majority Vote")
    axes[1].set(xlabel="N (# samples)", ylabel="Accuracy",
                title="Best-of-N vs Majority Vote (p=0.6)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Compute budget (N × FLOPs proxy)
    flops_per_sample = 1e9
    compute = np.array(n_values) * flops_per_sample
    accs_60 = accs_by_p[2]  # p=0.7
    axes[2].plot(compute / 1e9, accs_60, "o-", color="mediumseagreen", lw=2)
    axes[2].set(xlabel="Compute (GFLOPs)", ylabel="Accuracy",
                title="Accuracy vs Compute Budget (p=0.7)")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT / "reasoning_compute.png", dpi=100)
    plt.close()
    print("  Saved reasoning_compute.png")


if __name__ == "__main__":
    demo()
