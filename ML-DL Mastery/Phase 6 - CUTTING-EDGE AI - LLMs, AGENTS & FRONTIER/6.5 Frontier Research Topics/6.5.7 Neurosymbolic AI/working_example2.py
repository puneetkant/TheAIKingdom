"""
Working Example 2: Neurosymbolic AI
Hybrid neural + symbolic solver: neural component predicts digit,
symbolic component applies arithmetic rules.
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


# --- Symbolic component ---
class SymbolicSolver:
    RULES = {
        "add": lambda a, b: a + b,
        "mul": lambda a, b: a * b,
        "max": lambda a, b: max(a, b),
        "min": lambda a, b: min(a, b),
    }

    def solve(self, op, a, b):
        fn = self.RULES.get(op)
        if fn is None:
            raise ValueError(f"Unknown op: {op}")
        return fn(a, b)


# --- Neural component (proxy) ---
def neural_predict_digit(input_vec, noise_std=0.3, rng=None):
    """Proxy: argmax of noisy 10-class softmax over input."""
    if rng is None:
        rng = np.random.default_rng(42)
    logits = input_vec + rng.normal(0, noise_std, len(input_vec))
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()
    return np.argmax(probs), probs


def hybrid_predict(input_a, input_b, op, noise_std=0.3, rng=None):
    """Full hybrid pipeline: neural digit prediction → symbolic solve."""
    digit_a, probs_a = neural_predict_digit(input_a, noise_std, rng)
    digit_b, probs_b = neural_predict_digit(input_b, noise_std, rng)
    solver = SymbolicSolver()
    result = solver.solve(op, int(digit_a), int(digit_b))
    return result, int(digit_a), int(digit_b), probs_a, probs_b


def demo():
    print("=== Neurosymbolic AI ===")
    rng = np.random.default_rng(42)

    # Inputs: one-hot-ish vectors for digits 0-9
    n_problems = 20
    true_a = rng.integers(0, 10, n_problems)
    true_b = rng.integers(0, 10, n_problems)
    ops = rng.choice(["add", "mul", "max", "min"], n_problems)

    solver = SymbolicSolver()
    true_results = [solver.solve(op, int(a), int(b)) for op, a, b in zip(ops, true_a, true_b)]

    correct_pure_neural = 0
    correct_hybrid = 0
    hybrid_results = []
    for i in range(n_problems):
        # Input: one-hot + noise
        input_a = np.zeros(10); input_a[true_a[i]] = 2.0
        input_b = np.zeros(10); input_b[true_b[i]] = 2.0
        result, da, db, pa, pb = hybrid_predict(input_a, input_b, ops[i], noise_std=0.5, rng=rng)
        hybrid_results.append(result)
        # Hybrid correct if both digits correct and result matches
        if da == true_a[i] and db == true_b[i]:
            correct_hybrid += 1
            correct_pure_neural += 1
        elif solver.solve(ops[i], da, db) == true_results[i]:
            correct_hybrid += 1  # symbolic saved it

    hybrid_acc = correct_hybrid / n_problems
    neural_acc = correct_pure_neural / n_problems
    print(f"  Hybrid accuracy:       {hybrid_acc:.2f}")
    print(f"  Pure neural accuracy:  {neural_acc:.2f}")

    # Noise sensitivity analysis
    noise_levels = [0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 2.0]
    hybrid_accs, neural_accs = [], []
    for noise in noise_levels:
        h_correct = 0; n_correct = 0
        for i in range(n_problems):
            input_a = np.zeros(10); input_a[true_a[i]] = 2.0
            input_b = np.zeros(10); input_b[true_b[i]] = 2.0
            _, da, db, _, _ = hybrid_predict(input_a, input_b, ops[i], noise_std=noise, rng=rng)
            if da == true_a[i] and db == true_b[i]:
                n_correct += 1
            h_symbolic_result = solver.solve(ops[i], da, db)
            if h_symbolic_result == true_results[i]:
                h_correct += 1
        hybrid_accs.append(h_correct / n_problems)
        neural_accs.append(n_correct / n_problems)

    # Op distribution
    from collections import Counter
    op_counts = Counter(ops)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Noise sensitivity
    axes[0].plot(noise_levels, hybrid_accs, "o-", color="steelblue", lw=2, label="Hybrid")
    axes[0].plot(noise_levels, neural_accs, "s--", color="tomato", lw=2, label="Neural Only")
    axes[0].set(xlabel="Noise σ", ylabel="Accuracy",
                title="Neurosymbolic vs Neural: Noise Robustness")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Operator accuracy breakdown
    for op in ["add", "mul", "max", "min"]:
        idxs = [i for i in range(n_problems) if ops[i] == op]
        if idxs:
            op_correct = sum(1 for i in idxs if hybrid_results[i] == true_results[i])
            axes[1].bar(op, op_correct / len(idxs) if idxs else 0,
                        color={"add": "#3498db", "mul": "#e74c3c", "max": "#2ecc71", "min": "#9b59b6"}[op])
    axes[1].set(ylabel="Accuracy", title="Accuracy by Symbolic Operation")
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, axis="y", alpha=0.3)

    # System comparison bar chart
    axes[2].bar(["Hybrid\n(Neuro+Sym)", "Neural\nOnly", "Symbolic\nOnly"],
                [hybrid_acc, neural_acc, 1.0],
                color=["steelblue", "tomato", "mediumseagreen"])
    axes[2].set(ylabel="Accuracy", title="System Comparison")
    axes[2].set_ylim(0, 1.1)
    axes[2].grid(True, axis="y", alpha=0.3)
    for i, v in enumerate([hybrid_acc, neural_acc, 1.0]):
        axes[2].text(i, v + 0.02, f"{v:.2f}", ha="center")

    plt.tight_layout()
    plt.savefig(OUTPUT / "neurosymbolic.png", dpi=100)
    plt.close()
    print("  Saved neurosymbolic.png")


if __name__ == "__main__":
    demo()
