"""
Working Example 2: Prompt Engineering
Compares few-shot vs zero-shot accuracy on a synthetic classification task
and builds a chain-of-thought prompt template.
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

POSITIVE_WORDS = {"good", "great", "excellent", "love", "fantastic", "wonderful", "amazing"}
NEGATIVE_WORDS = {"bad", "terrible", "awful", "hate", "poor", "horrible", "worst"}


def zero_shot_classify(text):
    """Zero-shot: no examples, keyword-based proxy."""
    words = set(text.lower().split())
    pos = len(words & POSITIVE_WORDS)
    neg = len(words & NEGATIVE_WORDS)
    return "positive" if pos > neg else ("negative" if neg > pos else "neutral")


def few_shot_classify(text, examples):
    """Few-shot: use examples to calibrate threshold."""
    words = set(text.lower().split())
    pos = len(words & POSITIVE_WORDS)
    neg = len(words & NEGATIVE_WORDS)
    # Calibrate with example labels
    pos_examples = sum(1 for _, label in examples if label == "positive")
    bias = 0.3 if pos_examples > len(examples) / 2 else -0.3
    score = pos - neg + bias
    return "positive" if score > 0 else "negative"


def cot_template(question, reasoning_steps):
    """Chain-of-thought prompt template."""
    steps = "\n".join(f"  Step {i+1}: {s}" for i, s in enumerate(reasoning_steps))
    return f"""Question: {question}

Let's think step by step:
{steps}

Therefore, the answer is:"""


def demo():
    print("=== Prompt Engineering: Few-Shot vs Zero-Shot ===")
    rng = np.random.default_rng(42)

    test_texts = [
        ("This product is great and wonderful!", "positive"),
        ("Terrible service, I hate it.", "negative"),
        ("The product is good but delivery was poor.", "neutral"),
        ("Amazing quality, fantastic experience!", "positive"),
        ("Worst purchase ever, absolutely horrible.", "negative"),
        ("Excellent product, love the design!", "positive"),
        ("The bad quality is awful.", "negative"),
        ("A great product with some bad aspects.", "neutral"),
    ]

    few_shot_examples = [
        ("I love this!", "positive"),
        ("This is terrible.", "negative"),
        ("Great product!", "positive"),
    ]

    zero_shot_correct, few_shot_correct = 0, 0
    for text, true_label in test_texts:
        z = zero_shot_classify(text)
        f = few_shot_classify(text, few_shot_examples)
        # For neutral ground truth, both positive and negative can match
        zc = int(z == true_label)
        fc = int(f == true_label)
        zero_shot_correct += zc
        few_shot_correct += fc
        print(f"  '{text[:35]}...' → true={true_label}, zs={z}, fs={f}")

    zs_acc = zero_shot_correct / len(test_texts)
    fs_acc = few_shot_correct / len(test_texts)
    print(f"\n  Zero-shot accuracy: {zs_acc:.2f}")
    print(f"  Few-shot accuracy:  {fs_acc:.2f}")

    # Simulate accuracy vs # examples
    n_examples = [0, 1, 2, 3, 5, 8, 13, 20]
    zs_accs = [0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50]
    fs_accs = [0.50, 0.62, 0.68, 0.73, 0.79, 0.83, 0.86, 0.88]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(n_examples, zs_accs, "o--", color="steelblue", lw=2, label="Zero-Shot")
    axes[0].plot(n_examples, fs_accs, "s-", color="tomato", lw=2, label="Few-Shot")
    axes[0].set(xlabel="Number of Examples", ylabel="Accuracy",
                title="Prompt Engineering: Few-Shot vs Zero-Shot")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 1)

    # CoT template visualisation
    cot = cot_template(
        "Is 17 a prime number?",
        ["17 is not divisible by 2 (17/2 = 8.5)",
         "17 is not divisible by 3 (17/3 ≈ 5.67)",
         "17 is not divisible by 4, 5, 6, ..., √17 ≈ 4.1",
         "No divisor found → 17 is prime"]
    )
    print("\n  CoT Template:\n" + cot)

    # Visualise prompt technique comparison
    techniques = ["Zero-Shot", "Few-Shot (3)", "Few-Shot (8)", "CoT Zero-Shot", "CoT Few-Shot"]
    accuracy_sim = [0.50, 0.73, 0.83, 0.65, 0.89]
    colors = ["#95a5a6", "#3498db", "#2980b9", "#e67e22", "#e74c3c"]
    axes[1].bar(techniques, accuracy_sim, color=colors)
    axes[1].set(ylabel="Simulated Accuracy", title="Prompt Technique Comparison")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, axis="y", alpha=0.3)
    for i, acc in enumerate(accuracy_sim):
        axes[1].text(i, acc + 0.02, f"{acc:.2f}", ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT / "prompt_engineering.png", dpi=100)
    plt.close()
    print("  Saved prompt_engineering.png")


if __name__ == "__main__":
    demo()
