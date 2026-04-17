"""
Working Example 2: Interview Preparation
ML Q&A quiz scorer, time complexity table, and coding problem tracker.
Run: python working_example2.py
"""
from pathlib import Path
import time

try:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit("pip install numpy matplotlib")

OUTPUT = Path(__file__).parent / "output"
OUTPUT.mkdir(exist_ok=True)


QUESTIONS = [
    {"q": "What is the bias-variance tradeoff?",
     "keywords": ["bias", "variance", "tradeoff", "underfitting", "overfitting"],
     "difficulty": "Easy"},
    {"q": "Explain gradient descent variants (SGD, Adam, RMSProp).",
     "keywords": ["sgd", "adam", "rmsprop", "momentum", "learning rate"],
     "difficulty": "Medium"},
    {"q": "What is the curse of dimensionality?",
     "keywords": ["dimensionality", "sparse", "distance", "exponential"],
     "difficulty": "Medium"},
    {"q": "Explain the attention mechanism.",
     "keywords": ["query", "key", "value", "softmax", "dot product"],
     "difficulty": "Medium"},
    {"q": "What is regularisation (L1 vs L2)?",
     "keywords": ["l1", "lasso", "l2", "ridge", "penalty", "sparsity"],
     "difficulty": "Easy"},
    {"q": "What are LoRA and PEFT?",
     "keywords": ["lora", "peft", "rank", "efficient", "fine-tuning"],
     "difficulty": "Hard"},
    {"q": "Explain backpropagation.",
     "keywords": ["chain rule", "gradient", "backward", "loss"],
     "difficulty": "Medium"},
    {"q": "What is a confusion matrix?",
     "keywords": ["true positive", "false positive", "precision", "recall"],
     "difficulty": "Easy"},
]

COMPLEXITY = {
    "Linear Search":       ("O(n)",      "O(1)"),
    "Binary Search":       ("O(log n)",  "O(1)"),
    "Bubble Sort":         ("O(n²)",     "O(1)"),
    "Merge Sort":          ("O(n log n)","O(n)"),
    "Quick Sort (avg)":    ("O(n log n)","O(log n)"),
    "Hash Map lookup":     ("O(1) avg",  "O(n)"),
    "BFS/DFS":             ("O(V+E)",    "O(V)"),
    "Dijkstra":            ("O(E log V)","O(V)"),
}


def score_answer(answer_text, keywords):
    """Score answer by keyword coverage."""
    text = answer_text.lower()
    hits = sum(1 for kw in keywords if kw in text)
    return hits / len(keywords)


def simulate_quiz(n_rounds=5, rng=None):
    """Simulate quiz performance over practice rounds."""
    if rng is None:
        rng = np.random.default_rng(42)
    scores_by_round = []
    for r in range(n_rounds):
        round_scores = []
        for q in QUESTIONS:
            # Simulate improving score each round
            base_score = 0.4 + 0.1 * r + rng.normal(0, 0.1)
            round_scores.append(np.clip(base_score, 0, 1))
        scores_by_round.append(round_scores)
    return scores_by_round


def demo():
    print("=== Interview Preparation ===")
    rng = np.random.default_rng(42)

    # Show Q&A summary
    print("\n  ML Q&A Bank:")
    for i, q in enumerate(QUESTIONS):
        print(f"  [{q['difficulty']:6s}] Q{i+1}: {q['q'][:55]}...")

    # Simulate a practice answer
    sample_answer = "The bias-variance tradeoff describes the tension between underfitting (high bias) and overfitting (high variance)."
    score = score_answer(sample_answer, QUESTIONS[0]["keywords"])
    print(f"\n  Sample Q1 answer score: {score:.2f}")

    # Quiz simulation
    scores_by_round = simulate_quiz(n_rounds=6, rng=rng)
    round_means = [np.mean(r) for r in scores_by_round]

    # Complexity table printout
    print("\n  Algorithm Complexity Table:")
    print(f"  {'Algorithm':20s} {'Time':15s} {'Space':10s}")
    print("  " + "-" * 47)
    for algo, (time_c, space_c) in COMPLEXITY.items():
        print(f"  {algo:20s} {time_c:15s} {space_c:10s}")

    # Difficulty breakdown
    difficulty_acc = {"Easy": [], "Medium": [], "Hard": []}
    for i, q in enumerate(QUESTIONS):
        difficulty_acc[q["difficulty"]].append(scores_by_round[-1][i])
    diff_means = {d: np.mean(v) if v else 0 for d, v in difficulty_acc.items()}
    print(f"\n  Final round accuracy by difficulty: {diff_means}")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # Score per question (final round)
    final_scores = scores_by_round[-1]
    colors = ["mediumseagreen" if s >= 0.7 else "tomato" for s in final_scores]
    axes[0][0].bar(range(1, len(QUESTIONS) + 1), final_scores, color=colors)
    axes[0][0].axhline(0.7, color="black", linestyle="--", alpha=0.5, label="Target (0.7)")
    axes[0][0].set(xlabel="Question #", ylabel="Score",
                   title="Final Round: Q&A Scores")
    axes[0][0].set_ylim(0, 1.1)
    axes[0][0].legend()
    axes[0][0].grid(True, axis="y", alpha=0.3)

    # Progress over rounds
    axes[0][1].plot(range(1, 7), round_means, "o-", color="steelblue", lw=2)
    for i, q_data in enumerate(zip(*scores_by_round)):
        axes[0][1].plot(range(1, 7), q_data, alpha=0.2, lw=1, color="steelblue")
    axes[0][1].set(xlabel="Practice Round", ylabel="Mean Score",
                   title="Quiz Performance Over Practice")
    axes[0][1].set_ylim(0, 1)
    axes[0][1].grid(True, alpha=0.3)

    # Difficulty breakdown
    diffs = list(diff_means.keys())
    diff_vals = list(diff_means.values())
    colors_d = ["#2ecc71", "#f39c12", "#e74c3c"]
    axes[1][0].bar(diffs, diff_vals, color=colors_d)
    axes[1][0].axhline(0.7, color="black", linestyle="--", alpha=0.5)
    axes[1][0].set(ylabel="Mean Score", title="Accuracy by Difficulty")
    axes[1][0].set_ylim(0, 1)
    axes[1][0].grid(True, axis="y", alpha=0.3)

    # Complexity ladder (bar chart with algo length proxy)
    algo_names = list(COMPLEXITY.keys())
    # Complexity rank (manual)
    complexity_rank = [1, 2, 3, 4, 4, 2, 3, 4]
    complexity_colors = {1: "#2ecc71", 2: "#f1c40f", 3: "#e67e22", 4: "#e74c3c"}
    axes[1][1].barh(algo_names, complexity_rank,
                    color=[complexity_colors[r] for r in complexity_rank])
    axes[1][1].set(xlabel="Complexity Tier", title="Algorithm Complexity Ladder")
    axes[1][1].grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT / "interview_prep.png", dpi=100)
    plt.close()
    print("  Saved interview_prep.png")


if __name__ == "__main__":
    demo()
