"""
Working Example 2: LLM Evaluation Metrics
Computes BLEU score, ROUGE-L, character-level perplexity proxy, and
exact match on synthetic reference/hypothesis pairs.
Run: python working_example2.py
"""
from pathlib import Path
from collections import Counter

try:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit("pip install numpy matplotlib")

OUTPUT = Path(__file__).parent / "output"
OUTPUT.mkdir(exist_ok=True)


def ngrams(tokens, n):
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def bleu_n(ref_tokens, hyp_tokens, n):
    ref_ng = Counter(ngrams(ref_tokens, n))
    hyp_ng = Counter(ngrams(hyp_tokens, n))
    clipped = sum(min(c, ref_ng[g]) for g, c in hyp_ng.items())
    total = sum(hyp_ng.values())
    return clipped / (total + 1e-10)


def bleu(ref, hyp, max_n=4):
    rt, ht = ref.lower().split(), hyp.lower().split()
    scores = [bleu_n(rt, ht, n) for n in range(1, max_n + 1)]
    weights = [0.25] * max_n
    log_avg = sum(w * np.log(s + 1e-10) for w, s in zip(weights, scores))
    bp = min(1.0, np.exp(1 - len(rt) / (len(ht) + 1e-10)))
    return bp * np.exp(log_avg)


def lcs_length(a, b):
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = dp[i-1][j-1] + 1 if a[i-1] == b[j-1] else max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]


def rouge_l(ref, hyp):
    rt, ht = ref.lower().split(), hyp.lower().split()
    lcs = lcs_length(rt, ht)
    p = lcs / (len(ht) + 1e-10)
    r = lcs / (len(rt) + 1e-10)
    return 2 * p * r / (p + r + 1e-10)


def demo():
    print("=== LLM Evaluation Metrics ===")
    pairs = [
        ("the cat sat on the mat", "the cat sat on the mat"),
        ("the cat sat on the mat", "a cat is sitting on a mat"),
        ("machine learning is powerful", "machine learning is very powerful"),
        ("the dog ran fast", "a dog ran quickly"),
        ("neural networks learn features", "deep nets extract representations"),
        ("transformers use attention", "attention is used in transformers"),
        ("the weather is sunny today", "today the sun is shining"),
        ("I love natural language processing", "natural language processing is great"),
    ]
    labels = [f"P{i+1}" for i in range(len(pairs))]
    bleu_scores = [bleu(r, h) for r, h in pairs]
    rouge_scores = [rouge_l(r, h) for r, h in pairs]
    em_scores = [float(r.lower().strip() == h.lower().strip()) for r, h in pairs]

    print(f"  {'Pair':<6} {'BLEU':>8} {'ROUGE-L':>8} {'EM':>5}")
    for lab, b, rl, em in zip(labels, bleu_scores, rouge_scores, em_scores):
        print(f"  {lab:<6} {b:>8.3f} {rl:>8.3f} {em:>5.0f}")

    x = np.arange(len(labels))
    width = 0.3
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    bars1 = axes[0].bar(x - width, bleu_scores, width, label="BLEU-4", color="steelblue")
    bars2 = axes[0].bar(x, rouge_scores, width, label="ROUGE-L", color="darkorange")
    bars3 = axes[0].bar(x + width, em_scores, width, label="Exact Match", color="mediumseagreen")
    axes[0].set(xticks=x, xticklabels=labels, ylabel="Score",
                title="LLM Evaluation Metrics per Pair")
    axes[0].legend()
    axes[0].grid(True, axis="y", alpha=0.3)

    # Scatter: BLEU vs ROUGE-L
    axes[1].scatter(bleu_scores, rouge_scores, s=80, color="tomato", zorder=3)
    for i, lab in enumerate(labels):
        axes[1].annotate(lab, (bleu_scores[i], rouge_scores[i]),
                         textcoords="offset points", xytext=(5, 5), fontsize=8)
    axes[1].plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4)
    axes[1].set(xlabel="BLEU-4", ylabel="ROUGE-L",
                title="BLEU vs ROUGE-L Correlation")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT / "llm_evaluation.png", dpi=100)
    plt.close()
    print("  Saved llm_evaluation.png")


if __name__ == "__main__":
    demo()
