"""
Working Example 2: Autoregressive Generation — character-level language model
=============================================================================
Trains a bigram (n-gram) character model and samples from it.

Run:  python working_example2.py
"""
from pathlib import Path
from collections import Counter, defaultdict
import random

try:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit("pip install numpy matplotlib")

OUTPUT = Path(__file__).parent / "output"
OUTPUT.mkdir(exist_ok=True)

CORPUS = (
    "the quick brown fox jumps over the lazy dog. "
    "pack my box with five dozen liquor jugs. "
    "sphinx of black quartz judge my vow. "
    "how vexingly quick daft zebras jump! "
    "the five boxing wizards jump quickly. " * 10
)

def build_ngram(text, n=2):
    model = defaultdict(Counter)
    for i in range(len(text) - n):
        ctx = text[i:i+n]
        nxt = text[i+n]
        model[ctx][nxt] += 1
    return model

def sample_ngram(model, ctx, temperature=1.0):
    counts = model.get(ctx, {})
    if not counts:
        return " "
    chars, logits = zip(*counts.items())
    logits = np.array(logits, float) / temperature
    probs = np.exp(logits - logits.max()); probs /= probs.sum()
    return np.random.choice(list(chars), p=probs)

def generate(model, n=2, length=120, seed="th"):
    text = seed
    for _ in range(length):
        ctx = text[-n:]
        text += sample_ngram(model, ctx)
    return text

def demo():
    print("=== Autoregressive Character n-gram Model ===")
    model2 = build_ngram(CORPUS, n=2)
    model3 = build_ngram(CORPUS, n=3)
    print("  Bigram model vocab:", len(model2), "contexts")
    print("  Trigram model vocab:", len(model3), "contexts")

    print("\n  Bigram sample:")
    print("  ", generate(model2, n=2, seed="th"))
    print("\n  Trigram sample:")
    print("  ", generate(model3, n=3, seed="the"))

    # Perplexity on held-out slice
    test = CORPUS[300:400]
    log_prob = 0; count = 0
    for i in range(len(test) - 2):
        ctx = test[i:i+2]; nxt = test[i+2]
        ctr = model2.get(ctx, Counter()); total = sum(ctr.values()) or 1
        p = ctr.get(nxt, 0) / total + 1e-8
        log_prob += np.log(p); count += 1
    ppl = np.exp(-log_prob / count)
    print(f"\n  Bigram perplexity on held-out text: {ppl:.2f}")

    # Visualise transition matrix for top chars
    common_chars = [c for c, _ in Counter(CORPUS).most_common(12)]
    mat = np.zeros((12, 12))
    for i, c1 in enumerate(common_chars):
        for j, c2 in enumerate(common_chars):
            ctr = model2.get(c1+c2, Counter())
            mat[i, j] = sum(ctr.values())
    plt.figure(figsize=(6, 5))
    plt.imshow(np.log1p(mat), cmap="Blues")
    plt.xticks(range(12), common_chars); plt.yticks(range(12), common_chars)
    plt.colorbar(); plt.title("Bigram Transition Matrix (log count)")
    plt.tight_layout(); plt.savefig(OUTPUT / "autoregressive_ngram.png"); plt.close()
    print("  Saved autoregressive_ngram.png")

if __name__ == "__main__":
    demo()
