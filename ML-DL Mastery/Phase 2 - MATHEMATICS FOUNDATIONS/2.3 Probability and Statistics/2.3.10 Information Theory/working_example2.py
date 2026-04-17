"""
Working Example 2: Information Theory — Entropy, KL Divergence, Mutual Information
===================================================================================
Shannon entropy, cross-entropy loss, KL divergence, mutual information,
Jensen-Shannon divergence — with ML applications.

Run:  python working_example2.py
"""
import math
from pathlib import Path
try:
    import numpy as np
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit("pip install numpy matplotlib")

OUTPUT = Path(__file__).parent / "output"
OUTPUT.mkdir(exist_ok=True)

def safe_entropy(p, base=2):
    """Shannon entropy in bits (base=2) or nats (base=e)."""
    p = np.array(p, dtype=float)
    p = p / p.sum()
    mask = p > 0
    if base == 2:
        return -np.sum(p[mask] * np.log2(p[mask]))
    else:
        return -np.sum(p[mask] * np.log(p[mask]))

def demo_entropy():
    print("=== Shannon Entropy ===")
    # Fair coin: max entropy
    print(f"  H(fair coin) = {safe_entropy([0.5, 0.5]):.4f} bits  (max for binary = 1)")
    # Biased coin
    for p in [0.1, 0.3, 0.5, 0.7, 0.9]:
        print(f"  H(Bernoulli({p})) = {safe_entropy([p, 1-p]):.4f} bits")

    # 6-sided die
    print(f"  H(fair die) = {safe_entropy([1/6]*6):.4f} bits  (=log2(6)={math.log2(6):.4f})")

    # Plot entropy vs p
    ps = np.linspace(0.01, 0.99, 200)
    hs = [-p*math.log2(p) - (1-p)*math.log2(1-p) for p in ps]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ps, hs); ax.set_xlabel("p"); ax.set_ylabel("H(p) bits"); ax.set_title("Binary entropy")
    fig.savefig(OUTPUT / "binary_entropy.png", dpi=120, bbox_inches="tight")
    plt.close(fig); print(f"  Saved: binary_entropy.png")

def demo_kl_divergence():
    print("\n=== KL Divergence ===")
    # KL(P||Q) = Σ p * log(p/q)  — not symmetric
    P = np.array([0.4, 0.3, 0.2, 0.1])
    Q_close = np.array([0.35, 0.3, 0.25, 0.1])
    Q_far   = np.array([0.1, 0.1, 0.1, 0.7])

    def kl(p, q):
        p, q = np.array(p, dtype=float), np.array(q, dtype=float)
        mask = p > 0
        return np.sum(p[mask] * np.log(p[mask] / q[mask]))

    print(f"  KL(P||Q_close) = {kl(P, Q_close):.4f} nats")
    print(f"  KL(P||Q_far)   = {kl(P, Q_far):.4f} nats")
    print(f"  KL is asymmetric: KL(Q_far||P) = {kl(Q_far, P):.4f} nats")

def demo_cross_entropy():
    print("\n=== Cross-Entropy Loss ===")
    # H(P, Q) = H(P) + KL(P||Q)
    # For one-hot true label: H(y, y_hat) = -log(y_hat[true_class])
    def cross_entropy(y_true, y_pred):
        return -math.log(y_pred[y_true] + 1e-15)

    def softmax(logits):
        e = np.exp(logits - np.max(logits))
        return e / e.sum()

    # Confident and correct
    logits_good = np.array([5.0, 1.0, 0.5])
    logits_bad  = np.array([0.5, 0.5, 0.5])
    for name, logits in [("Confident correct", logits_good), ("Uncertain", logits_bad)]:
        probs = softmax(logits)
        ce = cross_entropy(0, probs)   # true class = 0
        print(f"  {name}: probs={probs.round(3)}  CE={ce:.4f}")

def demo_mutual_information():
    print("\n=== Mutual Information ===")
    np.random.seed(42)
    n = 10_000
    # Strong dependence: X -> Y
    X = np.random.binomial(1, 0.5, n)
    Y_dep = np.where(X==1, np.random.binomial(1, 0.9, n), np.random.binomial(1, 0.1, n))
    Y_ind = np.random.binomial(1, 0.5, n)

    def mi_binary(X, Y):
        # I(X;Y) = H(Y) - H(Y|X)
        h_y = safe_entropy([np.mean(Y), 1-np.mean(Y)])
        h_y_given_x = (np.mean(X==0) * safe_entropy([np.mean(Y[X==0]+1e-9), np.mean(1-Y[X==0]+1e-9)]) +
                       np.mean(X==1) * safe_entropy([np.mean(Y[X==1]+1e-9), np.mean(1-Y[X==1]+1e-9)]))
        return max(0, h_y - h_y_given_x)

    print(f"  MI(X, Y_dependent)  = {mi_binary(X, Y_dep):.4f} bits")
    print(f"  MI(X, Y_independent) = {mi_binary(X, Y_ind):.4f} bits  (≈0 expected)")

if __name__ == "__main__":
    demo_entropy()
    demo_kl_divergence()
    demo_cross_entropy()
    demo_mutual_information()
