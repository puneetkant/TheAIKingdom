"""
Working Example 2: Efficient Attention Variants — linear attention, local window, comparison
==============================================================================================
Complexity comparison: O(n²) vs O(n) for various sequence lengths.

Run:  python working_example2.py
"""
from pathlib import Path
try:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import time
except ImportError:
    raise SystemExit("pip install numpy matplotlib")

OUTPUT = Path(__file__).parent / "output"
OUTPUT.mkdir(exist_ok=True)

def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True); e = np.exp(x); return e / e.sum(axis=axis, keepdims=True)

def full_attention(Q, K, V):
    """Standard O(n²) attention."""
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    return softmax(scores) @ V

def linear_attention(Q, K, V):
    """Linear (kernel) attention: O(n) via associativity trick. Feature map = elu+1."""
    Q = np.maximum(Q + 1, 1e-6); K = np.maximum(K + 1, 1e-6)
    KV = K.T @ V                 # (d_k, d_v) — independent of n
    Z  = Q @ K.sum(axis=0, keepdims=True).T  # normaliser
    return Q @ KV / (Z + 1e-8)

def local_window_attention(Q, K, V, window=4):
    """O(n·w) sliding window attention."""
    seq, d_k = Q.shape; d_v = V.shape[1]
    out = np.zeros((seq, d_v))
    for i in range(seq):
        start = max(0, i - window//2); end = min(seq, i + window//2 + 1)
        scores = Q[i:i+1] @ K[start:end].T / np.sqrt(d_k)
        w = softmax(scores); out[i] = w @ V[start:end]
    return out

def demo():
    print("=== Efficient Attention Variants ===")
    variants = {"Full O(n²)": full_attention, "Linear O(n)": linear_attention,
                "Window O(n·w)": local_window_attention}

    seq_lens = [16, 32, 64, 128]
    times = {name: [] for name in variants}

    rng = np.random.default_rng(42)
    for seq in seq_lens:
        Q = rng.standard_normal((seq, 8)); K = rng.standard_normal((seq, 8)); V = rng.standard_normal((seq, 8))
        for name, fn in variants.items():
            t0 = time.perf_counter(); fn(Q, K, V); times[name].append(time.perf_counter() - t0)
        print(f"  seq={seq:4d}: " + "  ".join(f"{n[:6]}={times[n][-1]*1000:.2f}ms" for n in variants))

    fig, ax = plt.subplots(figsize=(8, 4))
    for name, ts in times.items():
        ax.plot(seq_lens, ts, marker="o", label=name)
    ax.set_xlabel("Sequence Length"); ax.set_ylabel("Time (s)"); ax.set_title("Attention Variants Scaling")
    ax.legend(); plt.tight_layout(); plt.savefig(OUTPUT / "attention_variants.png"); plt.close()
    print("  Saved attention_variants.png")

def demo_causal_local_window():
    """Causal variant of local window attention (for autoregressive models)."""
    print("\n=== Causal Local Window Attention ===")
    rng = np.random.default_rng(42)
    seq, d_k, d_v, window = 10, 8, 8, 4
    Q = rng.standard_normal((seq, d_k))
    K = rng.standard_normal((seq, d_k))
    V = rng.standard_normal((seq, d_v))

    out = np.zeros((seq, d_v))
    for i in range(seq):
        # Causal: only look at past tokens within window
        start = max(0, i - window + 1)
        end   = i + 1   # can't attend to future
        scores = Q[i:i+1] @ K[start:end].T / np.sqrt(d_k)
        w = softmax(scores)
        out[i] = w @ V[start:end]
    print(f"  Causal window output shape: {out.shape}")
    print(f"  Token 0 attends to 1 token, token 5 attends to {min(5, window)} tokens")


def demo_complexity_summary():
    """Print attention complexity comparison table."""
    print("\n=== Attention Complexity Summary ===")
    variants = [
        ("Full (Vanilla)",     "O(n²d)",  "O(n²)",  "All tokens"),
        ("Linear",             "O(nd²)",  "O(nd)",  "All (approx)"),
        ("Local Window",       "O(n·w·d)","O(n·w)", "Nearby tokens"),
        ("Sparse (Longformer)","O(n·w)",  "O(n·w)", "Local + global"),
        ("Random (BigBird)",   "O(n·r)",  "O(n·r)", "Random + local + global"),
    ]
    print(f"  {'Variant':25s}  {'Compute':12s}  {'Memory':10s}  {'Coverage'}")
    for name, comp, mem, cov in variants:
        print(f"  {name:25s}  {comp:12s}  {mem:10s}  {cov}")


if __name__ == "__main__":
    demo()
    demo_causal_local_window()
    demo_complexity_summary()
