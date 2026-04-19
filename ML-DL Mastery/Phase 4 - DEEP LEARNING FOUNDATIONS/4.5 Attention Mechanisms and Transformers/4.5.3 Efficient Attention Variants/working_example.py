"""
Working Example: Efficient Attention Variants
Covers sparse attention, linear attention, Linformer, Performer,
Longformer, BigBird, Flash Attention concepts, and complexity analysis.
"""
import numpy as np
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_efficient_attn")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def softmax(z, axis=-1):
    e = np.exp(z - z.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


# -- 1. The quadratic bottleneck -----------------------------------------------
def quadratic_bottleneck():
    print("=== The Quadratic Bottleneck of Full Self-Attention ===")
    print("  Full attention: O(T²·d) time, O(T²) memory for the attention matrix")
    print()
    print(f"  {'Seq length T':<16} {'Attention matrix':<20} {'Memory (float32)'}")
    for T in [128, 512, 1024, 4096, 16384, 65536]:
        elems  = T * T
        mem_mb = elems * 4 / 1024**2
        print(f"  {T:<16} {str(T)+'×'+str(T):<20} {mem_mb:>8.1f} MB")
    print()
    print("  At T=65536 (book-length), attention matrix = 16GB — impractical!")
    print("  Solution: approximate or restructure the attention computation")


# -- 2. Local / Sliding window attention --------------------------------------
def local_window_attention(X, W_Q, W_K, W_V, window_size=3):
    """Each token attends to at most 2w+1 tokens (window_size each side)."""
    T, d = X.shape
    Q = X @ W_Q; K = X @ W_K; V = X @ W_V
    d_k = Q.shape[-1]
    out = np.zeros_like(Q)
    for i in range(T):
        lo = max(0, i - window_size)
        hi = min(T, i + window_size + 1)
        q  = Q[i:i+1]           # (1, d_k)
        k  = K[lo:hi]           # (w, d_k)
        v  = V[lo:hi]           # (w, d_v)
        scores = q @ k.T / np.sqrt(d_k)   # (1, w)
        alpha  = softmax(scores, axis=-1)
        out[i] = (alpha @ v)[0]
    return out

def local_attention_demo():
    print("\n=== Local (Sliding Window) Attention ===")
    print("  Each token attends to at most w neighbours")
    print("  Complexity: O(T·w·d) — linear in T if w is fixed")
    print("  Used in: Longformer, BigBird, Mistral (SWA)")

    rng = np.random.default_rng(1)
    T, d, dk = 12, 16, 8
    X   = rng.standard_normal((T, d))
    W_Q = rng.standard_normal((d, dk)) * 0.1
    W_K = rng.standard_normal((d, dk)) * 0.1
    W_V = rng.standard_normal((d, dk)) * 0.1

    for w in [1, 2, 4, T//2]:
        out = local_window_attention(X, W_Q, W_K, W_V, window_size=w)
        print(f"  window_size={w:<4}: each token attends {min(2*w+1, T)} tokens  out={out.shape}")


# -- 3. Global tokens (Longformer) ---------------------------------------------
def global_tokens_demo():
    print("\n=== Global Tokens (Longformer / BigBird) ===")
    print("  Mix local window + a few global tokens (e.g. [CLS])")
    print("  Global tokens attend to ALL other tokens (and vice versa)")
    print("  Total complexity: O(T·w + T·g) where g = n_global_tokens ≪ T")
    print()
    T, w, g = 1024, 64, 4
    full_attn = T * T
    local_attn = T * (2*w + 1) + g * T
    print(f"  T={T}, window={w}, global_tokens={g}")
    print(f"  Full attention:    {full_attn:,} ops")
    print(f"  Longformer:        {local_attn:,} ops  ({local_attn/full_attn*100:.1f}%)")


# -- 4. Random / sparse attention ----------------------------------------------
def sparse_attention_demo():
    print("\n=== Sparse Attention (BigBird, Routing Transformer) ===")
    print("  Attention matrix is sparse: only attend to selected positions")
    print()
    print("  BigBird patterns:")
    print("    1. Local window: w tokens left/right")
    print("    2. Random:       r random tokens per position")
    print("    3. Global:       g global tokens (e.g. [CLS])")
    print()

    T, w, r, g = 512, 3, 2, 1
    total_ops = T * (2*w + 1 + r + g)
    print(f"  T={T}, w={w}, r={r}, g={g}")
    print(f"  Ops per layer: {total_ops:,}  (full attention: {T*T:,})")
    print(f"  Reduction: {total_ops/T/T*100:.1f}% of full")


# -- 5. Linear attention (Performer) ------------------------------------------
def linear_attention_demo():
    print("\n=== Linear Attention (Performer, Linformer) ===")
    print("  Standard: Attn = softmax(QK^T/sqrtd) · V   O(T²)")
    print()
    print("  Key insight: approximate softmax(QK^T) using kernel decomposition")
    print("  FAVOR+ (Performer): phi(q)·phi(k)^T ~= exp(q^T k / sqrtd)")
    print("  -> Compute (Sigma phi(k_i)^T V_i) first, then multiply by phi(q)")
    print("  -> O(T·r) where r = random feature dimension ≪ T")
    print()

    rng = np.random.default_rng(2)
    T, d, r_feat = 200, 32, 64

    Q = rng.standard_normal((T, d))
    K = rng.standard_normal((T, d))
    V = rng.standard_normal((T, d))

    # Random feature map: phi(x) = exp(Wx) / sqrtr   (simplified FAVOR)
    Omega = rng.standard_normal((d, r_feat)) / np.sqrt(d)
    def phi(X):
        return np.exp(X @ Omega) / np.sqrt(r_feat)

    phi_Q = phi(Q)   # (T, r_feat)
    phi_K = phi(K)   # (T, r_feat)

    # Linear attention: O(T·r) instead of O(T²)
    KV = phi_K.T @ V        # (r_feat, d)
    Zk = phi_K.sum(axis=0)  # (r_feat,) normalisation
    out_linear = phi_Q @ KV / (phi_Q @ Zk).reshape(-1, 1)

    # Exact attention (for comparison)
    scores = Q @ K.T / np.sqrt(d)
    alpha  = softmax(scores, axis=-1)
    out_exact  = alpha @ V

    # Compare
    err = np.mean(np.abs(out_linear - out_exact))
    print(f"  T={T}, d={d}, r_feat={r_feat}")
    print(f"  Mean absolute error vs exact attention: {err:.4f}")
    print(f"  (Approx quality improves with larger r_feat)")


# -- 6. Linformer -------------------------------------------------------------
def linformer_demo():
    print("\n=== Linformer (Wang et al., 2020) ===")
    print("  Observation: attention matrices have low-rank structure")
    print("  Project keys and values from T -> k  (k ≪ T)")
    print("  Attn ~= softmax(Q · (E·K)^T / sqrtd) · (F·V)")
    print("  Complexity: O(T·k·d) — linear in T if k is fixed")
    print()

    rng = np.random.default_rng(3)
    T, d, k_rank = 256, 64, 32
    Q = rng.standard_normal((T, d))
    K = rng.standard_normal((T, d))
    V = rng.standard_normal((T, d))
    E = rng.standard_normal((k_rank, T)) * 0.01  # projection matrix
    F = rng.standard_normal((k_rank, T)) * 0.01

    K_proj = E @ K   # (k, d)
    V_proj = F @ V   # (k, d)

    scores = Q @ K_proj.T / np.sqrt(d)   # (T, k)
    alpha  = softmax(scores, axis=-1)     # (T, k)
    out    = alpha @ V_proj               # (T, d)

    print(f"  T={T}, d={d}, projection_rank={k_rank}")
    print(f"  K projected: {K.shape} -> {K_proj.shape}")
    print(f"  Attention map: (T, k) = ({T}, {k_rank}) instead of ({T}, {T})")
    print(f"  Output: {out.shape}")
    print(f"  Memory reduction: {k_rank}/{T} = {k_rank/T*100:.0f}% of full attention")


# -- 7. Flash Attention -------------------------------------------------------
def flash_attention_concept():
    print("\n=== Flash Attention (Dao et al., 2022) ===")
    print("  NOT an approximation — computes exact attention!")
    print("  Key insight: reorder computation to avoid materialising full T×T matrix")
    print()
    print("  Algorithm:")
    print("    Process Q in blocks; for each block:")
    print("      1. Compute QK^T block by block (fits in fast SRAM)")
    print("      2. Use online softmax (numerically stable, incremental)")
    print("      3. Accumulate output — never store full attention map")
    print()
    print("  Result:")
    print("    Memory: O(T) instead of O(T²)")
    print("    Speed:  2-4× faster than standard PyTorch attention")
    print("    Quality: EXACT (no approximation)")
    print("    Supported: A100, H100 GPUs; PyTorch 2.0+ via F.scaled_dot_product_attention")


# -- 8. Complexity summary -----------------------------------------------------
def complexity_summary():
    print("\n=== Efficient Attention Variants — Complexity Summary ===")
    print(f"  {'Method':<24} {'Time':<22} {'Memory':<20} {'Exact?'}")
    rows = [
        ("Full Attention",       "O(T²·d)",        "O(T²)",         "Yes"),
        ("Local Window",         "O(T·w·d)",       "O(T·w)",        "Yes (local)"),
        ("Longformer",           "O(T·w + T·g)",   "O(T·w)",        "Yes (hybrid)"),
        ("BigBird",              "O(T·(w+r+g))",   "O(T·(w+r+g))", "Yes (sparse)"),
        ("Linformer",            "O(T·k·d)",       "O(T·k)",        "No (approx)"),
        ("Performer (FAVOR+)",   "O(T·r·d)",       "O(T·r)",        "No (approx)"),
        ("Flash Attention",      "O(T²·d)",        "O(T)",          "Yes (exact)"),
        ("Multi-Query Attn",     "O(T²·d)",        "O(T²/h)",       "Yes"),
    ]
    for name, time, mem, exact in rows:
        print(f"  {name:<24} {time:<22} {mem:<20} {exact}")
    print()
    print("  w=window, r=random features, k=projection rank, g=global tokens, h=heads")


if __name__ == "__main__":
    quadratic_bottleneck()
    local_attention_demo()
    global_tokens_demo()
    sparse_attention_demo()
    linear_attention_demo()
    linformer_demo()
    flash_attention_concept()
    complexity_summary()
