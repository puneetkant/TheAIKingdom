"""
Working Example: Context Length Extension
Covers RoPE scaling, ALiBi, long-context training, and chunked attention.
"""
import numpy as np
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_context_ext")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── 1. The long-context problem ───────────────────────────────────────────────
def long_context_problem():
    print("=== Context Length Extension ===")
    print()
    print("  Why long context matters:")
    print("    • Code repos (millions of tokens)")
    print("    • Multi-turn conversations")
    print("    • Entire books / legal documents")
    print("    • RAG without chunking (just pass the whole doc)")
    print()
    print("  Challenges:")
    challenges = [
        ("Quadratic attention",  "O(T²) compute and memory for standard attention"),
        ("Positional OOD",       "RoPE/absolute PE trained at 4k; breaks at 128k"),
        ("Lost in the middle",   "Models attend well to start/end, poorly to middle"),
        ("Memory",               "KV cache grows linearly with context length"),
        ("Long-context data",    "Scarce long documents; hard to fine-tune effectively"),
    ]
    for c, d in challenges:
        print(f"  {c:<22} {d}")


# ── 2. RoPE and its extensions ────────────────────────────────────────────────
def rope_extensions():
    print("\n=== RoPE and Context Extension ===")
    print()
    print("  Rotary Position Embedding (RoPE):")
    print("    Multiply Q, K by complex rotation: e^{iθm}  where θ_i = base^{-2i/d}")
    print("    Relative position encoded in dot product: <q_m, k_n> = f(m-n)")
    print()

    def rope_angle(pos, dim_idx, d_model, base=10000):
        theta = base ** (-2 * dim_idx / d_model)
        return pos * theta

    d = 8; T = 6
    print("  RoPE angles for first 4 dimensions at positions 0,1,2,3:")
    for t in range(4):
        angles = [rope_angle(t, i, d) for i in range(4)]
        print(f"  pos={t}: {[round(a, 4) for a in angles]}")

    print()
    print("  Context extension methods for RoPE:")
    methods = [
        ("Position interpolation", "Linear scale positions: m → m × (L_orig/L_new); Meta"),
        ("YaRN",                   "Non-uniform scaling; scale high-freq less; LLaMA-3"),
        ("LongRoPE",               "Search optimal per-dimension scale; 2M tokens"),
        ("NTK-aware scaling",      "Change RoPE base to extrapolate naturally"),
        ("Dynamic NTK",            "Adjust scale per batch based on actual seq length"),
        ("Code Llama (PI)",        "4k → 100k via position interpolation + fine-tuning"),
    ]
    for m, d in methods:
        print(f"  {m:<26} {d}")

    print()
    print("  Example: YaRN NTK scaling formula")
    print("    new_base = old_base × (new_len / old_len)^(d/(d-2))")
    old_base = 10_000; old_len = 4_096; new_len = 128_000; d = 128
    new_base = old_base * (new_len / old_len) ** (d / (d - 2))
    print(f"    base: {old_base} → {new_base:.0f}")


# ── 3. Efficient long-context attention ───────────────────────────────────────
def efficient_attention():
    print("\n=== Efficient Long-Context Attention ===")
    print()
    methods = [
        ("FlashAttention-2",  "IO-aware; block-sparse; 2-8× faster; same output"),
        ("FlashAttention-3",  "H100-specific; FP8; overlapped pipeline; best"),
        ("Sliding window",    "Each token attends only to W neighbours; O(T·W)"),
        ("Longformer",        "Local + global attention; linear complexity"),
        ("BigBird",           "Sparse = random + local + global; O(T) attention"),
        ("Ring attention",    "Sequence split across GPUs; no communication bottleneck"),
        ("Linear attention",  "Kernel trick; O(T); approximation quality varies"),
        ("Mamba/SSM",         "State space model; O(T); no attention at all"),
    ]
    print(f"  {'Method':<20} {'Notes'}")
    for m, d in methods:
        print(f"  {m:<20} {d}")

    print()
    print("  Memory comparison for T tokens, d=4096, 32 heads:")
    for T in [4096, 32768, 131072, 1_000_000]:
        # Standard attention: T × T × heads × bytes
        std_gb = T * T * 32 * 2 / 1e9
        # Flash attention: T × d × bytes (just the output buffer)
        flash_gb = T * 4096 * 2 / 1e9
        print(f"  T={T:>9}: standard={std_gb:.1f}GB  flashattn={flash_gb:.3f}GB")


# ── 4. Long-context models ────────────────────────────────────────────────────
def long_context_models():
    print("\n=== Long-Context Model Landscape ===")
    print()
    models = [
        ("LLaMA-3.1-8/70/405B",  "128k",   "YaRN + 800B long-context tokens"),
        ("Mistral-Nemo 12B",      "128k",   "Tekken tokeniser; sliding window"),
        ("Claude-3.5 Sonnet",     "200k",   "Best long-doc QA; document analysis"),
        ("GPT-4o",                "128k",   "Strong retrieval within long context"),
        ("Gemini 1.5 Pro",        "1M+",    "Video, audio, code; best recall"),
        ("MegaLong (research)",   "10M+",   "Ring attention; sequence parallel"),
    ]
    print(f"  {'Model':<26} {'Context':<10} {'Notes'}")
    for m, ctx, d in models:
        print(f"  {m:<26} {ctx:<10} {d}")

    print()
    print("  Long-context evaluation benchmarks:")
    benchmarks = [
        ("RULER",       "Needle-in-haystack; multi-key; ruler for recall"),
        ("L-Eval",      "256k tokens; summarisation; QA; evidence retrieval"),
        ("ZeroSCROLLS", "Long summarisation; QA; 100k+ token inputs"),
        ("HELMET",      "Hard long-context evaluation; comprehensive"),
    ]
    for b, d in benchmarks:
        print(f"  {b:<14} {d}")


if __name__ == "__main__":
    long_context_problem()
    rope_extensions()
    efficient_attention()
    long_context_models()
