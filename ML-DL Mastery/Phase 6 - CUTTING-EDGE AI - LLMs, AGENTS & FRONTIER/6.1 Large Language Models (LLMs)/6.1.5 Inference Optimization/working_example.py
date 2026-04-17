"""
Working Example: LLM Inference Optimisation
Covers KV cache, speculative decoding, quantisation,
continuous batching, and throughput/latency trade-offs.
"""
import numpy as np
import os, time

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_inference_opt")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── 1. KV cache ───────────────────────────────────────────────────────────────
def kv_cache():
    print("=== KV Cache Optimisation ===")
    print()
    print("  Without KV cache: recompute all K,V for every new token → O(T²)")
    print("  With KV cache:    store K,V of past tokens → O(T) per new token")
    print()
    print("  Memory usage (LLaMA-3-70B, BF16):")
    print("    Layers=80, KV heads=8 (GQA), head_dim=128")
    n_layers = 80; n_kv_heads = 8; head_dim = 128; bytes_per_elem = 2
    for seq_len in [1_000, 8_000, 32_000, 128_000]:
        mem = 2 * n_layers * n_kv_heads * head_dim * seq_len * bytes_per_elem / 1e9
        print(f"    seq_len={seq_len:>7}: KV cache = {mem:.2f} GB")
    print()
    print("  KV cache optimisations:")
    opts = [
        ("Paged attention (vLLM)", "Allocate KV in fixed pages; handle variable lengths"),
        ("Prefix caching",         "Reuse KV for shared system prompt across requests"),
        ("Sliding window",         "Only cache last W tokens; Mistral; O(W) memory"),
        ("Multi-query (MQA)",      "Share 1 K,V head across all query heads; 8× less KV"),
        ("Grouped-query (GQA)",    "G shared K,V heads; balance memory vs accuracy"),
        ("KV quantisation",        "INT8 KV cache; 2× memory reduction; small quality hit"),
        ("StreamingLLM",           "Attention sinks + sliding window; ∞ context"),
    ]
    for o, d in opts:
        print(f"  {o:<28} {d}")


# ── 2. Speculative decoding ───────────────────────────────────────────────────
def speculative_decoding():
    print("\n=== Speculative Decoding ===")
    print()
    print("  Problem: LLM generation is memory-bandwidth-bound, not compute-bound")
    print("           Most of the time is spent on loading model weights")
    print()
    print("  Speculative decoding (Leviathan et al. 2023):")
    print("    1. Small draft model generates K tokens (cheap)")
    print("    2. Target model verifies all K+1 tokens in 1 forward pass")
    print("    3. Accept tokens up to first disagreement; resample tail")
    print()
    print("  Key properties:")
    print("    - Identical output distribution to target model")
    print("    - 2-3× faster empirically on typical text")
    print("    - Overhead is near zero when draft model is good")
    print()

    # Simulate acceptance rates
    rng = np.random.default_rng(0)
    p_accept = [0.85, 0.78, 0.70, 0.63]   # acceptance at each draft position
    K = len(p_accept)
    total_trials = 10_000

    print(f"  Simulated acceptance at each draft position:")
    accepted_counts = []
    for k, p in enumerate(p_accept):
        accepted = int(rng.binomial(total_trials, p))
        accepted_counts.append(accepted / total_trials)
        print(f"    Position {k+1}: p_accept = {p:.2f}, simulated = {accepted/total_trials:.3f}")

    E_tokens = 1.0
    for p in p_accept:
        E_tokens += p
    print(f"\n  Expected tokens accepted per target call: {E_tokens:.2f}")
    print(f"  Effective speedup ≈ {E_tokens:.2f}× (ignoring draft model cost)")
    print()
    print("  Variants:")
    variants = [
        ("SpecInfer",     "Tree of draft tokens; max acceptance"),
        ("Lookahead",     "Parallel draft from Jacobi iterations; no small model"),
        ("Medusa",        "Multiple prediction heads on target model"),
        ("EAGLE-2",       "Feature-level draft model; 3-4× speedup"),
    ]
    for v, d in variants:
        print(f"  {v:<16} {d}")


# ── 3. Quantisation ───────────────────────────────────────────────────────────
def quantisation():
    print("\n=== LLM Quantisation ===")
    print()
    print("  Why quantise: LLaMA-3-70B in BF16 = 140GB; in INT4 = ~35GB")
    print()
    print("  Post-Training Quantisation (PTQ) methods:")
    methods = [
        ("GPTQ",     "INT4/INT8; layer-wise quantisation; ~perplexity +0.1"),
        ("AWQ",      "Activation-aware weight quantisation; search scale per channel"),
        ("SmoothQuant","Migrate quantisation difficulty from activations to weights"),
        ("GGUF/llama.cpp","CPU-friendly; 2-8 bit; many quant types; iMatrix"),
        ("bitsandbytes","4-bit NF4/FP4; LLM.int8(); QLoRA training"),
        ("SpQR",     "Mixed-precision; sensitive weights in FP16; rest INT4"),
    ]
    for m, d in methods:
        print(f"  {m:<16} {d}")
    print()

    # Memory and speed estimates
    print("  LLaMA-3-70B memory requirements:")
    formats = [
        ("BF16",   2,   1.00),
        ("INT8",   1,   0.95),
        ("INT4",   0.5, 0.88),
        ("INT2",   0.25,0.72),
    ]
    base_gb = 140
    print(f"  {'Format':<8} {'Size (GB)':<12} {'Relative quality'}")
    for fmt, mult, qual in formats:
        gb = base_gb * mult
        print(f"  {fmt:<8} {gb:<12.0f} {qual:.2f}")

    print()
    print("  Quantisation-Aware Training (QAT):")
    print("    Simulate quantisation during training → better accuracy")
    print("    Expensive but best quality; used in Phi-1, Phi-2 (partially)")


# ── 4. Continuous batching and vLLM ──────────────────────────────────────────
def continuous_batching():
    print("\n=== Continuous Batching (vLLM) ===")
    print()
    print("  Traditional serving: static batch → wait for all to finish")
    print("  Continuous batching: add new requests as slots free → higher throughput")
    print()
    print("  PagedAttention (vLLM, Kwon et al. 2023):")
    print("    - KV cache managed as pages (like OS virtual memory)")
    print("    - No internal fragmentation; share prefix pages")
    print("    - 2-4× more requests served simultaneously")
    print()
    print("  Serving frameworks comparison:")
    serving = [
        ("vLLM",         "PagedAttention; fastest OSS; continuous batching"),
        ("TGI (HF)",     "Text Generation Inference; Flash Attention; tensor parallel"),
        ("llama.cpp",    "CPU/Apple Silicon; GGUF; no GPU required"),
        ("Triton + TRT", "NVIDIA; FP8; maximum GPU utilisation"),
        ("SGLang",       "Structured generation; RadixAttention; cache reuse"),
        ("Ollama",       "Local run; llama.cpp backend; easy API"),
    ]
    for s, d in serving:
        print(f"  {s:<16} {d}")

    print()
    print("  Throughput vs latency trade-off:")
    print("    Small batch: low latency, low throughput")
    print("    Large batch: higher latency, high throughput")
    print("    Sweet spot: continuous batching with max seq budget")


if __name__ == "__main__":
    kv_cache()
    speculative_decoding()
    quantisation()
    continuous_batching()
