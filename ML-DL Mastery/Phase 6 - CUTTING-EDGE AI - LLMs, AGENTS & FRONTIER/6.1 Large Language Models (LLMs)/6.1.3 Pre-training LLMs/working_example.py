"""
Working Example: Pre-training LLMs
Covers training objectives, data curation, scaling laws,
distributed training strategies, and compute budgets.
"""
import numpy as np
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_pretraining")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -- 1. Pre-training objectives ------------------------------------------------
def training_objectives():
    print("=== LLM Pre-training Objectives ===")
    print()
    objectives = [
        ("Causal LM (CLM)",    "Predict next token; GPT family; decoder-only"),
        ("Masked LM (MLM)",    "Predict masked tokens; BERT; bidirectional"),
        ("Prefix LM",          "Prefix visible, suffix predicted; T5, PaLM"),
        ("Span corruption",    "Replace spans with single sentinel; T5"),
        ("UL2",                "Mix of CLM, MLM, and span corruption; Flan-T5"),
    ]
    print(f"  {'Objective':<22} {'Notes'}")
    for o, d in objectives:
        print(f"  {o:<22} {d}")

    print()
    print("  CLM cross-entropy loss:")
    print("    L = -1/T Sigma_t log P(x_t | x_<t)")
    print()

    # Simulate a training loss curve
    rng = np.random.default_rng(0)
    print("  Simulated training loss (tokens seen vs perplexity):")
    tokens_seen = [1e9 * (2**i) for i in range(8)]
    for t in tokens_seen:
        # Rough Chinchilla-inspired loss curve
        perp = 200 * (t / 1e9)**(-0.076) + rng.normal(0, 0.5)
        print(f"  {t/1e9:>8.1f}B tokens: perplexity ~= {perp:.1f}")


# -- 2. Scaling laws -----------------------------------------------------------
def scaling_laws():
    print("\n=== Scaling Laws (Chinchilla) ===")
    print()
    print("  Kaplan et al. (2020) OpenAI scaling laws:")
    print("    L(N) = (N_c / N)^alpha  + L_inf   (model size)")
    print("    L(D) = (D_c / D)^beta  + L_inf   (dataset size)")
    print("    alpha ~= 0.076, beta ~= 0.095, L_inf ~= 1.69 (irreducible entropy)")
    print()
    print("  Hoffmann et al. (2022) Chinchilla — REVISED laws:")
    print("    Optimal: N_tokens ~= 20 x N_params")
    print("    Previous GPT-3 models were undertrained!")
    print()
    print("  Compute-optimal frontier:")
    print(f"  {'Model':<16} {'Params':<10} {'Tokens':<12} {'FLOPs'}")
    models = [
        ("Chinchilla",  "70B",  "1.4T",  "5.7×10^23"),
        ("LLaMA-3-8B",  "8B",   "15T",   "~5.5×10^23"),
        ("LLaMA-3-70B", "70B",  "15T",   ">10^24"),
        ("GPT-3",       "175B", "300B",  "3.1×10^23"),
        ("Llama-3.1-405B","405B","~15T", "~3.8×10^25"),
    ]
    for m, p, tok, fl in models:
        print(f"  {m:<16} {p:<10} {tok:<12} {fl}")
    print()
    print("  FLOPs for training ~= 6 x N x D  (N=params, D=tokens)")
    N = 7e9; D = 1e12
    flops = 6 * N * D
    print(f"    7B model × 1T tokens: {flops:.2e} FLOPs")
    print(f"    On 1000× A100 (312 TFLOP/s), 50% MFU:")
    hours = flops / (1000 * 312e12 * 0.5) / 3600
    print(f"    Training time ~= {hours:.0f} hours = {hours/24:.0f} days")


# -- 3. Data curation ----------------------------------------------------------
def data_curation():
    print("\n=== Training Data Curation ===")
    print()
    sources = [
        ("Common Crawl",   "Web; raw HTML -> text; ~petabytes; majority of tokens"),
        ("C4",             "Cleaned CC; English; 305GB; T5"),
        ("The Pile",       "825GB; 22 diverse sources; EleutherAI"),
        ("RedPajama",      "Reproduction of LLaMA training data; open"),
        ("FineWeb",        "5T+ token web corpus; HuggingFace; quality filtered"),
        ("Dolma",          "3T tokens; OLMo; permissive license"),
        ("DCLM",           "10T tokens; DataComp for LLMs; best CC filtering"),
    ]
    print(f"  {'Source':<18} {'Notes'}")
    for s, d in sources:
        print(f"  {s:<18} {d}")
    print()
    print("  Typical token mix (approximate, varies by model):")
    mix = [
        ("Web text",     65),
        ("Books",        15),
        ("Code",         10),
        ("Wikipedia",     5),
        ("Papers",        3),
        ("Other",         2),
    ]
    for domain, pct in mix:
        bar = "#" * (pct // 3)
        print(f"  {domain:<14} {pct:>3}% {bar}")
    print()
    print("  Data quality steps:")
    steps = [
        "Language identification (fasttext)",
        "Quality filtering (perplexity, heuristics)",
        "Deduplication (MinHash, suffix arrays)",
        "PII removal (regex + NER)",
        "Toxicity filtering",
        "Domain up-sampling (code, books)",
    ]
    for s in steps:
        print(f"  • {s}")


# -- 4. Distributed training ---------------------------------------------------
def distributed_training():
    print("\n=== Distributed Training Strategies ===")
    print()
    strategies = [
        ("Data Parallelism",     "Each GPU has full model copy; different data batches; all-reduce grads"),
        ("Tensor Parallelism",   "Split weight matrices across GPUs; Megatron-LM; intra-layer"),
        ("Pipeline Parallelism", "Split layers across GPUs; micro-batching; GPipe/PipeDream"),
        ("Sequence Parallelism", "Split long sequences across GPUs; ring attention"),
        ("ZeRO (DeepSpeed)",     "Partition params/grads/optimiser states; ZeRO-1/2/3"),
        ("FSDP (PyTorch)",       "Fully Sharded Data Parallel; built into PyTorch 2.0"),
    ]
    print(f"  {'Strategy':<24} {'Notes'}")
    for s, d in strategies:
        print(f"  {s:<24} {d}")
    print()
    print("  3D parallelism (used for very large models):")
    print("    TP × PP × DP  e.g. 8 × 4 × 8 = 256 GPUs")
    print("    Megatron-Turing NLG (530B): 280 A100s, 3 weeks training")
    print()
    print("  Mixed precision:")
    print("    BF16 (Brain Float): same range as FP32; better than FP16; A100+")
    print("    FP8 (H100+): 2× faster than BF16; careful gradient scaling")
    print()
    print("  Communication bottleneck:")
    print("    NVLink: 600 GB/s (within node)")
    print("    InfiniBand: 400 Gbps (across nodes)")
    print("    Ring all-reduce time: O(2(N-1)/N × data)")


if __name__ == "__main__":
    training_objectives()
    scaling_laws()
    data_curation()
    distributed_training()
