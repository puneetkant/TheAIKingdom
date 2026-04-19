"""
Working Example: Parameter-Efficient Fine-Tuning (PEFT)
Covers LoRA, QLoRA, Prefix Tuning, Prompt Tuning, and IA3.
"""
import numpy as np
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_peft")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -- 1. PEFT overview ----------------------------------------------------------
def peft_overview():
    print("=== Parameter-Efficient Fine-Tuning (PEFT) ===")
    print()
    print("  Full fine-tuning: update all N parameters -> expensive")
    print("  PEFT: train only a small subset / add tiny adapters")
    print()
    methods = [
        ("LoRA",          "Low-Rank Adapters; decompose DeltaW into AB; most popular"),
        ("QLoRA",         "LoRA + 4-bit quantised base; fits 70B on 48GB GPU"),
        ("Prefix Tuning", "Learn virtual token prefixes; prepend to K,V"),
        ("Prompt Tuning", "Learn soft prompt embeddings; very few params"),
        ("IA3",           "Learn scale vectors for K,V,FFN; few params; fast"),
        ("Adapter",       "Small bottleneck layers inserted in each transformer block"),
        ("BitFit",        "Only fine-tune bias terms; minimal params"),
    ]
    print(f"  {'Method':<16} {'Description'}")
    for m, d in methods:
        print(f"  {m:<16} {d}")


# -- 2. LoRA deep dive ---------------------------------------------------------
def lora_demo():
    print("\n=== LoRA (Low-Rank Adaptation) ===")
    print()
    print("  W_new = W_0 + DeltaW = W_0 + alpha/r · B·A")
    print("  A in ℝ^{rxd_in}, B in ℝ^{d_outxr}, B init to 0, A ~ N(0,sigma²)")
    print("  Only A, B trained; W_0 frozen; merged at inference")
    print()

    rng = np.random.default_rng(0)
    d_in = 64; d_out = 64; r = 8; alpha = 16.0

    W_0 = rng.normal(0, 0.02, (d_out, d_in))   # frozen
    A   = rng.normal(0, 1/r, (r, d_in))         # trainable
    B   = np.zeros((d_out, r))                  # trainable (init 0)

    # Simulate one forward pass
    x = rng.normal(0, 1, (16, d_in))   # batch of 16
    out_0   = x @ W_0.T                        # base model
    delta_W = (alpha / r) * (B @ A)            # LoRA adaptor
    out_lora = x @ (W_0 + delta_W).T           # fine-tuned model

    print(f"  d_in={d_in}, d_out={d_out}, rank r={r}, alpha={alpha}")
    trainable_params = r*d_in + d_out*r
    total_params = d_in * d_out
    print(f"  Total params in W: {total_params:,}")
    print(f"  LoRA trainable:    {trainable_params:,}")
    print(f"  Reduction factor:  {total_params/trainable_params:.1f}×")
    print()
    print(f"  At init (B=0): out_lora == out_0: {np.allclose(out_lora, out_0)}")

    # After some training (simulate)
    B_trained = rng.normal(0, 0.01, (d_out, r))
    delta_W_trained = (alpha / r) * (B_trained @ A)
    out_trained = x @ (W_0 + delta_W_trained).T
    diff = np.abs(out_trained - out_0).mean()
    print(f"  After training: mean output diff = {diff:.4f}")

    print()
    print("  LoRA rank selection guide:")
    ranks = [
        (4,    "Very few params; for style/format changes"),
        (8,    "Good balance; most common choice"),
        (16,   "Higher capacity; complex tasks"),
        (32,   "Full-rank for small models"),
        (64,   "Near full-rank fine-tuning"),
    ]
    for r_val, d in ranks:
        print(f"  r={r_val:<4} {d}")

    print()
    print("  LoRA target modules (LLaMA-style):")
    modules = ["q_proj", "k_proj", "v_proj", "o_proj",
               "gate_proj", "up_proj", "down_proj"]
    print(f"  {modules}")
    print("  Typical: q_proj + v_proj only -> fewer params, often sufficient")


# -- 3. QLoRA ------------------------------------------------------------------
def qlora_overview():
    print("\n=== QLoRA (Quantised LoRA) ===")
    print()
    print("  Dettmers et al. (2023): Fine-tune 65B LLM on single A100 (40GB)")
    print()
    print("  Key innovations:")
    innovations = [
        ("4-bit NormalFloat (NF4)", "Quantile quantisation; optimal for normal distributions"),
        ("Double quantisation",     "Quantise the quantisation constants; saves ~0.37 bits/param"),
        ("Paged optimisers",        "Manage optimiser state memory with unified GPU/CPU memory"),
        ("LoRA on quantised model", "Adapters in BF16; gradients through 4-bit frozen weights"),
    ]
    for i, d in innovations:
        print(f"  {i:<30} {d}")
    print()
    print("  Memory savings (70B model):")
    print("    Full BF16 FT: ~280GB GPU  (not possible on typical hardware)")
    print("    LoRA BF16:    ~140GB GPU  (2× A100 80GB)")
    print("    QLoRA INT4:   ~40GB GPU   (single A100 40GB!)")

    print()
    print("  QLoRA vs full FT quality gap:")
    print("    Guanaco-65B (QLoRA): 99.3% of ChatGPT on Vicuna benchmark")
    print("    Essentially no quality loss in practice for most tasks")


# -- 4. Prompt tuning and prefix tuning ----------------------------------------
def soft_prompt_methods():
    print("\n=== Soft Prompt Methods ===")
    print()
    print("  Prompt Tuning (Lester 2021):")
    print("    Prepend n trainable vectors to input embeddings")
    print("    Only these vectors are updated; model frozen")
    print("    Scales well: at 10B+ params, matches full fine-tuning")
    print()
    # Toy demo
    rng = np.random.default_rng(0)
    n_tokens = 4; d_model = 32
    soft_prompts = rng.normal(0, 0.02, (n_tokens, d_model))
    task_input   = rng.normal(0, 1, (8, d_model))   # 8 real tokens
    # Prepend soft prompts
    full_input = np.vstack([soft_prompts, task_input])
    print(f"  Task input: {task_input.shape} -> with prompt tokens: {full_input.shape}")

    print()
    print("  Prefix Tuning (Li & Liang 2021):")
    print("    Learn virtual K, V vectors prepended at every layer")
    print("    More expressive than prompt tuning")
    print("    Virtual prefix length P: 10-100 tokens")
    print()
    print("  Comparison:")
    comp = [
        ("Full fine-tuning",    "100%",  "Best quality; most expensive"),
        ("LoRA r=8",            "0.1%",  "Near full-FT; most practical"),
        ("Prefix Tuning P=100", "~0.1%", "Competitive; no weight merge needed"),
        ("Prompt Tuning P=20",  "<0.1%", "Only for large models (10B+)"),
        ("IA3",                 "0.01%", "Fastest; for many-task inference"),
    ]
    print(f"  {'Method':<24} {'Params%':<10} {'Notes'}")
    for m, p, d in comp:
        print(f"  {m:<24} {p:<10} {d}")


if __name__ == "__main__":
    peft_overview()
    lora_demo()
    qlora_overview()
    soft_prompt_methods()
