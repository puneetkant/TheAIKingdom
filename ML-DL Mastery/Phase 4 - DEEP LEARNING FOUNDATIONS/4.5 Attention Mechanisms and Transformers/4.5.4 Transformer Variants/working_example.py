"""
Working Example: Transformer Variants
Covers BERT (encoder-only), GPT (decoder-only), T5 (encoder-decoder),
Vision Transformer (ViT), and key architectural differences.
"""
import numpy as np


# ── 1. The Transformer Family Tree ───────────────────────────────────────────
def family_tree():
    print("=== Transformer Family Tree ===")
    print()
    print("  Original Transformer (Vaswani 2017)")
    print("    ├── Encoder-only")
    print("    │     BERT (Devlin 2018)  → RoBERTa, ALBERT, DeBERTa, ELECTRA")
    print("    ├── Decoder-only (GPT-style)")
    print("    │     GPT (OpenAI 2018)   → GPT-2, GPT-3, GPT-4, LLaMA, Mistral")
    print("    │                            PaLM, Claude, Gemini, Falcon")
    print("    └── Encoder-Decoder")
    print("          T5 (Raffel 2020)    → BART, mT5, Flan-T5, UL2")
    print()
    print("  Vision variants:")
    print("    ViT (Dosovitskiy 2020)    → DeiT, Swin, BEiT, MAE")
    print()
    print("  Multi-modal:")
    print("    CLIP, DALL-E, Flamingo, GPT-4V, LLaVA")


# ── 2. BERT ───────────────────────────────────────────────────────────────────
def bert():
    print("\n=== BERT (Bidirectional Encoder Representations from Transformers) ===")
    print("  Architecture: Encoder-only Transformer")
    print("  Pre-training: self-supervised on massive text corpus")
    print()
    print("  Pre-training tasks:")
    print("    1. Masked Language Modelling (MLM):")
    print("       15% of tokens randomly masked → predict original token")
    print("       [CLS] The [MASK] sat on the mat → 'cat'")
    print()
    print("    2. Next Sentence Prediction (NSP):")
    print("       [CLS] sentence A [SEP] sentence B [SEP] → IsNext or NotNext")
    print("       (Later work showed NSP not very helpful)")
    print()
    print("  Fine-tuning: add task-specific head, train all weights end-to-end")
    print()
    variants = [
        ("BERT-base",    "12",  "768",  "12",  "110M",  "Standard"),
        ("BERT-large",   "24",  "1024", "16",  "340M",  "Better acc"),
        ("RoBERTa",      "24",  "1024", "16",  "355M",  "No NSP, larger batches"),
        ("ALBERT",       "12",  "768",  "12",  "12M",   "Parameter sharing"),
        ("DistilBERT",   "6",   "768",  "12",  "66M",   "Knowledge distillation"),
        ("DeBERTa-v3",   "12",  "768",  "12",  "86M",   "Disentangled attention"),
    ]
    print(f"  {'Model':<15} {'Layers':<8} {'d_model':<9} {'Heads':<7} {'Params':<8} {'Note'}")
    for name, L, d, h, p, note in variants:
        print(f"  {name:<15} {L:<8} {d:<9} {h:<7} {p:<8} {note}")

    print()
    print("  Fine-tuning tasks:")
    tasks = [
        ("Classification",  "[CLS] representation → classification head"),
        ("NER",             "Each token representation → NER label head"),
        ("QA (SQuAD)",      "Start/end position prediction over passage tokens"),
        ("NLI",             "[CLS] of [premise][SEP][hypothesis] → 3-way label"),
    ]
    for task, approach in tasks:
        print(f"    {task:<18}: {approach}")


# ── 3. GPT (decoder-only) ─────────────────────────────────────────────────────
def gpt():
    print("\n=== GPT Family (Decoder-Only / Autoregressive) ===")
    print("  Architecture: Decoder-only (masked self-attention only, no cross-attn)")
    print("  Pre-training: next token prediction (causal LM)")
    print("    P(w_1, w_2, ..., w_T) = Π P(w_t | w_1, ..., w_{t-1})")
    print()
    versions = [
        ("GPT-1",   "12",  "768",  "~117M",   "2018", "12-layer decoder"),
        ("GPT-2",   "48",  "1600", "~1.5B",   "2019", "Zero-shot generalization"),
        ("GPT-3",   "96",  "12288","~175B",   "2020", "Few-shot prompting"),
        ("GPT-4",   "?",   "?",    "~1T?",    "2023", "Multi-modal, RLHF"),
        ("LLaMA 2", "80",  "8192", "70B",     "2023", "Open weights"),
        ("Mistral", "32",  "4096", "7B",      "2023", "GQA + SWA"),
        ("Falcon",  "60",  "8192", "40B",     "2023", "Multi-query attn"),
    ]
    print(f"  {'Model':<12} {'Layers':<8} {'d_model':<9} {'Params':<10} {'Year':<6} {'Note'}")
    for name, L, d, p, yr, note in versions:
        print(f"  {name:<12} {L:<8} {d:<9} {p:<10} {yr:<6} {note}")

    print()
    print("  Key innovations in modern GPT models:")
    print("    RoPE:  rotary positional embeddings (relative positions)")
    print("    GQA:   grouped-query attention (fewer KV heads → less memory)")
    print("    SWA:   sliding window attention (Mistral)")
    print("    RLHF:  reinforcement learning from human feedback (GPT-4, Claude)")
    print("    SFT:   supervised fine-tuning on instruction datasets")


# ── 4. T5 (encoder-decoder) ──────────────────────────────────────────────────
def t5():
    print("\n=== T5 (Text-to-Text Transfer Transformer) ===")
    print("  Architecture: Encoder-Decoder (original Transformer)")
    print("  Key idea: EVERY NLP task reformulated as text-to-text")
    print()
    tasks = [
        ("Translation",     "translate English to German: The cat sat",  "Die Katze saß"),
        ("Summarisation",   "summarize: <article text>",                 "<summary>"),
        ("Classification",  "sentiment: The movie was great",            "positive"),
        ("QA",              "question: What is the capital? context: ..","London"),
        ("NLI",             "nli hypothesis: ... premise: ...",          "entailment"),
    ]
    print(f"  {'Task':<18} {'Input prefix':<45} {'Output'}")
    for task, inp, out in tasks:
        print(f"  {task:<18} {inp:<45} {out}")

    print()
    variants = [
        ("T5-Small",  "6",  "512",  "60M"),
        ("T5-Base",   "12", "768",  "220M"),
        ("T5-Large",  "24", "1024", "770M"),
        ("T5-3B",     "24", "1024", "3B"),
        ("T5-11B",    "24", "1024", "11B"),
        ("Flan-T5",   "24", "1024", "3-11B  instruction-tuned"),
    ]
    print(f"\n  {'Model':<14} {'Layers':<8} {'d_model':<10} {'Params'}")
    for name, L, d, p in variants:
        print(f"  {name:<14} {L:<8} {d:<10} {p}")


# ── 5. Vision Transformer (ViT) ───────────────────────────────────────────────
def vit():
    print("\n=== Vision Transformer (ViT, Dosovitskiy 2020) ===")
    print("  Split image into fixed-size patches; treat each patch as a token")
    print()
    print("  Pipeline:")
    print("    1. Split 224×224 image into 16×16 patches → 14×14 = 196 patches")
    print("    2. Flatten each patch: 16×16×3 = 768 values")
    print("    3. Linear projection: 768 → d_model (patch embeddings)")
    print("    4. Prepend learnable [CLS] token (used for classification)")
    print("    5. Add positional embeddings (learnable 1D)")
    print("    6. Pass through Transformer Encoder layers")
    print("    7. [CLS] output → classification head")
    print()

    # Compute patch tokens
    img_size  = 224
    patch_size = 16
    n_patches = (img_size // patch_size) ** 2
    d_patch   = patch_size * patch_size * 3
    print(f"  Image: {img_size}×{img_size}×3")
    print(f"  Patches: {img_size//patch_size}×{img_size//patch_size} = {n_patches} patches")
    print(f"  Patch dim: {patch_size}×{patch_size}×3 = {d_patch}")
    print(f"  Sequence length (incl [CLS]): {n_patches + 1}")

    variants = [
        ("ViT-Ti/16",   "12", "192",  "5.7M",  "~79%"),
        ("ViT-S/16",    "12", "384",  "22M",   "81.4%"),
        ("ViT-B/16",    "12", "768",  "86M",   "81.8%"),
        ("ViT-L/16",    "24", "1024", "307M",  "85.2%"),
        ("ViT-H/14",    "32", "1280", "632M",  "87.6%"),
        ("Swin-T",      "12", "768",  "28M",   "81.3%  shifted window"),
        ("Swin-L",      "24", "1536", "197M",  "86.4%  shifted window"),
    ]
    print(f"\n  {'Model':<14} {'Layers':<8} {'d_model':<9} {'Params':<8} {'Top-1'}")
    for name, L, d, p, acc in variants:
        print(f"  {name:<14} {L:<8} {d:<9} {p:<8} {acc}")

    print()
    print("  Key variants:")
    print("    DeiT:  data-efficient ViT (distillation token)")
    print("    Swin:  hierarchical + shifted windows (faster, local inductive bias)")
    print("    BEiT:  BERT-style pre-training for image patches")
    print("    MAE:   masked autoencoder (75% masking, reconstruct pixels)")


# ── 6. Architectural comparison ───────────────────────────────────────────────
def architectural_comparison():
    print("\n=== Transformer Variant Comparison ===")
    print(f"  {'Feature':<24} {'BERT':<20} {'GPT':<20} {'T5':<20} {'ViT'}")
    rows = [
        ("Architecture",    "Encoder-only",    "Decoder-only",    "Enc-Decoder",     "Encoder-only"),
        ("Attention type",  "Bidirectional",   "Causal (masked)", "Bi-enc + causal", "Bidirectional"),
        ("Pre-training",    "MLM + NSP",       "Next token pred", "Span masking",    "Supervised / MAE"),
        ("Best for",        "Understanding",   "Generation",      "Both tasks",      "Image tasks"),
        ("Fine-tuning",     "Add head",        "Prompt tuning",   "Text-to-text",    "Add head"),
        ("Typical use",     "NLU benchmarks",  "Chatbots, code",  "Translation/summ","Classification"),
    ]
    for name, bert, gpt, t5, vit in rows:
        print(f"  {name:<24} {bert:<20} {gpt:<20} {t5:<20} {vit}")


# ── 7. Key architectural improvements ─────────────────────────────────────────
def key_improvements():
    print("\n=== Key Architectural Improvements Over Time ===")
    improvements = [
        ("Pre-LayerNorm",       "Move LN before sublayer (more stable training)"),
        ("SwiGLU activation",   "Replace FFN ReLU with SwiGLU (LLaMA, PaLM)"),
        ("RoPE",                "Rotary position embeddings (relative, no learnables)"),
        ("ALiBi",               "Attention with linear biases (good extrapolation)"),
        ("Grouped Query Attn",  "Fewer KV heads; reduces KV-cache memory"),
        ("Flash Attention 2",   "Exact attention, 2-4× faster, O(T) memory"),
        ("MoE",                 "Mixture of Experts (Mixtral): sparse expert routing"),
        ("Multi-token pred",    "Predict next N tokens jointly (better representations)"),
    ]
    for name, desc in improvements:
        print(f"  {name:<26}: {desc}")


if __name__ == "__main__":
    family_tree()
    bert()
    gpt()
    t5()
    vit()
    architectural_comparison()
    key_improvements()
