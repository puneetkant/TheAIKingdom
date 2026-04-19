"""
Working Example: Compute Resources
Covers GPU selection, cloud compute, cost optimisation, free resources,
and efficient training strategies.
"""
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_compute")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -- 1. GPU overview -----------------------------------------------------------
def gpu_overview():
    print("=== Compute Resources for ML ===")
    print()
    print("  GPU comparison (2024-2025):")
    gpus = [
        # (Name, VRAM, TFlops BF16, bandwidth GB/s, NVLink, ~Cloud $/hr)
        ("RTX 3090",    "24 GB", 142,  936, False, "~$0.70 (local)"),
        ("RTX 4090",    "24 GB", 330,  1008, False, "~$0.90-1.20"),
        ("A100 40GB",   "40 GB", 312,  1935, True,  "~$2.50-3.50"),
        ("A100 80GB",   "80 GB", 312,  2000, True,  "~$3.50-4.50"),
        ("H100 SXM",    "80 GB", 1979, 3350, True,  "~$5-8"),
        ("H100 NVL",    "94 GB", 1979, 3938, True,  "~$6-9"),
        ("H200",        "141 GB",1979, 4800, True,  "~$7-10"),
        ("B200",        "192 GB",4500, 8000, True,  "~$12-15 est."),
    ]
    print(f"  {'GPU':<14} {'VRAM':<8} {'TFLOPs':<8} {'BW GB/s':<10} {'NVLink':<8} {'$/hr est.'}")
    for name, vram, tflops, bw, nvlink, cost in gpus:
        nv = "Yes" if nvlink else "No"
        print(f"  {name:<14} {vram:<8} {tflops:<8} {bw:<10} {nv:<8} {cost}")


# -- 2. Cloud platforms --------------------------------------------------------
def cloud_platforms():
    print("\n=== Cloud Compute Platforms ===")
    print()
    platforms = [
        ("AWS SageMaker",    "Managed; p4d (A100x8); p5 (H100x8); g4dn (T4)"),
        ("GCP Vertex AI",    "TPU v4/v5; A100/H100; BigQuery integration"),
        ("Azure ML",         "ND H100; good for enterprise Microsoft stack"),
        ("Lambda Labs",      "GPU cloud; cheapest A100; per-hour; ML-focused"),
        ("CoreWeave",        "GPU specialist; H100 clusters; lowest H100 price"),
        ("RunPod",           "Community + secure; cheap spot; good for experiments"),
        ("Vast.ai",          "Peer-to-peer GPU rental; very cheap spot"),
        ("Modal",            "Serverless; pay per second; great for inference"),
        ("Replicate",        "Model deployment; per-prediction billing"),
        ("Together.ai",      "Fine-tuning + inference; competitive pricing"),
    ]
    print(f"  {'Platform':<18} {'Notes'}")
    for p, d in platforms:
        print(f"  {p:<18} {d}")


# -- 3. Free resources ---------------------------------------------------------
def free_resources():
    print("\n=== Free Compute Resources ===")
    print()
    resources = [
        ("Google Colab",       "T4/V100/A100; 12-24h session; 15 GB RAM; free tier"),
        ("Colab Pro ($10/mo)", "A100; priority; longer sessions; 50 GB RAM"),
        ("Kaggle Kernels",     "T4x2 / P100; 30h GPU/week; persistent storage"),
        ("Hugging Face Spaces","Free CPU; paid GPU; auto-sleep"),
        ("Lightning.ai",       "T4 free tier; L4 paid; Jupyter-like"),
        ("Paperspace Gradient","Free M4000; paid A100; notebooks"),
        ("GitHub Codespaces",  "Free 60h/month; CPU only; good for CPU tasks"),
        ("Google Cloud TPU",   "Research program; apply for credits"),
    ]
    print(f"  {'Resource':<26} {'Notes'}")
    for r, d in resources:
        print(f"  {r:<26} {d}")
    print()
    print("  Tips for free tiers:")
    tips = [
        "Use Kaggle for GPU (30h/week, persistent datasets)",
        "Use Colab for quick prototyping; download checkpoints often",
        "Batch multiple experiments then disconnect to save session time",
        "Use gradient checkpointing to fit larger models in 16GB VRAM",
        "8-bit optimisers (bitsandbytes) halve memory for weight states",
    ]
    for t in tips:
        print(f"  • {t}")


# -- 4. Cost optimisation ------------------------------------------------------
def cost_optimisation():
    print("\n=== Cost Optimisation Strategies ===")
    print()
    strategies = [
        ("Spot instances",      "50-90% cheaper; interruptible; use checkpointing"),
        ("Mixed precision",     "BF16 training: same quality, half memory, faster"),
        ("Gradient checkpointing","Recompute activations; halve memory; 20% slower"),
        ("Gradient accumulation","Simulate large batch without large VRAM"),
        ("Data parallelism",    "Multi-GPU: cheaper than single large GPU"),
        ("Quantisation (QLoRA)","Fine-tune 70B on 1x A100 with NF4 + LoRA"),
        ("Parameter-efficient", "LoRA/IA3 train <1% params; 10x fewer GPU days"),
        ("Smaller model first",  "Validate idea on 1B before scaling to 70B"),
        ("Compile models",      "torch.compile: 20-40% faster training free"),
        ("Flash Attention 2",   "Memory-efficient attention; 2x faster on A100"),
    ]
    print(f"  {'Strategy':<26} {'Notes'}")
    for s, d in strategies:
        print(f"  {s:<26} {d}")

    print()
    print("  Rough cost estimates for common tasks:")
    tasks = [
        ("GPT-2 fine-tune (1h)", "1x T4, ~$0.40"),
        ("LLaMA-3 8B SFT (8h)",  "1x A100, ~$28"),
        ("LLaMA-3 70B QLoRA",    "2x A100, ~$56"),
        ("Train from scratch 7B","32x A100, ~weeks, ~$50k-200k"),
        ("Inference 7B, 1M tok", "Lambda g5.xlarge, ~$0.50"),
    ]
    for t, c in tasks:
        print(f"  {t:<30} {c}")


if __name__ == "__main__":
    gpu_overview()
    cloud_platforms()
    free_resources()
    cost_optimisation()
