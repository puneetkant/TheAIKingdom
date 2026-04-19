"""
Working Example: Text-to-Video Generation
Covers video diffusion models, architectures, temporal modelling,
and major model families.
"""
import numpy as np
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_t2v")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -- 1. T2V landscape ----------------------------------------------------------
def t2v_landscape():
    print("=== Text-to-Video Generation ===")
    print()
    models = [
        ("Sora",          "OpenAI 2024; video DiT; world model; 1080p 1-min"),
        ("Runway Gen-3",  "Runway; commercial; motion consistency; API"),
        ("Kling",         "Kuaishou; 2 min; 1080p; strong motion"),
        ("CogVideoX",     "ZHIPU AI; open weights; expert attention"),
        ("Wan 2.1",       "Alibaba; open-source; best open T2V 2025"),
        ("Mochi 1",       "Genmo; open; 480p; high fidelity motion"),
        ("HunyuanVideo",  "Tencent; 720p; 5s; open weights"),
        ("Lumiere",       "Google; space-time UNet; temporal consistency"),
        ("AnimateDiff",   "Open; SD-based; motion module add-on"),
    ]
    for m, d in models:
        print(f"  {m:<18} {d}")


# -- 2. Architecture -----------------------------------------------------------
def video_architecture():
    print("\n=== Video Diffusion Architectures ===")
    print()
    print("  Challenge vs images: TEMPORAL CONSISTENCY")
    print("  Videos = sequences of image frames; motion must be coherent")
    print()
    print("  Architecture families:")
    arches = [
        ("Inflate UNet",   "2D Conv -> 3D Conv; temporal layers; AnimateDiff, ModelScopeT2V"),
        ("Video DiT",      "Diffusion Transformer over spacetime patches; Sora, CogVideoX"),
        ("U-Net + temporal","Spatial UNet + temporal attention blocks; Runway, Stable Video"),
        ("Autoregressive", "Generate frame-by-frame; VILM; slow but long videos"),
    ]
    for a, d in arches:
        print(f"  {a:<18} {d}")
    print()
    print("  Temporal modelling strategies:")
    strategies = [
        ("Temporal attention", "Attend over frames at each spatial position"),
        ("3D attention",       "Spacetime attention: width x height x time"),
        ("Factorised attn",    "Spatial then temporal (cheaper); quality tradeoff"),
        ("Causal conv",        "1D conv over time axis; streaming generation"),
        ("Flow matching",      "Rectified flow; straighter ODE trajectories"),
    ]
    for s, d in strategies:
        print(f"  {s:<22} {d}")


# -- 3. Sora architecture ------------------------------------------------------
def sora_architecture():
    print("\n=== Sora (Video DiT) Architecture ===")
    print()
    print("  Key innovations:")
    print("  1. Spacetime patches: videos -> 3D patch tokens (any resolution/duration)")
    print("  2. Vision encoder: compress video -> latents with 3D VAE")
    print("  3. DiT backbone: transformer over spacetime tokens")
    print("  4. Text conditioning: CLIP/T5 -> cross-attention throughout")
    print("  5. Flexible tokens: handles variable length/resolution natively")
    print()
    print("  Why DiT beats UNet for video:")
    reasons = [
        "Scales better with compute (consistent quality vs size)",
        "No hard-coded resolution bias (patches are size-agnostic)",
        "Global attention captures long-range temporal dependencies",
        "Simpler to parallelise across tokens",
    ]
    for r in reasons:
        print(f"  + {r}")
    print()
    print("  Scaling laws for video generation:")
    print("    More compute -> better temporal coherence and motion quality")
    print("    Estimated Sora training: ~4000+ A100 GPU days")


# -- 4. Evaluation -------------------------------------------------------------
def t2v_evaluation():
    print("\n=== Video Generation Evaluation ===")
    print()
    metrics = [
        ("FVD",            "Frechet Video Distance; quality + diversity"),
        ("CLIP-SIM",       "Frame-level prompt alignment; temporal average"),
        ("Flow consistency","Optical flow coherence; motion smoothness"),
        ("DOVER",          "Perception + technical quality; no reference"),
        ("EvalCrafter",    "Prompt adherence, quality, action; 700 prompts"),
        ("VBench",         "Comprehensive; 16 dimensions; 950 prompts"),
        ("T2V-CompBench",  "Temporal composition; 'A then B' type prompts"),
    ]
    print(f"  {'Metric':<18} {'Description'}")
    for m, d in metrics:
        print(f"  {m:<18} {d}")
    print()
    print("  Current limitations:")
    limitations = [
        "Physics: fluid dynamics, collisions often wrong",
        "Hands and faces: still artefacts at high motion",
        "Long video (>30s): consistency degrades",
        "Text in video: mostly fails",
        "Cost: very expensive per second of video",
    ]
    for l in limitations:
        print(f"  - {l}")


if __name__ == "__main__":
    t2v_landscape()
    video_architecture()
    sora_architecture()
    t2v_evaluation()
