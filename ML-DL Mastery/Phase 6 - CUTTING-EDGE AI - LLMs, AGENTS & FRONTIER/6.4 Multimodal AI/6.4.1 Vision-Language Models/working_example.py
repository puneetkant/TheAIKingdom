"""
Working Example: Vision-Language Models (VLMs)
Covers CLIP, LLaVA, GPT-4V, Gemini Vision, and multimodal architectures.
"""
import numpy as np
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_vlm")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── 1. VLM landscape ──────────────────────────────────────────────────────────
def vlm_landscape():
    print("=== Vision-Language Models (VLMs) ===")
    print()
    print("  Two main paradigms:")
    print("  1. Contrastive (CLIP): align image + text embeddings")
    print("  2. Generative (LLaVA, GPT-4o): LLM generates text conditioned on image")
    print()
    models = [
        ("CLIP",              "OpenAI 2021; contrastive; image-text similarity"),
        ("Flamingo",          "DeepMind; few-shot VL via cross-attention"),
        ("LLaVA-1.5/1.6",    "Open; LLaMA + CLIP vision encoder; popular open VLM"),
        ("InstructBLIP",      "Salesforce; Q-Former bridge; instruction following"),
        ("GPT-4o",            "OpenAI; natively multimodal; vision + audio + text"),
        ("Claude 3.5 Sonnet", "Anthropic; strong document + visual reasoning"),
        ("Gemini 1.5 Pro",    "Google; 1M context; video understanding"),
        ("Qwen-VL",           "Alibaba; open; strong on OCR + charts"),
        ("InternVL2",         "Open-source; competitive with proprietary"),
        ("PaliGemma",         "Google; Gemma + SigLIP; open weights"),
    ]
    for m, d in models:
        print(f"  {m:<24} {d}")


# ── 2. CLIP architecture ──────────────────────────────────────────────────────
def clip_architecture():
    print("\n=== CLIP Architecture ===")
    print()
    print("  CLIP = Contrastive Language-Image Pre-training")
    print("  Trained on 400M (image, text) pairs from the web")
    print()
    print("  Architecture:")
    print("    Image encoder: ViT-B/32, ViT-L/14, ViT-L/14@336px")
    print("    Text encoder:  Transformer (63M params; 77 token max)")
    print("    Projection:    Both → shared embedding space (512d / 768d)")
    print()

    # Simulate CLIP similarity
    rng = np.random.default_rng(42)
    d = 8

    # Image and text embeddings
    image_embed = rng.normal(0, 1, d)
    image_embed /= np.linalg.norm(image_embed)

    texts = ["a photo of a cat", "a photo of a dog", "a beach at sunset", "computer code"]
    text_embeds = []
    for t in texts:
        e = rng.normal(0, 1, d)
        e /= np.linalg.norm(e)
        text_embeds.append(e)

    # Make "a photo of a cat" most similar
    text_embeds[0] = image_embed + rng.normal(0, 0.1, d)
    text_embeds[0] /= np.linalg.norm(text_embeds[0])

    print("  Simulated CLIP zero-shot classification:")
    print("  (image vs text candidates)")
    sims = np.array([float(image_embed @ t) for t in text_embeds])
    probs = np.exp(sims * 10)
    probs /= probs.sum()
    for t, p in sorted(zip(texts, probs), key=lambda x: -x[1]):
        bar = "█" * int(p * 40)
        print(f"    {t:<28} {p:.3f} {bar}")

    print()
    print("  CLIP applications:")
    apps = [
        "Zero-shot image classification",
        "Text-guided image retrieval",
        "Vision encoder for LLaVA / BLIP-2",
        "Image-text matching for DALL-E 3 / Stable Diffusion",
        "Video understanding (frame-level similarity)",
    ]
    for a in apps:
        print(f"  • {a}")


# ── 3. LLaVA architecture ─────────────────────────────────────────────────────
def llava_architecture():
    print("\n=== LLaVA Architecture ===")
    print()
    print("  LLaVA = Large Language and Vision Assistant")
    print("  Components:")
    components = [
        ("Vision encoder",  "CLIP ViT-L/14 → 256 visual tokens (14x14 grid)"),
        ("Projection",      "MLP (LLaVA-1.5) or Q-Former (InstructBLIP)"),
        ("LLM",             "LLaMA-2 / Mistral / Vicuna; receives projected tokens"),
        ("Training",        "Stage 1: alignment (freeze LLM); Stage 2: instruction tuning"),
    ]
    for c, d in components:
        print(f"  {c:<18} {d}")
    print()
    print("  LLaVA conversation format:")
    conv = [
        ("System",     "A chat between a curious user and an AI assistant."),
        ("User",       "<image> Describe what is happening in this image."),
        ("Assistant",  "The image shows a golden retriever playing fetch..."),
        ("User",       "What breed of dog is it?"),
        ("Assistant",  "It appears to be a Golden Retriever based on..."),
    ]
    for role, content in conv:
        print(f"  [{role}]: {content}")
    print()
    print("  Visual token strategies:")
    strategies = [
        ("Naive",       "Flatten all ViT patches → token sequence"),
        ("High-res",    "LLaVA-1.6: dynamic resolution; slice + encode tiles"),
        ("Anyres",      "Adaptive tiling; preserve aspect ratio"),
        ("StreamingLLM","Attention sinks; process video frames sequentially"),
    ]
    for s, d in strategies:
        print(f"  {s:<12} {d}")


# ── 4. VLM evaluation ─────────────────────────────────────────────────────────
def vlm_evaluation():
    print("\n=== VLM Evaluation Benchmarks ===")
    print()
    benchmarks = [
        ("MMBench",      "Comprehensive; 2K+ questions; perception/reasoning"),
        ("MMMU",         "College-level multi-discipline; 11.5K questions"),
        ("VQAv2",        "Visual QA; natural images; accuracy metric"),
        ("TextVQA",      "OCR + QA; reading text in images"),
        ("DocVQA",       "Document understanding; forms, receipts"),
        ("ChartQA",      "Chart and graph interpretation"),
        ("ScienceQA",    "Multi-modal science; 21K questions"),
        ("MMStar",       "Anti-contamination; 1.5K balanced; hard"),
        ("HallusionBench","Hallucination detection; visual illusions"),
    ]
    for b, d in benchmarks:
        print(f"  {b:<18} {d}")


if __name__ == "__main__":
    vlm_landscape()
    clip_architecture()
    llava_architecture()
    vlm_evaluation()
