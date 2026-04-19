"""
Working Example: Multimodal Embeddings and Retrieval
Covers joint vision-language embeddings, cross-modal retrieval,
and multimodal vector databases.
"""
import numpy as np
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_mm_embed")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def cosine_sim_matrix(A, B):
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-9)
    return A @ B.T


# -- 1. Multimodal embedding overview -----------------------------------------
def mm_embedding_overview():
    print("=== Multimodal Embeddings ===")
    print()
    print("  Goal: map different modalities into a shared vector space")
    print("  cos_sim(image_embed, text_embed) measures semantic alignment")
    print()
    models = [
        ("CLIP",          "512/768d; contrastive; universal; OpenAI"),
        ("SigLIP",        "Google; sigmoid loss; better at scale; used in Gemma"),
        ("ALIGN",         "Google; 1.8B image-text pairs; EfficientNet"),
        ("BLIP-2",        "Bootstrapped; Q-Former; generative + retrieval"),
        ("ImageBind",     "Meta; 6 modalities: image/text/audio/depth/IMU/video"),
        ("E5-Mistral",    "Text-only but strong; instruction-tuned embeddings"),
        ("OpenCLIP",      "Open-source CLIP training; many model sizes"),
        ("CoCa",          "Contrastive + captioning; dual-task training"),
    ]
    print(f"  {'Model':<16} {'Notes'}")
    for m, d in models:
        print(f"  {m:<16} {d}")


# -- 2. Cross-modal retrieval demo ---------------------------------------------
def cross_modal_retrieval():
    print("\n=== Cross-Modal Retrieval Demo ===")
    print()

    rng = np.random.default_rng(42)
    d = 12  # embedding dim

    # Simulate shared embedding space
    texts = [
        "a golden retriever playing fetch on the beach",
        "a city skyline at night with neon lights",
        "a scientist working in a laboratory",
        "mountain landscape with snow-capped peaks",
        "a chef preparing food in a restaurant kitchen",
    ]

    images = [
        "photo_golden_dog_beach.jpg",
        "photo_tokyo_night.jpg",
        "photo_lab_scientist.jpg",
        "photo_alps_snow.jpg",
        "photo_chef_kitchen.jpg",
    ]

    # Simulate aligned embeddings (text and matching image close in space)
    base_embeds = rng.normal(0, 1, (len(texts), d))
    text_embeds  = base_embeds + rng.normal(0, 0.1, base_embeds.shape)
    image_embeds = base_embeds + rng.normal(0, 0.1, base_embeds.shape)

    # Text -> Image retrieval
    print("  Text -> Image retrieval:")
    query_text = text_embeds[2]   # "a scientist..."
    sims = cosine_sim_matrix(query_text[None], image_embeds)[0]
    ranked = np.argsort(-sims)
    for rank, idx in enumerate(ranked):
        marker = " <-- correct" if idx == 2 else ""
        print(f"    {rank+1}. [sim={sims[idx]:.3f}] {images[idx]}{marker}")

    print()
    print("  Image -> Text retrieval:")
    query_image = image_embeds[4]  # chef kitchen
    sims = cosine_sim_matrix(query_image[None], text_embeds)[0]
    ranked = np.argsort(-sims)
    for rank, idx in enumerate(ranked):
        marker = " <-- correct" if idx == 4 else ""
        print(f"    {rank+1}. [sim={sims[idx]:.3f}] {texts[idx][:50]}{marker}")

    print()
    # Compute retrieval recall@k
    correct_at_1 = int(np.argmax(sims) == 4)
    print(f"  Recall@1 (image->text): {correct_at_1}")


# -- 3. ImageBind: 6 modalities -----------------------------------------------
def imagebind_demo():
    print("\n=== ImageBind: 6-Modality Embedding ===")
    print()
    print("  Meta 2023: single shared embedding space for 6 modalities")
    print("  Train: always pair with images (image as anchor)")
    print()
    modalities = [
        ("Image",  "ViT-H/14 patch embeddings"),
        ("Text",   "Transformer (CLIP-style)"),
        ("Audio",  "Spectrogram patches -> ViT"),
        ("Video",  "SpaceTime ViT (image frames)"),
        ("Depth",  "Single-channel image patches"),
        ("IMU",    "Accelerometer/gyro: 1D conv + transformer"),
    ]
    for m, d in modalities:
        print(f"  {m:<8} {d}")
    print()
    print("  Emergent cross-modal zero-shot:")
    print("    Audio of dog barking -> retrieve dog images (no audio-image pairs trained)")
    print("    Because audio <-> image <-> text alignment generalises")


# -- 4. Multimodal RAG ---------------------------------------------------------
def multimodal_rag():
    print("\n=== Multimodal RAG ===")
    print()
    print("  Challenge: retrieve relevant content from mixed text+image+table docs")
    print()
    approaches = [
        ("Late fusion",    "Separate text/image vectorstores; fuse retrieved results"),
        ("Caption-based",  "Caption images -> treat as text; store text embeddings"),
        ("Multi-vector",   "Store image embedding + text embedding separately"),
        ("ColPali",        "ViT patch embeddings for PDF pages; no OCR needed"),
        ("RAGatouille",    "ColBERT for text retrieval; late interaction"),
    ]
    print(f"  {'Approach':<18} {'Description'}")
    for a, d in approaches:
        print(f"  {a:<18} {d}")
    print()
    print("  ColPali (2024) key insight:")
    print("    Embed entire PDF page as image -> multi-vector using ViT patches")
    print("    Query: embed text -> late interaction score with page patches")
    print("    Handles figures, charts, tables without OCR preprocessing")


if __name__ == "__main__":
    mm_embedding_overview()
    cross_modal_retrieval()
    imagebind_demo()
    multimodal_rag()
