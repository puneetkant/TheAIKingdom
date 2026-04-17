# 6.4.1 Vision-Language Models

Vision-language models (CLIP, BLIP, LLaVA, GPT-4V) align image and text representations into a shared embedding space, enabling zero-shot image classification, visual Q&A, and image captioning. This folder implements CLIP-style contrastive learning with InfoNCE loss and cosine similarity retrieval using numpy.

## Files

| File | Description |
|------|-------------|
| `working_example2.py` | CLIP cosine similarity, InfoNCE contrastive loss, temperature sweep, image-text retrieval |
| `working_example.ipynb` | Interactive notebook version |
| `output/` | `vision_language.png` |

## Quick Start

```bash
python working_example2.py
```

## Key Concepts

| Concept | Description |
|---------|-------------|
| CLIP | Contrastive Language-Image Pre-training; dual encoders |
| InfoNCE loss | Contrastive loss; pull positives, push negatives |
| Zero-shot classification | Encode class names and compare to image embedding |
| Projection head | MLP maps vision/text encoders to shared space |
| Temperature τ | Scales logit sharpness; τ↓ = more peaked |

## Learning Resources

- Radford et al. *CLIP* (2021)
- Li et al. *BLIP* (2022)
- Liu et al. *LLaVA* (2023)
