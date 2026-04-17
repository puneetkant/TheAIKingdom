# 6.5.6 Synthetic Data and Self-Improvement

Synthetic data generation (self-play, model-generated labels, Mixtral distillation) allows models to bootstrap training beyond human-curated datasets. Self-improvement loops apply pseudo-labelling, Mixup augmentation, and iterative fine-tuning. This folder implements feature noise augmentation, Mixup, and a self-training loop that adds high-confidence pseudo-labels over multiple rounds.

## Files

| File | Description |
|------|-------------|
| `working_example2.py` | Feature noise augmentation, Mixup, self-training pseudo-label loop (5 rounds) |
| `working_example.ipynb` | Interactive notebook version |
| `output/` | `synthetic_data.png` |

## Quick Start

```bash
python working_example2.py
```

## Key Concepts

| Concept | Description |
|---------|-------------|
| Pseudo-labelling | Use model’s own high-confidence predictions as labels |
| Mixup | Linearly interpolate pairs of samples and labels |
| Self-play | Model generates problems and solutions for itself |
| Distillation | Small model trained on large model’s soft logits |
| Rejection sampling | Filter generated outputs by quality metric |

## Learning Resources

- Lee *Pseudo-Label* (2013)
- Zhang et al. *Mixup* (2018)
- Gunasekar et al. *Phi-1* (2023)
