# 6.2.2 Parameter-Efficient Fine-Tuning (PEFT)

PEFT techniques adapt large models by updating only a small fraction of parameters — enabling fine-tuning on consumer GPUs. LoRA (Low-Rank Adaptation) injects trainable rank-decomposition matrices into frozen weight layers. This folder demonstrates LoRA forward passes, rank sensitivity, and parameter-count comparisons.

## Files

| File | Description |
|------|-------------|
| `working_example2.py` | LoRA matrix decomposition demo, rank vs quality trade-off, param count comparison chart |
| `working_example.ipynb` | Interactive notebook version |
| `output/` | `peft_lora.png` |

## Quick Start

```bash
python working_example2.py
```

## Key Concepts

| Concept | Description |
|---------|-------------|
| LoRA | W ≈ W₀ + BA; train only B and A (rank r ≪ d) |
| Adapter layers | Small MLP modules inserted between transformer layers |
| Prefix tuning | Prepend soft trainable tokens to input |
| QLoRA | Quantise base model to 4-bit, fine-tune LoRA adapters |
| Rank r | Controls capacity vs parameter budget |

## Learning Resources

- Hu et al. *LoRA* (2021)
- Dettmers et al. *QLoRA* (2023)
- HuggingFace PEFT library
