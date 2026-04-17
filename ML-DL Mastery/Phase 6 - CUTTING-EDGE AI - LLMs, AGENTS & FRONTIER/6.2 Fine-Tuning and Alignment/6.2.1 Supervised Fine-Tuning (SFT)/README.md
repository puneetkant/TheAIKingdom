# 6.2.1 Supervised Fine-Tuning (SFT)

Supervised fine-tuning adapts a pre-trained LLM to follow instructions by training on curated (prompt, response) pairs. It bridges the gap between raw language modelling and useful assistants. This folder simulates loss curves for full fine-tuning vs instruction fine-tuning, comparing convergence speed and generalisation.

## Files

| File | Description |
|------|-------------|
| `working_example2.py` | SFT vs base model loss comparison, format template demo, convergence curves |
| `working_example.ipynb` | Interactive notebook version |
| `output/` | `sft_comparison.png` |

## Quick Start

```bash
python working_example2.py
```

## Key Concepts

| Concept | Description |
|---------|-------------|
| Instruction following | Training on (prompt, completion) pairs |
| Chat template | System / user / assistant message formatting |
| Data quality | Small high-quality datasets often beat large noisy ones |
| Catastrophic forgetting | Fine-tuning can degrade pre-training knowledge |
| Alpaca / FLAN | Early open instruction-tuning datasets |

## Learning Resources

- Wei et al. *FLAN* (2021)
- Taori et al. *Alpaca* (2023)
- OpenAI *InstructGPT* (2022)
