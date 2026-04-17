# 6.1.3 Pre-training LLMs

Pre-training is the large-scale unsupervised phase that gives LLMs their world knowledge. A causal language model learns to predict the next token over trillions of tokens, building up a rich internal representation of language, facts, and reasoning patterns. This folder explores bigram statistics, loss curves, perplexity, and scaling dynamics from scratch.

## Files

| File | Description |
|------|-------------|
| `working_example2.py` | Bigram language model, perplexity tracking, loss-curve simulation, heatmap of bigram co-occurrences |
| `working_example.ipynb` | Interactive notebook version with step-by-step cells |
| `output/` | `pretraining_curves.png`, `bigram_heatmap.png` |

## Quick Start

```bash
python working_example2.py
```

## Key Concepts

| Concept | Description |
|---------|-------------|
| Causal LM objective | Predict next token; minimise cross-entropy loss |
| Perplexity | Exponentiated average NLL; lower = better model |
| Bigram statistics | Character/token co-occurrence counts |
| Scaling laws | Loss ∝ N^-α · D^-β (Chinchilla) |
| Training dynamics | Learning-rate warmup → decay |

## Learning Resources

- Karpathy "makemore" series
- Hoffmann et al. *Chinchilla* (2022)
- Brown et al. *GPT-3* (2020)
