# 7.5 Compute Resources

Understanding compute economics is essential for practical ML: FLOPs per forward pass, GPU memory footprint, training time, and cost. Chinchilla scaling laws prescribe compute-optimal token budgets (D ≈ 20N). This folder implements a FLOP estimator, GPU memory calculator, Chinchilla scaling visualisation, and training-time vs MFU analysis.

## Files

| File | Description |
|------|-------------|
| `working_example2.py` | FLOP estimator, GPU memory by model size, Chinchilla D=20N curve, MFU vs training hours |
| `working_example.ipynb` | Interactive notebook version |
| `output/` | `compute_resources.png` |

## Quick Start

```bash
python working_example2.py
```

## Key Concepts

| Concept | Description |
|---------|-------------|
| FLOPs | 6ND per training run (N=params, D=tokens) |
| MFU | Model FLOP Utilisation: actual vs theoretical peak |
| Chinchilla | Compute-optimal: D ≈ 20N; both scale together |
| GPU memory | Weights + gradients + optimizer states + activations |
| Cloud costs | H100 ~$3/hr; A100 ~$2/hr; spot pricing |  

## Learning Resources

- Hoffmann et al. *Chinchilla* (2022)
- Narayanan et al. *Megatron-LM* (2021)
- *Transformer Math 101* (EleutherAI blog)
