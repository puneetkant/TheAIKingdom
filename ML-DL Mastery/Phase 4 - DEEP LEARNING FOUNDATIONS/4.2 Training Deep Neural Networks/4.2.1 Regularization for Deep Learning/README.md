# 4.2.1 Regularization for Deep Learning

L2 weight decay, Dropout, early stopping, Batch Normalization — fighting overfitting.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | L2 penalty demo, overfitting baseline |
| `working_example2.py` | Train/val loss curves: no reg vs L2 vs Dropout vs both |
| `working_example.ipynb` | Interactive: 3-way comparison → loss plots |

## Quick Reference

```python
import torch.nn as nn

# Dropout (training mode)
nn.Dropout(p=0.5)   # zeroes 50% of neurons, scales remaining by 1/(1-p)

# L2 (weight decay) in optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

# Batch Normalization
nn.BatchNorm1d(num_features)  # after linear, before activation

# Early stopping (manual)
if val_loss < best_val: best_val = val_loss; patience_counter = 0
else: patience_counter += 1
if patience_counter >= patience: break
```

## Regularization Methods

| Method | Mechanism | Adds |
|--------|-----------|------|
| L2 / Weight Decay | Shrinks weights | λ‖W‖² to loss |
| Dropout | Random zero out | p (drop rate) |
| BatchNorm | Normalizes activations | μ, σ per batch |
| Early Stopping | Stop at best val | patience |

## Learning Resources
- [Dropout paper (Srivastava 2014)](https://jmlr.org/papers/v15/srivastava14a.html)
- [PyTorch regularization guide](https://pytorch.org/docs/stable/nn.html)

Explore this topic with a small practical project or coding exercise.

## What to build

- Try a small hands-on exercise focused on this topic.
- Keep the code in `project.py` in this folder.
- Add notes, examples, or results inside this directory.

## Suggestions

1. Read the checklist topic and identify one practice task.
2. Write code in `project.py` that illustrates the main concept.
3. Run your code and iterate until it works.

## Notes

- Use Python and standard libraries when possible.
- For data topics, install `numpy`, `pandas`, `matplotlib` as needed.
