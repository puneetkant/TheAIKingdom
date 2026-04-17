# 4.1.6 Weight Initialization

Zero init symmetry problem, large random (exploding), Xavier (sigmoid/tanh), He (ReLU).

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | Variance analysis per initialization scheme |
| `working_example2.py` | Training comparison: 4 strategies, loss curve plot |
| `working_example.ipynb` | Interactive: std/max table → training curves comparison |

## Quick Reference

```python
import numpy as np

n_in, n_out = 256, 128

# Xavier / Glorot (sigmoid/tanh)
W = np.random.randn(n_in, n_out) * np.sqrt(2 / (n_in + n_out))

# He (ReLU)
W = np.random.randn(n_in, n_out) * np.sqrt(2 / n_in)

# PyTorch default: Kaiming He
import torch.nn as nn
layer = nn.Linear(n_in, n_out)  # uses Kaiming He by default
```

## Initialization Rules

| Activation | Recommended | Formula |
|-----------|-------------|---------|
| Sigmoid / Tanh | Xavier | $\sqrt{2/(n_{in}+n_{out})}$ |
| ReLU | He (Kaiming) | $\sqrt{2/n_{in}}$ |
| SELU | LeCun | $\sqrt{1/n_{in}}$ |

## Learning Resources
- [He et al. 2015 (He init)](https://arxiv.org/abs/1502.01852)
- [Glorot & Bengio 2010 (Xavier init)](http://proceedings.mlr.press/v9/glorot10a.html)

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
