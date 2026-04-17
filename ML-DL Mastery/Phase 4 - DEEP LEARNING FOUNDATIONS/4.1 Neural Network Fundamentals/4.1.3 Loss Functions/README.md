# 4.1.3 Loss Functions

MSE, MAE, Binary Cross-Entropy, Softmax CE, Hinge, Huber — formulas, gradients, numerical stability.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | sklearn loss functions, custom loss demo |
| `working_example2.py` | Manual implementations, stability, loss curve plots |
| `working_example.ipynb` | Interactive: loss comparison → softmax CE → visualisation |

## Quick Reference

```python
import numpy as np

# Numerically stable BCE
def bce(y, p, eps=1e-7):
    p = np.clip(p, eps, 1-eps)
    return -np.mean(y*np.log(p) + (1-y)*np.log(1-p))

# Numerically stable softmax cross-entropy
def softmax_ce(true_idx, logits):
    z = logits - logits.max()          # subtract max for stability
    log_sm = z - np.log(np.exp(z).sum())
    return -log_sm[true_idx]
```

## Loss Reference

| Loss | Formula | Use |
|------|---------|-----|
| MSE | $\frac{1}{n}\sum(y-\hat y)^2$ | Regression |
| MAE | $\frac{1}{n}\sum|y-\hat y|$ | Robust regression |
| BCE | $-[y\log p+(1-y)\log(1-p)]$ | Binary classification |
| Softmax CE | $-\log\text{softmax}(z)_y$ | Multiclass |
| Hinge | $\max(0,1-y\cdot\hat y)$ | SVM |
| Huber | MSE/MAE hybrid | Outlier-robust |

## Learning Resources
- [PyTorch loss functions](https://pytorch.org/docs/stable/nn.html#loss-functions)
- [CS231n: Loss functions](https://cs231n.github.io/linear-classify/#loss)

Define functions, use args/kwargs, and document them.

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
