# 4.1.2 Activation Functions

Sigmoid, Tanh, ReLU, Leaky ReLU, ELU, GELU, Softmax — values, derivatives, dead neuron problem.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | Activation comparisons, vanishing gradient demo |
| `working_example2.py` | All activations + gradients table, 6-panel matplotlib plot |
| `working_example.ipynb` | Interactive: values table → 4-panel plot → softmax |

## Quick Reference

```python
import numpy as np

sigmoid  = lambda x: 1 / (1 + np.exp(-x))
relu     = lambda x: np.maximum(0, x)
leaky    = lambda x: np.where(x > 0, x, 0.01 * x)
gelu     = lambda x: 0.5*x*(1 + np.tanh(np.sqrt(2/np.pi)*(x + 0.044715*x**3)))

def softmax(z):
    e = np.exp(z - z.max())   # numerically stable
    return e / e.sum()
```

## Activation Cheat Sheet

| Function | Range | Gradient | Use |
|----------|-------|---------|-----|
| Sigmoid | (0,1) | Vanishes | Output (binary) |
| Tanh | (-1,1) | Vanishes | RNN hidden |
| ReLU | [0,∞) | Dead neurons | Hidden (default) |
| Leaky ReLU | (-∞,∞) | No dead | Hidden |
| GELU | (-∞,∞) | Smooth | Transformers |
| Softmax | (0,1) per class | — | Output (multiclass) |

## Learning Resources
- [Activation Functions overview](https://arxiv.org/abs/1811.03378)
- [Dead ReLU discussion](https://datascience.stackexchange.com/questions/5706/what-is-the-dying-relu-problem-in-neural-networks)

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
