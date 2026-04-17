# 4.1.4 Forward Propagation

Layer-by-layer matrix math: z = XW + b, activation, shape tracking.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | Single neuron forward pass, manual computation |
| `working_example2.py` | MLP class (numpy), verbose forward pass with shapes |
| `working_example.ipynb` | Interactive: MLP forward → shape walkthrough → predictions |

## Quick Reference

```python
import numpy as np

# One layer
z = X @ W + b          # (batch, n_in) @ (n_in, n_out) → (batch, n_out)
a = np.maximum(0, z)   # ReLU activation

# Full MLP forward
def forward(X, params):
    a = X
    for i, (W, b) in enumerate(params):
        z = a @ W + b
        a = sigmoid(z) if i == len(params)-1 else relu(z)
    return a
```

## Shape Rules

| Operation | Shape |
|-----------|-------|
| Input X | (batch, n_features) |
| Weight W | (n_in, n_out) |
| Bias b | (n_out,) |
| z = XW+b | (batch, n_out) |
| a = act(z) | (batch, n_out) |

## Learning Resources
- [CS231n: Neural Networks Part 1](https://cs231n.github.io/neural-networks-1/)

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
