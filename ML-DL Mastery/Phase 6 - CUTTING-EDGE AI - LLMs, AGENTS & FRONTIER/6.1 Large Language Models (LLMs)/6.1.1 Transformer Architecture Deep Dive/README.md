# 6.1.1 Transformer Architecture Deep Dive

Attention, multi-head attention, positional encoding, layer norm, FFN, encoder-decoder.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | Full transformer block |
| `working_example2.py` | SDPA + multi-head attention + sinusoidal PE from scratch |
| `working_example.ipynb` | Interactive: attention heatmap + PE visualisation |

## Quick Reference

```python
import numpy as np

# Scaled Dot-Product Attention
def sdpa(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    if mask is not None: scores[~mask] = -1e9
    return softmax(scores) @ V

# Sinusoidal Positional Encoding
def pe(T, d):
    pos = np.arange(T)[:, None]
    div = np.exp(np.arange(0, d, 2) * (-np.log(10000) / d))
    P = np.zeros((T, d))
    P[:, 0::2] = np.sin(pos * div)
    P[:, 1::2] = np.cos(pos * div)
    return P

# Layer Norm
def layer_norm(x, eps=1e-5):
    return (x - x.mean(-1, keepdims=True)) / (x.std(-1, keepdims=True) + eps)
```

## Transformer Components

| Component | Role |
|-----------|------|
| Attention | Contextual token mixing |
| FFN | Per-token non-linear transform |
| Layer norm | Stabilise activations |
| Positional encoding | Inject position info |

## Learning Resources
- [Attention is All You Need](https://arxiv.org/abs/1706.03762)
- [Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)

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
