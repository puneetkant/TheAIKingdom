# 4.5.1 Attention Fundamentals

Scaled dot-product attention, attention weights heatmap, causal mask for autoregressive models.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | Bahdanau vs dot-product attention comparison |
| `working_example2.py` | Manual Q/K/V attention + weight heatmap + causal mask |
| `working_example.ipynb` | Interactive: implement attention → visualise weights → causal mask |

## Quick Reference

```python
import torch, torch.nn.functional as F

# Scaled dot-product (PyTorch 2.0+)
out = F.scaled_dot_product_attention(Q, K, V, is_causal=False)

# Manual
def attention(Q, K, V):
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(-2, -1) / d_k**0.5
    weights = scores.softmax(dim=-1)
    return weights @ V
```

## Attention Formula

$$\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

## Learning Resources
- [Attention is All You Need (Vaswani 2017)](https://arxiv.org/abs/1706.03762)
- [Illustrated Attention (Jay Alammar)](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)

Inspect attention and transformer architecture.

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
