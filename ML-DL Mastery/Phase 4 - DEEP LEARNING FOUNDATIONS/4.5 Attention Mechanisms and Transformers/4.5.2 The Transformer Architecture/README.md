# 4.5.2 The Transformer Architecture

Multi-head attention + Add&Norm + FFN encoder block, positional encoding, full architecture.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | Positional encoding visualisation |
| `working_example2.py` | Numpy transformer encoder block: MHA → LayerNorm → FFN |
| `working_example.ipynb` | Interactive: MHA → Add&Norm → FFN → verify shapes |

## Quick Reference

```python
import torch.nn as nn

encoder_layer = nn.TransformerEncoderLayer(
    d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1, batch_first=True)
encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
out = encoder(src)   # (B, T, 512)
```

## Transformer Block Components

| Component | Role |
|-----------|------|
| Multi-Head Attention | Attend to all positions in parallel |
| Add & LayerNorm | Residual connection + normalisation |
| Feed-Forward Network | 2-layer MLP per position |
| Positional Encoding | Inject sequence order (sin/cos) |

## Learning Resources
- [Attention is All You Need](https://arxiv.org/abs/1706.03762)
- [The Illustrated Transformer (Jay Alammar)](https://jalammar.github.io/illustrated-transformer/)

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
