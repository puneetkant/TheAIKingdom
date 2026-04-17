# 4.4.2 Gated Architectures (LSTM & GRU)

LSTM forget/input/output gates + cell state; GRU reset/update gates — gradient highway.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | LSTM cell trace with gate values |
| `working_example2.py` | LSTM and GRU from scratch, cell state heatmap |
| `working_example.ipynb` | Interactive: GRU from scratch → hidden state heatmap |

## Quick Reference

```python
import torch.nn as nn

lstm = nn.LSTM(input_size=1, hidden_size=64, num_layers=2, batch_first=True)
gru  = nn.GRU(input_size=1, hidden_size=64, batch_first=True)

out, (h_n, c_n) = lstm(x)   # c_n is cell state
out, h_n        = gru(x)
```

## LSTM Gates

| Gate | Formula | Purpose |
|------|---------|---------|
| Forget f | σ(W_f·[h,x]+b) | How much of c_{t-1} to keep |
| Input i | σ(W_i·[h,x]+b) | What new info to store |
| Cell g | tanh(W_g·[h,x]+b) | Candidate values |
| Output o | σ(W_o·[h,x]+b) | What to expose as h_t |

$$c_t = f \odot c_{t-1} + i \odot g \quad h_t = o \odot \tanh(c_t)$$

## Learning Resources
- [LSTM (Hochreiter 1997)](https://www.bioinf.jku.at/publications/older/2604.pdf)
- [Illustrated LSTM/GRU (colah)](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)

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
