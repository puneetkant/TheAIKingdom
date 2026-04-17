# 5.5.3 Deep Learning for Time Series

LSTM, Temporal CNN, Transformer, PatchTST, N-HiTS, sliding window, multi-step forecasting.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | LSTM-based forecasting (PyTorch) |
| `working_example2.py` | MLP sliding-window regression on synthetic series (sklearn) |
| `working_example.ipynb` | Interactive: sliding window + MLP forecast |

## Quick Reference

```python
# Sliding window (series → supervised)
X = [ts[i:i+look_back] for i in range(len(ts)-look_back-horizon)]
y = [ts[i+look_back:i+look_back+horizon] for i in range(len(ts)-look_back-horizon)]

# PyTorch LSTM
class TSForecaster(nn.Module):
    def __init__(self, in_dim=1, hidden=64, horizon=1):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, num_layers=2, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden, horizon)
    def forward(self, x):                    # (B, T, F)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1])           # (B, horizon)
```

## Model Comparison

| Model | Strengths | Horizon |
|-------|-----------|---------|
| LSTM | Sequential patterns | Short-mid |
| TCN | Parallel, longer memory | Short-long |
| Transformer | Global context | Long |
| PatchTST | Patched attention | Long |
| N-HiTS | Hierarchical interpolation | Long |

## Learning Resources
- [Time-series-library](https://github.com/thuml/Time-Series-Library)
- [N-BEATS/N-HiTS paper](https://arxiv.org/abs/2201.12886)

Forecast or analyze temporal data.

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
