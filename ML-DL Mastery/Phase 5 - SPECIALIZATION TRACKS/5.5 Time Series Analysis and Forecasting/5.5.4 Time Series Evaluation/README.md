# 5.5.4 Time Series Evaluation

MAE, RMSE, MAPE, sMAPE, MASE, backtesting, walk-forward validation, prediction intervals.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | Walk-forward backtesting loop |
| `working_example2.py` | Baseline methods (naive/seasonal/drift) vs metrics table |
| `working_example.ipynb` | Interactive: naive forecast + metric computation |

## Quick Reference

```python
import numpy as np

def mae(y, yh):   return np.mean(np.abs(y - yh))
def rmse(y, yh):  return np.sqrt(np.mean((y - yh)**2))
def mape(y, yh):  return np.mean(np.abs((y-yh) / np.abs(y))) * 100
def smape(y, yh): return 200 * np.mean(np.abs(y-yh) / (np.abs(y)+np.abs(yh)))

# MASE (Mean Absolute Scaled Error) — scale-free
def mase(y, yh, y_train, period=1):
    naive_mae = np.mean(np.abs(np.diff(y_train, n=period)))
    return mae(y, yh) / naive_mae

# Walk-forward validation
for i in range(n_splits):
    train_end = min_train + i * step
    X_tr, y_tr = X[:train_end], y[:train_end]
    X_te, y_te = X[train_end:train_end+step], y[train_end:train_end+step]
    model.fit(X_tr, y_tr); preds = model.predict(X_te)
```

## Metric Guide

| Metric | Scale-free | Handles 0 | Penalises large |
|--------|-----------|-----------|----------------|
| MAE | No | Yes | No |
| RMSE | No | Yes | Yes |
| MAPE | Yes | No | No |
| sMAPE | Yes | Partial | No |
| MASE | Yes | Yes | No |

## Learning Resources
- [FPP3 Forecast accuracy](https://otexts.com/fpp3/accuracy.html)

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
