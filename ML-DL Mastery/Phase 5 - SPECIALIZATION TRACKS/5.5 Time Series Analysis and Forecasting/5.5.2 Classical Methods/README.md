# 5.5.2 Classical Methods

ARIMA, SARIMA, ETS, Holt-Winters, exponential smoothing.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | ARIMA with statsmodels |
| `working_example2.py` | AR(1) simulation + SES + Holt's linear smoothing (numpy) |
| `working_example.ipynb` | Interactive: AR(1) + exponential smoothing |

## Quick Reference

```python
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# ARIMA(p, d, q)
model = ARIMA(ts, order=(2, 1, 2))
result = model.fit()
forecast = result.forecast(steps=10)

# SARIMA(p,d,q)(P,D,Q,m) — seasonal
from statsmodels.tsa.statespace.sarimax import SARIMAX
model = SARIMAX(ts, order=(1,1,1), seasonal_order=(1,1,1,12))
result = model.fit()

# Holt-Winters (triple exponential smoothing)
model = ExponentialSmoothing(ts, trend="add", seasonal="add", periods=12)
result = model.fit()
```

## Model Selection Guide

| Pattern | Suggested model |
|---------|----------------|
| No trend/seasonal | AR/MA, ARIMA(p,0,q) |
| Trend only | ARIMA(p,1,q), Holt linear |
| Trend + seasonal | SARIMA, Holt-Winters |
| Complex | Prophet, TBATS |

## Learning Resources
- [FPP3 ARIMA chapter](https://otexts.com/fpp3/arima.html)
- [Statsmodels ARIMA](https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html)

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
