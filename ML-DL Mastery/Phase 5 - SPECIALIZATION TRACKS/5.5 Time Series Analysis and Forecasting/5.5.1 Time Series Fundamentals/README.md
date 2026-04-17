# 5.5.1 Time Series Fundamentals

Stationarity, ACF/PACF, decomposition, seasonality, trend, white noise.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | ACF/PACF plots with statsmodels |
| `working_example2.py` | Synthetic series decomposition + ACF (numpy-only) |
| `working_example.ipynb` | Interactive: series + trend + ACF bar chart |

## Quick Reference

```python
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose

# Decomposition
result = seasonal_decompose(ts, model="additive", period=12)
result.plot()

# ACF / PACF
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plot_acf(ts, lags=40)
plot_pacf(ts, lags=40)

# ADF stationarity test
from statsmodels.tsa.stattools import adfuller
stat, p_value, *_ = adfuller(ts)
print("p-value:", p_value)   # < 0.05 → stationary

# Differencing to achieve stationarity
ts_diff = np.diff(ts, n=1)   # first difference
```

## Key Concepts

| Concept | Definition |
|---------|-----------|
| Stationarity | Constant mean/variance over time |
| ACF | Correlation with lagged version |
| PACF | Partial (removes indirect) lag correlation |
| Decomposition | Trend + Seasonal + Residual |

## Learning Resources
- [Forecasting: Principles and Practice (Hyndman)](https://otexts.com/fpp3/)
- [statsmodels time series](https://www.statsmodels.org/stable/tsa.html)

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
