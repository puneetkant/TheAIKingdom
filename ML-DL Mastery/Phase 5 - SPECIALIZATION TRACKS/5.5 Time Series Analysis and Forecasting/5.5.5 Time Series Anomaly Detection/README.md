# 5.5.5 Time Series Anomaly Detection

Z-score, IQR, CUSUM, Isolation Forest, LSTM reconstruction error, point/contextual/collective anomalies.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | CUSUM and reconstruction-based detection |
| `working_example2.py` | Rolling Z-score + IQR detection with precision/recall |
| `working_example.ipynb` | Interactive: injected anomaly detection + visualisation |

## Quick Reference

```python
# Rolling Z-score
window = 30
for i in range(window, len(ts)):
    mu, sigma = ts[i-window:i].mean(), ts[i-window:i].std()
    z_score[i] = abs(ts[i] - mu) / (sigma + 1e-8)
anomalies = z_score > 3.0

# IQR method
Q1, Q3 = np.percentile(ts, 25), np.percentile(ts, 75)
IQR = Q3 - Q1
anomalies = (ts < Q1 - 1.5*IQR) | (ts > Q3 + 1.5*IQR)

# Isolation Forest
from sklearn.ensemble import IsolationForest
clf = IsolationForest(contamination=0.05)
anomalies = clf.fit_predict(ts.reshape(-1, 1)) == -1

# LSTM reconstruction error
recon_err = np.mean((x - model(x))**2, axis=-1)
anomalies = recon_err > threshold
```

## Method Comparison

| Method | Type | Strengths | Weaknesses |
|--------|------|-----------|-----------|
| Z-score | Statistical | Simple | Assumes Gaussian |
| IQR | Statistical | Robust | Global only |
| CUSUM | Statistical | Detects shifts | Threshold tuning |
| Isolation Forest | ML | Multivariate | Hyperparameters |
| LSTM recon | DL | Contextual | Requires training |

## Learning Resources
- [Anomalib library](https://github.com/openvinotoolkit/anomalib)
- [TSAD benchmark](https://arxiv.org/abs/2209.13055)

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
