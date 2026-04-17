# 5.8.6 Monitoring and Observability

Data drift (KS, PSI), prediction drift, latency SLA, alerting, dashboards.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | Prometheus + Grafana setup |
| `working_example2.py` | KS + PSI drift monitoring with weekly alerts |
| `working_example.ipynb` | Interactive: KS test + PSI + distribution shift |

## Quick Reference

```python
import numpy as np

# Kolmogorov-Smirnov test
def ks_drift(ref, prod):
    vals = np.sort(np.concatenate([ref, prod]))
    cdf_r = np.searchsorted(np.sort(ref), vals, 'right') / len(ref)
    cdf_p = np.searchsorted(np.sort(prod), vals, 'right') / len(prod)
    stat = np.max(np.abs(cdf_r - cdf_p))
    return stat > 1.36 * np.sqrt((len(ref)+len(prod))/(len(ref)*len(prod)))

# Population Stability Index
def psi(ref, prod, bins=10):
    rh, edges = np.histogram(ref, bins=bins)
    ph, _ = np.histogram(prod, bins=edges)
    rp = (rh + 1e-8) / rh.sum(); pp = (ph + 1e-8) / ph.sum()
    return np.sum((pp - rp) * np.log(pp / rp))
# PSI < 0.1: no drift | 0.1-0.2: moderate | >0.2: significant

# Evidently AI
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=ref_df, current_data=prod_df)
```

## Monitoring Signals

| Signal | Tool | Alert trigger |
|--------|------|--------------|
| Data drift | KS, PSI, Evidently | PSI > 0.2 |
| Prediction drift | Rolling distribution | Shift > 2σ |
| Latency | Prometheus | p99 > SLA |
| Error rate | Prometheus | > 1% |
| Accuracy | Ground-truth labels | Drop > 5% |

## Learning Resources
- [Evidently AI](https://www.evidentlyai.com/)
- [Prometheus](https://prometheus.io/docs/introduction/)

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
