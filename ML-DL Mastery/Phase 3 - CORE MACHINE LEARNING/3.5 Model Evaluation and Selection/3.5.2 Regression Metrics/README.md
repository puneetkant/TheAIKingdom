# 3.5.2 Regression Metrics

MAE, MSE, RMSE, R², MedAE, MAPE — when to use each, residual analysis.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | Manual metric implementations |
| `working_example2.py` | Model comparison table, residual scatter + histogram |
| `working_example.ipynb` | Interactive: metrics → model comparison → residual plots |

## Quick Reference

```python
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                              r2_score, median_absolute_error)

mae  = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred)**0.5
r2   = r2_score(y_test, y_pred)

# MAPE (beware division by zero)
mape = np.abs((y_test - y_pred) / y_test).mean() * 100
```

## Metric Properties

$$\text{RMSE} = \sqrt{\frac{1}{n}\sum(y_i - \hat{y}_i)^2}$$
$$R^2 = 1 - \frac{SS_{res}}{SS_{tot}}$$

| Metric | Scale | Outlier Sensitive | Notes |
|--------|-------|-------------------|-------|
| MAE | Same as y | No | Robust |
| RMSE | Same as y | Yes | Penalizes large errors |
| R² | Dimensionless | Yes | 1=perfect, 0=baseline |
| MedAE | Same as y | No | Very robust |

## Learning Resources
- [sklearn regression metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics)

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
