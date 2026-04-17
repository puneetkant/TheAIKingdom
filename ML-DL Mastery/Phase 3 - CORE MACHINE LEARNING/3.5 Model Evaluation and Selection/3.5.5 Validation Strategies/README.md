# 3.5.5 Validation Strategies

KFold, StratifiedKFold, RepeatedKFold, TimeSeriesSplit, nested CV, train vs test score gap.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | Hold-out vs KFold comparison |
| `working_example2.py` | Strategy sweep, TimeSeriesSplit, train+test gap |
| `working_example.ipynb` | Interactive: KFold sweep → TimeSeriesSplit → cross_validate gap |

## Quick Reference

```python
from sklearn.model_selection import (KFold, StratifiedKFold, RepeatedKFold,
                                     TimeSeriesSplit, cross_validate)

# Standard
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# For classification (maintains class ratio)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# More stable estimate
cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)

# Time series (no future leakage)
cv = TimeSeriesSplit(n_splits=5)

# Train + test scores
results = cross_validate(pipe, X, y, cv=cv,
                          scoring="neg_rmse",
                          return_train_score=True)
```

## Strategy Selection Guide

| Data type | Strategy |
|-----------|---------|
| IID, balanced | KFold |
| IID, imbalanced | StratifiedKFold |
| Small dataset | RepeatedKFold |
| Time series | TimeSeriesSplit |

## Learning Resources
- [sklearn cross-validation guide](https://scikit-learn.org/stable/modules/cross_validation.html)

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
