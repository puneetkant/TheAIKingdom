# 3.4.4 Feature Engineering

Polynomial features, log/sqrt transforms, interaction terms, domain-specific derived features.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | Binning, interaction terms basics |
| `working_example2.py` | PolynomialFeatures d=2, log1p, domain feature derivation |
| `working_example.ipynb` | Interactive: baseline → poly → log → domain features |

## Quick Reference

```python
from sklearn.preprocessing import PolynomialFeatures, FunctionTransformer
import numpy as np

# Polynomial + interaction features
pipe = make_pipeline(
    StandardScaler(),
    PolynomialFeatures(degree=2, include_bias=False),
    StandardScaler(),   # re-scale after expansion
    Ridge(1.0)
)

# Log transform (safe for non-negative features)
log_fn = FunctionTransformer(lambda X: np.log1p(np.clip(X, 0, None)), validate=True)

# Domain features for Cal Housing
rooms_per_hh  = X[:, 2] / (X[:, 5] + 1)  # AveRooms / HouseHold
beds_per_room = X[:, 3] / (X[:, 2] + 1)
pop_per_hh    = X[:, 4] / (X[:, 5] + 1)
```

## Feature Engineering Ideas

| Raw → Derived | Rationale |
|---------------|-----------|
| log(price) | Right-skewed targets |
| rooms/household | Crowding proxy |
| age² | Non-linear depreciation |
| lat × lon | Spatial interaction |

## Learning Resources
- [Feature Engineering for ML (book)](https://www.oreilly.com/library/view/feature-engineering-for/9781491953235/)
- [sklearn PolynomialFeatures](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html)

Clean and transform data for modeling.

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
