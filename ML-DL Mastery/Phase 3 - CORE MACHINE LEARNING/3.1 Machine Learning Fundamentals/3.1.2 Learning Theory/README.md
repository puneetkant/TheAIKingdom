# 3.1.2 Learning Theory

PAC learning, Hoeffding generalisation bound, ERM, VC dimension, regularisation (Ridge/Lasso), cross-validation.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | ERM demo, overfitting illustration |
| `working_example2.py` | Hoeffding bound table, regularisation path plot, k-fold CV |
| `working_example.ipynb` | Interactive: Hoeffding → Ridge path → Ridge semilog → k-fold CV |

## Quick Reference

```python
import math, numpy as np

# Hoeffding generalisation bound (95% confidence)
bound = math.sqrt(math.log(2/0.05) / (2*n))

# Regularisation path
from sklearn.linear_model import RidgeCV
ridge_cv = RidgeCV(alphas=np.logspace(-3,3,30), cv=5)
ridge_cv.fit(X_train, y_train)
best_alpha = ridge_cv.alpha_

# Cross-validation
from sklearn.model_selection import cross_val_score
scores = cross_val_score(pipe, X, y, cv=5, scoring="neg_mean_squared_error")
mse = -scores.mean()
```

## Key Inequalities

| Bound | Expression |
|-------|-----------|
| Hoeffding | ε ≤ √(log(2/δ)/(2n)) |
| Union bound | P(any of k events) ≤ Σ P(event_i) |
| VC bound | ε ≤ O(√(d log(n/d) + log(1/δ))/n) |

## Learning Resources
- [Understanding ML (Shalev-Shwartz & Ben-David) — free PDF](https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/)
- [StatQuest: Ridge/Lasso](https://youtu.be/Q81RR3yKn30)

Implement basic ML concepts and theory examples.

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
