# 3.2.1 Linear Regression

OLS, Ridge/Lasso regularisation, gradient descent from scratch, residual analysis, feature coefficients.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | Closed-form normal equations, sklearn baseline |
| `working_example2.py` | OLS + Ridge sweep + GD from scratch + residual plots |
| `working_example.ipynb` | Interactive: OLS → Ridge sweep → GD convergence → residuals |

## Quick Reference

```python
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# OLS
pipe = make_pipeline(StandardScaler(), LinearRegression())
pipe.fit(X_train, y_train)

# Ridge with CV
pipe_cv = make_pipeline(StandardScaler(), RidgeCV(alphas=[0.1,1,10,100], cv=5))
pipe_cv.fit(X_train, y_train)
best_alpha = pipe_cv.named_steps['ridgecv'].alpha_
```

## Key Concepts

- **Normal equation**: w* = (XᵀX)⁻¹Xᵀy — exact but O(n·d²), unstable for high d
- **Ridge**: adds λ‖w‖² → shrinks coefficients, always invertible
- **Gradient descent**: w ← w - α·(2/n)·Xᵀ(Xw-y)

## Learning Resources
- [StatQuest: Linear Regression](https://youtu.be/nk2CQITm_eo)
- [sklearn LinearRegression](https://scikit-learn.org/stable/modules/linear_model.html)

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
