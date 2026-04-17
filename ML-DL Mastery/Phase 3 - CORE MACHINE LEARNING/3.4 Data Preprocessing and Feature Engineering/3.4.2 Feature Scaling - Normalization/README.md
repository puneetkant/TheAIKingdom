# 3.4.2 Feature Scaling & Normalization

StandardScaler, MinMaxScaler, RobustScaler — when and why to scale.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | Manual z-score / min-max formulas |
| `working_example2.py` | Feature range table, RMSE comparison across scalers and models |
| `working_example.ipynb` | Interactive: feature ranges → Ridge scaling → KNN scaling |

## Quick Reference

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# Fit on train, transform train+test (no leakage)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)   # NOT fit_transform!

# Or use Pipeline (handles leakage automatically)
from sklearn.pipeline import make_pipeline
pipe = make_pipeline(StandardScaler(), Ridge(1.0))
```

## When to Use Which Scaler

| Scaler | Formula | Best for |
|--------|---------|---------|
| Standard | (x-μ)/σ | Gaussian-ish, algorithms assuming normality |
| MinMax | (x-min)/(max-min) | Neural nets, bounded [0,1] range needed |
| Robust | (x-Q2)/(Q3-Q1) | Data with outliers |

## Models That Need Scaling
- KNN, SVM, Ridge/Lasso, PCA, Neural nets → **YES**
- Decision Trees, Random Forest, Gradient Boosting → **NO**

## Learning Resources
- [sklearn Preprocessing guide](https://scikit-learn.org/stable/modules/preprocessing.html)

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
