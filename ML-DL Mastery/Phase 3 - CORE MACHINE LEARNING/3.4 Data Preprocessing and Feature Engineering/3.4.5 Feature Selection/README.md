# 3.4.5 Feature Selection

Filter, wrapper, and embedded feature selection methods. Permutation importance.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | Correlation filter, variance threshold basics |
| `working_example2.py` | SelectKBest, RFE, Lasso embedded, permutation importance |
| `working_example.ipynb` | Interactive: k sweep → RFE → Lasso → permutation importance |

## Quick Reference

```python
from sklearn.feature_selection import SelectKBest, f_regression, RFE, SelectFromModel
from sklearn.inspection import permutation_importance

# Filter (univariate)
pipe = make_pipeline(StandardScaler(), SelectKBest(f_regression, k=4), Ridge())

# Wrapper (recursive)
rfe = RFE(Ridge(), n_features_to_select=4)
rfe.fit(X_train_scaled, y_train)
selected_names = np.array(feature_names)[rfe.support_]

# Embedded (Lasso zero-out)
lasso = make_pipeline(StandardScaler(), Lasso(alpha=0.05))

# Permutation importance (model-agnostic)
result = permutation_importance(model, X_test, y_test, n_repeats=10)
```

## Method Comparison

| Method | Speed | Accounts for interactions |
|--------|-------|--------------------------|
| Filter (f_regression) | Fast | No |
| RFE (wrapper) | Slow | Partially |
| Lasso (embedded) | Medium | No |
| Permutation | Medium | Yes |

## Learning Resources
- [sklearn Feature Selection guide](https://scikit-learn.org/stable/modules/feature_selection.html)
- [Boruta algorithm](https://github.com/scikit-learn-contrib/boruta_py)

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
