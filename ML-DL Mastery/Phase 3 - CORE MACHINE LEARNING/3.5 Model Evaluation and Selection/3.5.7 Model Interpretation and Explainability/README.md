# 3.5.7 Model Interpretation & Explainability

Permutation importance, Partial Dependence Plots (PDP), SHAP, LIME.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | Feature importances (tree-based), linear model coefficients |
| `working_example2.py` | Permutation importance, PDP, manual leave-one-out attribution |
| `working_example.ipynb` | Interactive: permutation rank → PDP plots → leave-one-out |

## Quick Reference

```python
from sklearn.inspection import permutation_importance, PartialDependenceDisplay

# Permutation importance (model-agnostic, use test set)
result = permutation_importance(model, X_test, y_test,
                                 n_repeats=10, random_state=42)
# result.importances_mean[i] = RMSE increase when feature i is shuffled

# Partial Dependence Plot
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
PartialDependenceDisplay.from_estimator(model, X_train, [0, 1],
                                         feature_names=feature_names, ax=ax)

# SHAP (requires: pip install shap)
import shap
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, feature_names=feature_names)
```

## XAI Methods

| Method | Model-agnostic | Local | Global |
|--------|----------------|-------|--------|
| Permutation importance | Yes | No | Yes |
| PDP | Yes | No | Yes |
| SHAP | Yes | Yes | Yes |
| LIME | Yes | Yes | No |

## Learning Resources
- [SHAP library](https://shap.readthedocs.io/)
- [Interpretable ML Book](https://christophm.github.io/interpretable-ml-book/)

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
