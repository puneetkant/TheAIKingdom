# 3.2.6 Decision Trees

Gini impurity, information gain, depth sweep, feature importances, pruning.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | Manual tree split, Gini calculation |
| `working_example2.py` | DT classifier + regressor, depth sweep, feature importances bar chart |
| `working_example.ipynb` | Interactive: depth sweep → importances bar → regressor RMSE |

## Quick Reference

```python
from sklearn.tree import DecisionTreeClassifier, export_text

dt = DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, random_state=42)
dt.fit(X_train, y_train)

# Feature importances
importances = dt.feature_importances_

# Visualise rules
print(export_text(dt, feature_names=feature_names))
```

## Key Concepts
- **Gini**: 1 - Σ pₖ²   (used by sklearn default)
- **Entropy**: -Σ pₖ log(pₖ)   (information gain)
- Deep trees = overfit; use `max_depth`, `min_samples_leaf`, `ccp_alpha`

## Learning Resources
- [StatQuest: Decision Trees](https://youtu.be/_L39rN6gz7Y)
- [sklearn Decision Trees](https://scikit-learn.org/stable/modules/tree.html)

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
