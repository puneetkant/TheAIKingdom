# 3.2.3 K-Nearest Neighbors (KNN)

Instance-based learning, k sweep, distance weighting, curse of dimensionality.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | KNN from scratch (Euclidean distance, majority vote) |
| `working_example2.py` | sklearn KNN regression/classification, k sweep, curse of dimensionality |
| `working_example.ipynb` | Interactive: regression k sweep → classification AUC → CoD table |

## Quick Reference

```python
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Always scale! KNN is distance-based
pipe = make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=5, weights="distance"))
pipe.fit(X_train, y_train)

# Choose k via CV
from sklearn.model_selection import GridSearchCV
gs = GridSearchCV(pipe, {"kneighborsregressor__n_neighbors": [3,5,10,20]}, cv=5)
```

## Key Points
- **Scale features** — mandatory (distance-based)
- **Large k** → high bias, low variance; **small k** → low bias, high variance
- **Curse of dimensionality** → distances become uniform in high d → use feature selection/PCA first

## Learning Resources
- [StatQuest: KNN](https://youtu.be/HVXime0nQeI)
- [sklearn KNeighborsRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html)

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
