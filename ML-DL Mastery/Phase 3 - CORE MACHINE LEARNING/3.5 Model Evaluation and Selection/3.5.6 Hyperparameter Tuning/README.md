# 3.5.6 Hyperparameter Tuning

GridSearchCV, RandomizedSearchCV, HalvingRandomSearchCV, Bayesian optimization.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | Manual grid loop, learning curves |
| `working_example2.py` | GridSearch (Ridge), RandomSearch (RF), HalvingSearch |
| `working_example.ipynb` | Interactive: grid → random → halving search |

## Quick Reference

```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from scipy.stats import randint, uniform

# Grid search (small space)
pipe = Pipeline([("sc", StandardScaler()), ("m", Ridge())])
gs = GridSearchCV(pipe, {"m__alpha": [0.01, 0.1, 1, 10]},
                  cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1)
gs.fit(X_train, y_train)
print(gs.best_params_, -gs.best_score_)

# Random search (large space)
param_dist = {"m__n_estimators": randint(50, 500),
              "m__max_features": uniform(0.3, 0.7)}
rs = RandomizedSearchCV(pipe, param_dist, n_iter=50, cv=5, random_state=42)
```

## Search Strategy Comparison

| Method | Evaluations | Best for |
|--------|-------------|---------|
| Grid | n₁×n₂×… | Small space, exhaustive |
| Random | n_iter | Large/continuous space |
| Halving | ~log(n) passes | Efficient large search |
| Bayesian (Optuna) | Adaptive | Expensive models |

## Learning Resources
- [Optuna (Bayesian)](https://optuna.org/)
- [sklearn tuning guide](https://scikit-learn.org/stable/modules/grid_search.html)

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
