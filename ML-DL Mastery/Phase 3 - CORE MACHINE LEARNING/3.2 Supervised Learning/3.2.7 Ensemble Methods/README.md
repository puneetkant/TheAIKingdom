# 3.2.7 Ensemble Methods

Bagging (Random Forest), Boosting (AdaBoost, GradientBoosting), OOB score, feature importances.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | Bagging from scratch, voting classifier |
| `working_example2.py` | RF OOB + importances, GBM lr sweep, AdaBoost AUC |
| `working_example.ipynb` | Interactive: RF n sweep → importances bar → GBM → AdaBoost |

## Quick Reference

```python
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Random Forest (bagging + feature subsampling)
rf = RandomForestRegressor(n_estimators=200, oob_score=True, n_jobs=-1, random_state=42)
rf.fit(X_train, y_train)
oob_rmse = ((y_train - rf.oob_prediction_)**2).mean()**0.5

# Gradient Boosting
gb = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05,
                                max_depth=3, subsample=0.8)
gb.fit(X_train, y_train)
```

## Key Concepts

| Method | Strategy | Variance | Bias |
|--------|----------|----------|------|
| Bagging | Parallel, bootstrap | ↓ | = |
| Boosting | Sequential, reweight | = | ↓ |
| Stacking | Meta-learner | ↓ | ↓ |

## Learning Resources
- [StatQuest: Random Forest](https://youtu.be/J4Wdy0Wc_xQ)
- [StatQuest: Gradient Boost](https://youtu.be/3CC4N4z3GJc)
- [XGBoost docs](https://xgboost.readthedocs.io/)

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
