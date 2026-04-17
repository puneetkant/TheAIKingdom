# 3.4.6 Handling Imbalanced Data

class_weight='balanced', threshold tuning, StratifiedKFold, PR-AUC over ROC-AUC.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | Oversampling, undersampling basics |
| `working_example2.py` | class_weight sweep, threshold tuning, stratified CV |
| `working_example.ipynb` | Interactive: class distribution → balanced weight → threshold → PR-AUC |

## Quick Reference

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import average_precision_score

# class_weight handles imbalance without resampling
pipe = make_pipeline(
    StandardScaler(),
    LogisticRegression(class_weight="balanced", max_iter=1000)
)

# Always use stratified splits with imbalanced data
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipe, X, y, cv=cv, scoring="average_precision")

# Threshold tuning
probs = pipe.predict_proba(X_test)[:, 1]
preds = (probs >= 0.3).astype(int)  # lower threshold = higher recall
```

## Key Metrics for Imbalanced Data

| Metric | Notes |
|--------|-------|
| ROC-AUC | Optimistic when negatives dominate |
| **PR-AUC** | Preferred — focuses on minority class |
| F1-score | Harmonic mean of precision and recall |
| Recall | Miss rate for minority class |

## Learning Resources
- [imbalanced-learn (SMOTE)](https://imbalanced-learn.org/)
- [sklearn class_weight docs](https://scikit-learn.org/stable/modules/svm.html#unbalanced-problems)

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
