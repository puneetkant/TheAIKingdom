# 3.5.1 Classification Metrics

Confusion matrix, precision, recall, F1, ROC-AUC, PR-AUC, MCC.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | Manual metric formulas from scratch |
| `working_example2.py` | classification_report, ROC/PR curve plots vs LR and RF |
| `working_example.ipynb` | Interactive: report → ROC-AUC → PR-AUC → curves |

## Quick Reference

```python
from sklearn.metrics import (classification_report, roc_auc_score,
                              average_precision_score, confusion_matrix)

# Full report
print(classification_report(y_test, y_pred, target_names=["Neg", "Pos"]))

# Probability-based metrics
probs = model.predict_proba(X_test)[:, 1]
roc = roc_auc_score(y_test, probs)    # Area under ROC curve
pr  = average_precision_score(y_test, probs)  # Area under PR curve
```

## Metric Formulas

$$\text{Precision} = \frac{TP}{TP+FP} \qquad \text{Recall} = \frac{TP}{TP+FN}$$
$$F_1 = 2 \cdot \frac{P \cdot R}{P+R}$$

## When to Use What

| Metric | Use when |
|--------|---------|
| F1 | Balanced class importance |
| PR-AUC | Imbalanced, minority class matters |
| ROC-AUC | Class distribution doesn't matter |

## Learning Resources
- [sklearn metrics guide](https://scikit-learn.org/stable/modules/model_evaluation.html)

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
