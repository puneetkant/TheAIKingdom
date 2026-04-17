# 3.2.2 Logistic Regression

Binary classification via sigmoid, log-loss, ROC-AUC, L1/L2 regularisation (C parameter).

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | Sigmoid implementation, decision boundary |
| `working_example2.py` | Cal Housing binary, ROC curve, regularisation sweep |
| `working_example.ipynb` | Interactive: fit → metrics report → ROC → C sweep |

## Quick Reference

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report

pipe = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))
pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)
y_prob = pipe.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred))
auc = roc_auc_score(y_test, y_prob)
```

## Key Equations

- σ(z) = 1/(1+e⁻ᶻ)
- Loss = -[y log(p) + (1-y) log(1-p)]  (binary cross-entropy)
- C = 1/λ — smaller C = stronger regularisation

## Learning Resources
- [StatQuest: Logistic Regression](https://youtu.be/yIYKR4sgzI8)
- [sklearn LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

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
