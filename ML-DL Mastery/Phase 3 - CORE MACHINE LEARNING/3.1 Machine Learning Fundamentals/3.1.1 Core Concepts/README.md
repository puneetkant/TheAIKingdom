# 3.1.1 Core Concepts

Supervised vs unsupervised vs reinforcement learning, train/val/test splits, bias-variance tradeoff, learning curves.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | Basic sklearn pipeline, fit/predict pattern |
| `working_example2.py` | Bias-variance demo (poly degree), learning curves, Cal Housing splits |
| `working_example.ipynb` | Interactive: splits → degree sweep → learning curves → test eval |

## Quick Reference

```python
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

# 70/15/15 split
X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=0.15)
X_train, X_val, y_train, y_val = train_test_split(X_tmp, y_tmp, test_size=0.15/0.85)

# Pipeline (prevents data leakage)
pipe = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

# Learning curves
sizes, train_scores, val_scores = learning_curve(pipe, X_train, y_train, cv=5)
```

## Key Concepts

| Concept | High Bias | High Variance |
|---------|-----------|---------------|
| Train error | High | Low |
| Val error | High | High (gap) |
| Fix | More complex model | More data / regularize |

## Learning Resources
- [StatQuest: Machine Learning Fundamentals](https://youtu.be/Gv9_4yMHFhI)
- [sklearn User Guide](https://scikit-learn.org/stable/user_guide.html)
- **Book:** *Hands-On Machine Learning* (Aurélien Géron) Ch. 1-2

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
