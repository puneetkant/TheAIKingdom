# 3.2.8 Perceptron and Linear Models

Perceptron convergence, SGDClassifier/Regressor, Elastic Net (L1+L2), online learning.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | Perceptron from scratch, weight update rule |
| `working_example2.py` | Perceptron, SGD losses, Elastic Net l1_ratio sweep |
| `working_example.ipynb` | Interactive: Perceptron → SGD losses → Elastic Net CV |

## Quick Reference

```python
from sklearn.linear_model import SGDClassifier, SGDRegressor, ElasticNetCV

# SGD classifier (large-scale / online learning)
clf = SGDClassifier(loss="log_loss", alpha=0.0001, max_iter=1000, random_state=42)

# Elastic Net with cross-validation
enet = ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9, 0.95, 1.0], cv=5)

# l1_ratio=1 → Lasso; l1_ratio=0 → Ridge
```

## Key Equations
- Perceptron update: w ← w + η·(y - ŷ)·x
- Elastic Net: Loss + α[l1·‖w‖₁ + (1-l1)·½‖w‖₂²]

## Learning Resources
- [StatQuest: SGD](https://youtu.be/vMh0zPT0tLI)
- [sklearn SGD](https://scikit-learn.org/stable/modules/sgd.html)
- [sklearn Elastic Net](https://scikit-learn.org/stable/modules/linear_model.html#elastic-net)

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
