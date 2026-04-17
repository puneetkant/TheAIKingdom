# 3.2.5 Support Vector Machines (SVM)

Max-margin classifier, soft margin (C), kernel trick (RBF/poly), SVR.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | 2D decision boundary, support vectors visualised |
| `working_example2.py` | LinearSVC C sweep, RBF SVC, SVR regression |
| `working_example.ipynb` | Interactive: LinearSVC → RBF AUC → SVR RMSE |

## Quick Reference

```python
from sklearn.svm import SVC, SVR, LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Classification (always scale!)
pipe = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, gamma="scale"))
pipe.fit(X_train, y_train)

# Regression with epsilon-insensitive loss
svr = make_pipeline(StandardScaler(), SVR(kernel="rbf", C=1.0, epsilon=0.1))
```

## Key Concepts
- **C**: soft-margin penalty — large C → smaller margin, fit training more
- **gamma** (`scale`=1/(n_features·Var)): RBF bandwidth — large gamma → narrow kernel → overfit
- **kernel trick**: K(x,z) = φ(x)·φ(z) without computing φ explicitly

## Learning Resources
- [StatQuest: SVM (Main Ideas)](https://youtu.be/efR1C6CvhmE)
- [sklearn SVM guide](https://scikit-learn.org/stable/modules/svm.html)

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
