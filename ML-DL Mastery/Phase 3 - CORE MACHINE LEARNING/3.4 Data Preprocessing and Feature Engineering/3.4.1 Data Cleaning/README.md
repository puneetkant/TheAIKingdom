# 3.4.1 Data Cleaning

Missing value imputation (mean/median/KNN), outlier detection (IQR, z-score), duplicate removal.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | Pandas-based cleaning: dropna, fillna, duplicates |
| `working_example2.py` | SimpleImputer/KNNImputer RMSE comparison, IQR outlier removal |
| `working_example.ipynb` | Interactive: inject missings → imputation sweep → IQR removal |

## Quick Reference

```python
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import make_pipeline

# Mean imputation (simple, fast)
pipe = make_pipeline(SimpleImputer(strategy="mean"), StandardScaler(), model)

# KNN imputation (better, slower)
pipe = make_pipeline(KNNImputer(n_neighbors=5), StandardScaler(), model)

# IQR outlier removal
Q1, Q3 = np.percentile(X, 25, axis=0), np.percentile(X, 75, axis=0)
mask = ((X >= Q1 - 1.5*(Q3-Q1)) & (X <= Q3 + 1.5*(Q3-Q1))).all(axis=1)
X_clean, y_clean = X[mask], y[mask]
```

## Key Rules
- **Always fit imputer on training data only** (no leakage)
- **Use pipeline** to prevent test set contamination
- **KNNImputer** works best when features are correlated

## Learning Resources
- [sklearn Imputation guide](https://scikit-learn.org/stable/modules/impute.html)
- [Pandas data cleaning docs](https://pandas.pydata.org/docs/user_guide/missing_data.html)

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
