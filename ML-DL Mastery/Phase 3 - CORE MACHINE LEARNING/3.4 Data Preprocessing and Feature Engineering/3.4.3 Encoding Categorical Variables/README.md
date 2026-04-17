# 3.4.3 Encoding Categorical Variables

One-hot encoding, ordinal encoding, target encoding, ColumnTransformer.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | pd.get_dummies, LabelEncoder basics |
| `working_example2.py` | Manual OHE, ColumnTransformer pipeline, target encoding |
| `working_example.ipynb` | Interactive: OHE → ColumnTransformer + Ridge → target encoding |

## Quick Reference

```python
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
])
pipe = Pipeline([("prep", preprocessor), ("model", Ridge())])
pipe.fit(X_train, y_train)
```

## Encoding Cheat Sheet

| Method | When to use |
|--------|-------------|
| One-Hot | Nominal (no order), few categories (<20) |
| Ordinal | Ordered categories (small < medium < large) |
| Target | High-cardinality nominal (with CV to prevent leakage) |
| Embedding | Very high cardinality (deep learning) |

## Learning Resources
- [sklearn ColumnTransformer](https://scikit-learn.org/stable/modules/compose.html#columntransformer-for-heterogeneous-data)
- [category_encoders library](https://contrib.scikit-learn.org/category_encoders/)

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
