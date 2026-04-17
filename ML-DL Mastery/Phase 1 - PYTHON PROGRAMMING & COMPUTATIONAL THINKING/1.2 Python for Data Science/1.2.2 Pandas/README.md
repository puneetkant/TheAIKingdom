# 1.2.2 Pandas

Load, clean, engineer features, group, merge and query DataFrames — production data science workflows.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | Series/DataFrame basics, read_csv, indexing, apply |
| `working_example2.py` | Titanic: missing data → feature engineering → groupby → pivot → merge → query |
| `working_example.ipynb` | Interactive: full pipeline with charts |

## Run

```bash
python working_example.py
python working_example2.py    # downloads titanic.csv
jupyter lab working_example.ipynb
```

## Quick Reference

```python
import pandas as pd, numpy as np

df = pd.read_csv('titanic.csv')
df.shape, df.dtypes, df.describe()
df.isnull().sum()

# Missing data
df['Age'].fillna(df['Age'].median(), inplace=True)
df.dropna(subset=['Fare'])

# Feature engineering
df = df.assign(
    FamilySize = df['SibSp'] + df['Parch'] + 1,
    LogFare    = np.log1p(df['Fare']),
    AgeBin     = pd.cut(df['Age'], bins=[0,18,35,60,100]),
)

# GroupBy
df.groupby('Pclass')['Survived'].agg(['count','mean'])

# Merge
df.merge(class_info, on='Pclass', how='left')

# Query
df.query("Fare > 100 and Survived == 1")

# Method chaining
result = (df.pipe(clean).pipe(feature_eng).groupby('Pclass').agg(...))
```

## Dataset
- **Titanic** — [phihung/titanic on HuggingFace](https://huggingface.co/datasets/phihung/titanic)

## Learning Resources
- [Pandas docs](https://pandas.pydata.org/docs/)
- [10 minutes to pandas](https://pandas.pydata.org/docs/user_guide/10min.html)
- [Real Python: Pandas tutorial](https://realpython.com/pandas-dataframe/)
- [Kaggle Pandas micro-course](https://www.kaggle.com/learn/pandas)
- **Book:** *Python for Data Analysis* (Wes McKinney) — the definitive reference
- **Book:** *Effective Pandas* (Matt Harrison)

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
