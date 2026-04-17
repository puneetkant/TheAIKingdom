# 1.2.4 Exploratory Data Analysis (EDA)

Structured EDA pipeline: data quality report → univariate → bivariate → correlation → markdown report.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | Manual EDA with csv + statistics module |
| `working_example2.py` | Titanic: quality report, IQR outliers, group survival rates, charts, markdown EDA report |
| `working_example.ipynb` | Interactive: full EDA pipeline with visualisations |

## Run

```bash
python working_example.py
python working_example2.py    # downloads titanic.csv, saves output/
jupyter lab working_example.ipynb
```

## EDA Checklist

```python
# 1. Shape and types
df.shape, df.dtypes, df.info()

# 2. Missing data
df.isnull().sum()
df.isnull().mean() * 100   # % missing per column

# 3. Duplicates
df.duplicated().sum()

# 4. Univariate (numeric)
df.describe()           # count, mean, std, min, quartiles, max
df['Age'].hist(bins=30)
df['Age'].skew()        # skewness

# 5. Outliers (IQR)
Q1, Q3 = df['Fare'].quantile([0.25, 0.75])
IQR = Q3 - Q1
outliers = df[(df['Fare'] < Q1 - 1.5*IQR) | (df['Fare'] > Q3 + 1.5*IQR)]

# 6. Categorical distribution
df['Pclass'].value_counts()

# 7. Bivariate
df.groupby('Pclass')['Survived'].mean()
df.corr()

# 8. Visualise
df.hist(figsize=(14,10))
pd.plotting.scatter_matrix(df[numeric_cols])
```

## Dataset
- **Titanic** — [phihung/titanic on HuggingFace](https://huggingface.co/datasets/phihung/titanic)

## Learning Resources
- [Pandas profiling (ydata-profiling)](https://github.com/ydataai/ydata-profiling)
- [Kaggle EDA tutorials](https://www.kaggle.com/learn/data-visualization)
- [Real Python: Python statistics](https://realpython.com/python-statistics/)
- **Book:** *Python for Data Analysis* (McKinney) Ch. 7 (data cleaning)
- **Course:** [fast.ai Practical Data Ethics](https://ethics.fast.ai/) — EDA with fairness lens

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
