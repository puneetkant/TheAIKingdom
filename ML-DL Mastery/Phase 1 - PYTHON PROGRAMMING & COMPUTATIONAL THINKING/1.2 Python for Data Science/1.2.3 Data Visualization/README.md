# 1.2.3 Data Visualization

Matplotlib charts for ML: feature distributions, correlation heatmap, scatter plots, learning curves, confusion matrix.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | Basic matplotlib: line, scatter, bar, pie, subplots |
| `working_example2.py` | Cal Housing: distributions → correlation heatmap → learning curves → confusion matrix → saves PNGs |
| `working_example.ipynb` | Interactive: inline charts — distributions, scatter, heatmap, learning curves |

## Run

```bash
python working_example.py
python working_example2.py    # saves PNGs to output/
jupyter lab working_example.ipynb
```

## Chart Cheat Sheet

```python
import matplotlib.pyplot as plt

# Histogram
ax.hist(data, bins=40, color='steelblue', alpha=0.8)

# Scatter with colour map
ax.scatter(X, y, c=z, cmap='plasma', alpha=0.4, s=5)

# Heatmap
im = ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar(im)

# Learning curves
ax.plot(epochs, train_loss, label='Train')
ax.plot(epochs, val_loss, '--', label='Val')
ax.legend(); ax.grid(alpha=0.3)

# Subplots layout
fig, axes = plt.subplots(2, 3, figsize=(14, 8))

# Save
fig.savefig('output/fig.png', dpi=150, bbox_inches='tight')
```

## Seaborn (bonus library)
```python
import seaborn as sns
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
sns.pairplot(df, hue='Species')
```

## Datasets
- **California Housing** — [scikit-learn/california-housing on HuggingFace](https://huggingface.co/datasets/scikit-learn/california-housing)

## Learning Resources
- [Matplotlib docs](https://matplotlib.org/stable/contents.html)
- [Matplotlib cheat sheet](https://matplotlib.org/cheatsheets/)
- [Seaborn docs](https://seaborn.pydata.org/)
- [Real Python: Matplotlib guide](https://realpython.com/python-matplotlib-guide/)
- [Kaggle: Data Visualization micro-course](https://www.kaggle.com/learn/data-visualization)
- **Book:** *Python for Data Analysis* (McKinney) Ch. 9 (plotting)
- **Book:** *Fundamentals of Data Visualization* (Claus Wilke) — free online

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
