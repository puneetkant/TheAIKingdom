# 3.3.4 Association Rule Learning

Apriori algorithm, frequent itemsets, support/confidence/lift, market basket analysis.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | Apriori from scratch on toy transactions |
| `working_example2.py` | Apriori from scratch + mlxtend if available |
| `working_example.ipynb` | Interactive: itemsets → association rules → sorted by lift |

## Quick Reference

```python
# pip install mlxtend
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# One-hot encode transactions
df = pd.get_dummies(pd.DataFrame(transactions).stack()).groupby(level=0).sum().clip(0,1)
freq_items = apriori(df.astype(bool), min_support=0.3, use_colnames=True)
rules = association_rules(freq_items, metric="lift", min_threshold=1.0)
rules.sort_values("lift", ascending=False)
```

## Key Metrics

| Metric | Formula | Meaning |
|--------|---------|---------|
| Support | P(A∪B) | How often itemset appears |
| Confidence | P(B\|A) = sup(A∪B)/sup(A) | Rule reliability |
| Lift | conf/P(B) | How much better than random |

## ML Connections
- **Recommender systems** — "customers who bought X also bought Y"
- **Feature co-occurrence** — discover correlated features
- **Sequential patterns** — LSTM/Transformer pretraining for sequence tasks

## Learning Resources
- [mlxtend docs: Frequent Patterns](http://rasbt.github.io/mlxtend/user_guide/frequent_patterns/apriori/)
- [Agrawal & Srikant 1994 (original Apriori paper)](https://dl.acm.org/doi/10.5555/645920.672836)

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
