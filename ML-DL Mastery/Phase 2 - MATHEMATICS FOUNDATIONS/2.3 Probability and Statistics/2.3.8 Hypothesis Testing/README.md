# 2.3.8 Hypothesis Testing

One-sample t-test, two-sample A/B test, chi-squared independence test, Type I/II errors, power analysis.

---

## Files

| File | Description |
|------|-------------|
| `working_example.py` | Z-test by hand, critical region, rejection |
| `working_example2.py` | t-test (one/two-sample), chi-squared, Type I/II error simulation |
| `working_example.ipynb` | Interactive: t-test → A/B test → chi-squared → power |

## Quick Reference

```python
import scipy.stats as stats

# One-sample t-test
t, p = stats.ttest_1samp(data, popmean=mu0)

# Two-sample (A/B test)
t, p = stats.ttest_ind(group_a, group_b)

# Chi-squared test of independence
chi2, p, dof, expected = stats.chi2_contingency(observed_table)

# p-value interpretation
# p < 0.05 → reject H0 at 5% significance level
```

## Key Concepts

| Error | Description |
|-------|-------------|
| Type I (α) | False positive — reject true H0 |
| Type II (β) | False negative — fail to reject false H0 |
| Power | 1 - β = probability of detecting true effect |

## ML Connections
- **A/B testing** for model/feature significance
- **Permutation tests** for feature importance
- **Multiple testing correction** (Bonferroni) in feature selection

## Learning Resources
- [StatQuest: p-values explained](https://youtu.be/vemZtEM63GY)
- [Josh Starmer: Hypothesis testing series](https://www.youtube.com/@statquest)

Write unit tests for your functions with simple asserts.

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
